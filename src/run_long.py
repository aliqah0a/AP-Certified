"""Experiment B: nearby attacker and long-term drift (UJI long-term dataset).

Zones are the two library floors (3 and 5) with the same footprint, so every
impostor stands inside the same building, two floors from the claimed zone.

Protocol
- Fit on month-1 'trn' scans.
- Selection on month-1 'tst' files 1-2 (stratified halves).
- Month 1: calibration = 'tst' files 1-2, evaluation = 'tst' files 3-5.
- Months 2..25: evaluation = that month's 'tst' scans. Two thresholds:
  static (month-1 calibration) and recalibrated (conformal on that month's
  'trn' sweep, which a deployment would collect as a short re-enrollment).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from attacks import (beam_attack, gradient_attack_mlp, group_flip_cost,  # noqa: E402
                     min_cost_success, zone_profiles)
from core import (GroupVerifier, certified_far_randomized, conformal_tau,  # noqa: E402
                  frr_far, partition_random, partition_spread, eer)
from data_long import load_month, zones_from_coords  # noqa: E402
from experiments import (ALPHA, DELTAS, K_RANGE, baseline_scores,  # noqa: E402
                         evaluate_split, fit_baselines, group_claim_votes,
                         kstar_from_curve, partition_table, split_claims)

RES = Path(__file__).resolve().parents[1] / "results" / "long"
RES.mkdir(parents=True, exist_ok=True)
E = float(np.e)
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)


def load_tst_files(m):
    import glob
    from data_long import DB, _load
    files = sorted(glob.glob(str(DB / f"{m:02d}" / "tst*rss.csv")))
    return [_load([f]) for f in files]


def strat_half(y, rng):
    cal = []
    for z in np.unique(y):
        idx = rng.permutation(np.where(y == z)[0]); cal += list(idx[: len(idx) // 2])
    cal = np.array(sorted(cal))
    return cal, np.setdiff1d(np.arange(len(y)), cal)


def run(scheme="floor", n_attack=200, k_attack=6, months=range(1, 26), full=True, do_months=True):
    out = RES / scheme
    out.mkdir(exist_ok=True)
    Xtr, Ctr, _ = load_month(1, "trn")
    ytr = zones_from_coords(Ctr, scheme)
    Z = int(ytr.max()) + 1
    t1 = load_tst_files(1)
    Xs = np.vstack([t[0] for t in t1[:2]]); ys = zones_from_coords(np.vstack([t[1] for t in t1[:2]]), scheme)
    Xe1 = np.vstack([t[0] for t in t1[2:]]); ye1 = zones_from_coords(np.vstack([t[1] for t in t1[2:]]), scheme)

    # ---------------- selection on month-1 tst files 1-2 --------------------
    rng = np.random.default_rng(0)
    sc, se = strat_half(ys, rng)
    sel = []
    if (out / "selection.csv").exists():
        sel = pd.read_csv(out / "selection.csv").to_dict("records")
        Gs = []
    else:
        Gs = (5, 10, 15, 20)
    for G in Gs:
        for theta in (0.3, 0.5, 0.7):
            for power in (1.0, E):
                gv = GroupVerifier(partition_spread(Xtr, G), Z, 10, theta, power=power).fit(Xtr, ytr)
                V = gv.votes(Xs).astype(float)
                r = evaluate_split(V[sc], ys[sc], V[se], ys[se], rng)
                sel.append(dict(G=G, theta=theta, power=power, EER=r["EER"], FRR=r["FRR"],
                                kstar=kstar_from_curve(r["cert"])))
    sel = pd.DataFrame(sel); sel.to_csv(out / "selection.csv", index=False)
    b = sel.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
    cfg = dict(G=int(b.G), theta=float(b.theta), power=float(b.power))
    log(f"{scheme}: selected {cfg}")
    gv = GroupVerifier(partition_spread(Xtr, cfg["G"]), Z, 10, cfg["theta"], power=cfg["power"]).fit(Xtr, ytr)
    gvr = GroupVerifier(partition_random(Xtr.shape[1], cfg["G"], 0), Z, 10, cfg["theta"],
                        power=cfg["power"]).fit(Xtr, ytr)
    base = fit_baselines(Xtr, ytr)

    # static thresholds from month-1 calibration (tst files 1-2)
    Vs = gv.votes(Xs).astype(float)
    tau_static = conformal_tau(split_claims(Vs, ys)[0], ALPHA, np.random.default_rng(1))
    Bs = {n: baseline_scores(m, Xs, Z) for n, m in base.items()}
    tau_static_b = {n: conformal_tau(split_claims(S, ys)[0], ALPHA, np.random.default_rng(1), 1e-9)
                    for n, S in Bs.items()}

    rows, curves = [], []
    for m in ((months if full else [1]) if do_months else []):
        rng = np.random.default_rng(100 + m)
        if m == 1:
            Xc, yc, Xe, ye = Xs, ys, Xe1, ye1
        else:
            Xc, Cc, _ = load_month(m, "trn"); yc = zones_from_coords(Cc, scheme)
            Xe, Ce, _ = load_month(m, "tst"); ye = zones_from_coords(Ce, scheme)
        Vc, Ve = gv.votes(Xc).astype(float), gv.votes(Xe).astype(float)
        Vre = gvr.votes(Xe).astype(float); Vrc = gvr.votes(Xc).astype(float)
        extra = {}
        if m in (1, 5, 10, 15, 20, 25):
            sub = np.random.default_rng(m).choice(len(Xe), min(600, len(Xe)), replace=False)
            for dl in DELTAS[1:]:
                Vd = group_claim_votes(gv, Xe[sub], dl)
                extra[f"d{int(dl)}"] = split_claims(Vd, ye[sub])[1]
        for calib, tau in (("recalibrated", None), ("static", tau_static)):
            r = evaluate_split(Vc, yc, Ve, ye, rng, extra_imp=extra if calib == "recalibrated" else None,
                               tau=tau)
            row = dict(month=m, model="AP-Certified (spread)", calib=calib, tau=r["tau"], FRR=r["FRR"],
                       FAR=r["FAR"], EER=r["EER"], kstar=kstar_from_curve(r["cert"]))
            for name in extra if calib == "recalibrated" else []:
                row["kstar_" + name] = kstar_from_curve(r["cert_" + name])
            rows.append(row)
            for k in K_RANGE:
                curves.append(dict(month=m, calib=calib, delta=0, k=k, cert=r["cert"][k]))
                if calib == "recalibrated":
                    for name in extra:
                        curves.append(dict(month=m, calib=calib, delta=int(name[1:]), k=k,
                                           cert=r["cert_" + name][k]))
        if m > 1:
            # retrained: refit on month-1 trn + half of this month's sweep,
            # calibrate on the other half (still disjoint from evaluation)
            h1, h2 = strat_half(yc, np.random.default_rng(m))
            gvm = GroupVerifier(gv.groups, Z, 10, cfg["theta"], power=cfg["power"]).fit(
                np.vstack([Xtr, Xc[h1]]), np.r_[ytr, yc[h1]])
            Vmc, Vme = gvm.votes(Xc[h2]).astype(float), gvm.votes(Xe).astype(float)
            rt = evaluate_split(Vmc, yc[h2], Vme, ye, rng)
            rows.append(dict(month=m, model="AP-Certified (spread)", calib="retrained", tau=rt["tau"],
                             FRR=rt["FRR"], FAR=rt["FAR"], EER=rt["EER"], kstar=kstar_from_curve(rt["cert"])))
            for k in K_RANGE:
                curves.append(dict(month=m, calib="retrained", delta=0, k=k, cert=rt["cert"][k]))
        rr = evaluate_split(Vrc, yc, Vre, ye, rng)
        rows.append(dict(month=m, model="AP-Certified (random)", calib="recalibrated", tau=rr["tau"],
                         FRR=rr["FRR"], FAR=rr["FAR"], EER=rr["EER"], kstar=kstar_from_curve(rr["cert"])))
        for n, mdl in base.items():
            Sc_, Se_ = baseline_scores(mdl, Xc, Z), baseline_scores(mdl, Xe, Z)
            for calib, tau in (("recalibrated", None), ("static", tau_static_b[n])):
                rb = evaluate_split(Sc_, yc, Se_, ye, rng, jitter=1e-9, tau=tau)
                rows.append(dict(month=m, model=n, calib=calib, tau=rb["tau"], FRR=rb["FRR"],
                                 FAR=rb["FAR"], EER=rb["EER"]))
        log(f"{scheme} month {m}: " + ", ".join(
            f"{r_['model'][:14]}/{r_['calib'][:5]} FRR={r_['FRR']:.3f} FAR={r_['FAR']:.4f} k*={r_.get('kstar', '')}"
            for r_ in rows[-8:] if r_["model"].startswith("AP") or r_["model"] == "MLP"))
    if do_months:
        pd.DataFrame(rows).to_csv(out / "months.csv", index=False)
        pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)

    if full:
        if do_months:
            pd.DataFrame(partition_table(Xe1, Xtr, [3, 5, 10, 15, 20, 30])).to_csv(out / "partition.csv", index=False)
        # ---------------- attacks on month-1 evaluation ----------------------
        rng = np.random.default_rng(7)
        prof = zone_profiles(Xtr, ytr, Z)
        tau_g = conformal_tau(split_claims(Vs, ys)[0], ALPHA, rng)
        Ve1 = gv.votes(Xe1).astype(float)
        imp = [(i, z) for i in range(len(ye1)) for z in range(Z) if z != ye1[i]]
        pick = [imp[j] for j in rng.choice(len(imp), n_attack, replace=False)]
        att = []
        for n, mdl in base.items():
            t = tau_static_b[n]
            sc_ = lambda Xb, z, mdl=mdl: baseline_scores(mdl, Xb, Z)[:, z]
            ac_ = lambda Xb, z, mdl=mdl, t=t: baseline_scores(mdl, Xb, Z)[:, z] >= t
            S = np.zeros(k_attack + 1)
            for i, z in pick:
                s = beam_attack(sc_, ac_, Xe1[i], z, prof, k_attack)
                if n == "MLP":
                    s = s | gradient_attack_mlp(mdl, t, Xe1[i], z, k_attack)
                S += s
            att += [dict(model=n, k=k, success=S[k] / len(pick)) for k in range(k_attack + 1)]
            log(f"attack {n}: " + " ".join(f"{S[k]/len(pick):.3f}" for k in range(k_attack + 1)))
        Sx, Cp = np.zeros(k_attack + 1), np.zeros(k_attack + 1)
        for i, z in pick:
            u = rng.uniform()
            v0 = Ve1[i, z]
            cost = group_flip_cost(gv, Xe1[i], z, 2, rate=prof[0], need=max(0, int(np.ceil(tau_g - u - v0))))
            Sx += min_cost_success(cost, v0, tau_g, u, k_attack)
            Cp += np.array([np.clip(v0 + k + 1 - tau_g, 0, 1) for k in range(k_attack + 1)])
        for k in range(k_attack + 1):
            att.append(dict(model="AP-Certified", k=k, success=Sx[k] / len(pick), certified=Cp[k] / len(pick)))
        log("attack AP-Certified (success/certified): " + " ".join(
            f"{Sx[k]/len(pick):.3f}/{Cp[k]/len(pick):.3f}" for k in range(k_attack + 1)))
        pd.DataFrame(att).to_csv(out / "attacks.csv", index=False)
    (out / "config.json").write_text(json.dumps(dict(cfg=cfg, tau_static=float(tau_static)), indent=2))


if __name__ == "__main__":
    run("floor", full=True, do_months="attacks_only" not in sys.argv)
    run("quadrant", full=False)
    log("finished")
