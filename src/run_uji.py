"""Experiment A (UJIIndoorLoc, building zones) and the UJI floor-level limit.

Protocol
1. Configuration selection on SOURCE data only: fit on training phones minus a
   held-out 20 %, evaluate candidate configurations on the held-out phones.
2. Refit the selected configuration on the full training set.
3. Deployment calibration: split the validation set (new phones, later
   period) R times into calibration / evaluation halves, stratified by zone.
   Conformal thresholds come from the calibration half only.
4. Source calibration (threshold from held-out training phones) is reported
   for contrast: it violates exchangeability with the deployment data.
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
from core import (GroupVerifier, HierarchicalVerifier, conformal_tau,  # noqa: E402
                  partition_random, partition_spread)
from data import load_uji, split_fit_calib  # noqa: E402
from experiments import (ALPHA, DELTAS, K_RANGE, baseline_scores,  # noqa: E402
                         evaluate_split, fit_baselines, group_claim_votes,
                         kstar_from_curve, partition_table, split_claims)

RES = Path(__file__).resolve().parents[1] / "results" / "uji"
RES.mkdir(parents=True, exist_ok=True)
E = float(np.e)
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)


def strat_split(y, rng):
    cal = []
    for z in np.unique(y):
        idx = rng.permutation(np.where(y == z)[0])
        cal += list(idx[: len(idx) // 2])
    cal = np.array(sorted(cal))
    ev = np.setdiff1d(np.arange(len(y)), cal)
    return cal, ev


def select_building(d):
    if (RES / "selection_building.csv").exists():
        df = pd.read_csv(RES / "selection_building.csv")
        best = df.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
        return dict(G=int(best.G), theta=float(best.theta), power=float(best.power))
    fit, hold, _ = split_fit_calib(d["train"], seed=0)
    rng = np.random.default_rng(0)
    cal, ev = strat_split(hold["building"], rng)
    rows = []
    for G in (5, 10, 15, 20):
        for theta in (0.3, 0.5, 0.7):
            for power in (1.0, E):
                gv = GroupVerifier(partition_spread(fit["X"], G), 3, 10, theta, power=power)
                gv.fit(fit["X"], fit["building"])
                V = gv.votes(hold["X"]).astype(float)
                r = evaluate_split(V[cal], hold["building"][cal], V[ev], hold["building"][ev], rng)
                rows.append(dict(G=G, theta=theta, power=power, EER=r["EER"], FRR=r["FRR"],
                                 kstar=kstar_from_curve(r["cert"])))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "selection_building.csv", index=False)
    best = df.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
    log(f"selected building config: {best.to_dict()}")
    return dict(G=int(best.G), theta=float(best.theta), power=float(best.power))


def main(R=10, n_attack=200, k_attack=6, stages=("splits", "partition", "attacks", "floor")):
    d = load_uji()
    tr, te = d["train"], d["test"]
    yb_tr, yb_te = tr["building"], te["building"]
    cfg = select_building(d)

    # ---------------- final models on full training data ---------------------
    gv = GroupVerifier(partition_spread(tr["X"], cfg["G"]), 3, 10, cfg["theta"],
                       power=cfg["power"]).fit(tr["X"], yb_tr)
    gv_rand = GroupVerifier(partition_random(tr["X"].shape[1], cfg["G"], 0), 3, 10, cfg["theta"],
                            power=cfg["power"]).fit(tr["X"], yb_tr)
    gv_sym = GroupVerifier(partition_spread(tr["X"], cfg["G"]), 3, 10, max(cfg["theta"], 0.6),
                           power=cfg["power"]).fit(tr["X"], yb_tr)
    base = fit_baselines(tr["X"], yb_tr)
    log("models fitted")

    V = {0.0: group_claim_votes(gv, te["X"])}
    for dl in (DELTAS[1:] if "splits" in stages else []):
        V[dl] = group_claim_votes(gv, te["X"], dl)
        log(f"delta={dl} votes done")
    Vr = group_claim_votes(gv_rand, te["X"])
    Vs = group_claim_votes(gv_sym, te["X"])
    # symmetric (FPA-style) margin score and its certified bound
    def margin(Vm):
        M = np.zeros_like(Vm)
        for z in range(Vm.shape[1]):
            o = np.delete(Vm, z, axis=1)
            M[:, z] = Vm[:, z] - o.max(1)
        return M
    Ms = margin(Vs)
    B = {n: baseline_scores(m, te["X"], 3) for n, m in base.items()}

    # source calibration thresholds (held-out training phones, seed 0)
    fit0, hold0, _ = split_fit_calib(tr, seed=0)
    gv_src = GroupVerifier(partition_spread(fit0["X"], cfg["G"]), 3, 10, cfg["theta"],
                           power=cfg["power"]).fit(fit0["X"], fit0["building"])
    Vh = gv_src.votes(hold0["X"]).astype(float)
    Vt_src = gv_src.votes(te["X"]).astype(float)

    rows, curves = [], []
    for r in (range(R) if "splits" in stages else []):
        rng = np.random.default_rng(1000 + r)
        cal, ev = strat_split(yb_te, rng)
        # AP-Certified with delta curves
        extra = {f"d{int(dl)}": split_claims(V[dl][ev], yb_te[ev])[1] for dl in DELTAS[1:]}
        res = evaluate_split(V[0.0][cal], yb_te[cal], V[0.0][ev], yb_te[ev], rng,
                             mondrian_groups=(yb_te[cal], yb_te[ev]), extra_imp=extra)
        rows.append(dict(split=r, model="AP-Certified (spread)", calib="deployment", **{
            k: res[k] for k in ("tau", "FRR", "FAR", "EER")}, kstar=kstar_from_curve(res["cert"]),
            **{f"kstar_d{int(dl)}": kstar_from_curve(res[f"cert_d{int(dl)}"]) for dl in DELTAS[1:]},
            mondrian=json.dumps(res["mondrian_FRR"])))
        for k in K_RANGE:
            curves.append(dict(split=r, model="AP-Certified (spread)", delta=0, k=k, cert=res["cert"][k]))
            for dl in DELTAS[1:]:
                curves.append(dict(split=r, model="AP-Certified (spread)", delta=dl, k=k,
                                   cert=res[f"cert_d{int(dl)}"][k]))
        # random partition
        rr = evaluate_split(Vr[cal], yb_te[cal], Vr[ev], yb_te[ev], rng)
        rows.append(dict(split=r, model="AP-Certified (random)", calib="deployment",
                         **{k: rr[k] for k in ("tau", "FRR", "FAR", "EER")},
                         kstar=kstar_from_curve(rr["cert"])))
        for k in K_RANGE:
            curves.append(dict(split=r, model="AP-Certified (random)", delta=0, k=k, cert=rr["cert"][k]))
        # symmetric FPA-style: margin score; k changed groups shift margin by <= 2k
        rs = evaluate_split(Ms[cal], yb_te[cal], Ms[ev], yb_te[ev], rng)
        _, imp_m = split_claims(Ms[ev], yb_te[ev])
        cert_s = [float(np.clip(imp_m + 2 * k + 1 - rs["tau"], 0, 1).mean()) for k in K_RANGE]
        rows.append(dict(split=r, model="FPA-style symmetric", calib="deployment",
                         **{k: rs[k] for k in ("tau", "FRR", "FAR", "EER")},
                         kstar=kstar_from_curve(cert_s)))
        for k in K_RANGE:
            curves.append(dict(split=r, model="FPA-style symmetric", delta=0, k=k, cert=cert_s[k]))
        # baselines
        for name, S in B.items():
            rb = evaluate_split(S[cal], yb_te[cal], S[ev], yb_te[ev], rng, jitter=1e-9)
            rows.append(dict(split=r, model=name, calib="deployment",
                             **{k: rb[k] for k in ("tau", "FRR", "FAR", "EER")}, kstar=np.nan))
        # source calibration (threshold from held-out training phones)
        tau_src = conformal_tau(split_claims(Vh, hold0["building"])[0], ALPHA, rng)
        rsrc = evaluate_split(Vt_src[cal], yb_te[cal], Vt_src[ev], yb_te[ev], rng, tau=tau_src)
        rows.append(dict(split=r, model="AP-Certified (spread)", calib="source",
                         **{k: rsrc[k] for k in ("tau", "FRR", "FAR", "EER")},
                         kstar=kstar_from_curve(rsrc["cert"])))
    if "splits" in stages:
        pd.DataFrame(rows).to_csv(RES / "splits.csv", index=False)
        pd.DataFrame(curves).to_csv(RES / "curves.csv", index=False)
        log("splits done")

    # ---------------- partition analysis --------------------------------------
    if "partition" in stages:
        pt = pd.DataFrame(partition_table(te["X"], tr["X"], [3, 5, 10, 15, 20, 30]))
        pt.to_csv(RES / "partition.csv", index=False)
        log("partition table done")
    if "attacks" in stages:
        run_attacks(gv, base, B, V, tr, te, yb_tr, yb_te, n_attack, k_attack)
    if "floor" in stages:
        run_floor(cfg, tr, te, R)
    log("finished")


def run_attacks(gv, base, B, V, tr, te, yb_tr, yb_te, n_attack, k_attack):

    # ---------------- attacks (split 0) -----------------------------------------
    rng = np.random.default_rng(1000)
    cal, ev = strat_split(yb_te, rng)
    prof = zone_profiles(tr["X"], yb_tr, 3)
    tau_g = conformal_tau(split_claims(V[0.0][cal], yb_te[cal])[0], ALPHA, rng)
    taus = {n: conformal_tau(split_claims(S[cal], yb_te[cal])[0], ALPHA, rng, 1e-9) for n, S in B.items()}
    imp_pairs = [(i, z) for i in ev for z in range(3) if z != yb_te[i]]
    pick = [imp_pairs[j] for j in rng.choice(len(imp_pairs), n_attack, replace=False)]
    att = []
    for name, m in base.items():
        sc = lambda Xb, z, m=m: baseline_scores(m, Xb, 3)[:, z]
        acc = lambda Xb, z, m=m, t=taus[name]: baseline_scores(m, Xb, 3)[:, z] >= t
        S = np.zeros(k_attack + 1)
        for i, z in pick:
            s = beam_attack(sc, acc, te["X"][i], z, prof, k_attack)
            if name == "MLP":
                s = s | gradient_attack_mlp(m, taus[name], te["X"][i], z, k_attack)
            S += s
        att += [dict(model=name, k=k, success=S[k] / len(pick)) for k in range(k_attack + 1)]
        log(f"attack {name}: " + " ".join(f"{S[k]/len(pick):.3f}" for k in range(k_attack + 1)))
    # AP-Certified: exact minimum-cost search over the value grid (each AP
    # touches one group, so the cheapest groups are flipped first)
    Sx, cert_pick = np.zeros(k_attack + 1), np.zeros(k_attack + 1)
    for i, z in pick:
        u = rng.uniform()
        v0 = V[0.0][i, z]
        cost = group_flip_cost(gv, te["X"][i], z, max_cost=2, rate=prof[0],
                               need=max(0, int(np.ceil(tau_g - u - v0))))
        Sx += min_cost_success(cost, v0, tau_g, u, k_attack)
        cert_pick += np.array([np.clip(v0 + k + 1 - tau_g, 0, 1) for k in range(k_attack + 1)])
    for k in range(k_attack + 1):
        att.append(dict(model="AP-Certified", k=k, success=Sx[k] / len(pick),
                        certified=cert_pick[k] / len(pick)))
    log("attack AP-Certified (success/certified): " + " ".join(
        f"{Sx[k]/len(pick):.3f}/{cert_pick[k]/len(pick):.3f}" for k in range(k_attack + 1)))
    pd.DataFrame(att).to_csv(RES / "attacks.csv", index=False)


def run_floor(cfg, tr, te, R):
    # ---------------- floor-level limit: hierarchical, selected on source data --
    fl_tr, fl_te = tr["zone"] % 10, te["zone"] % 10
    fit, hold, _ = split_fit_calib(tr, seed=0)
    sel = []
    zones = sorted({(b, f) for b, f in zip(tr["building"], fl_tr)})
    zi = {z: i for i, z in enumerate(zones)}
    def hscores(hv, X):
        S = hv.claim_scores(X)
        return np.stack([S[:, b, f] for b, f in zones], 1)
    hy = lambda D: np.array([zi[(b, f)] for b, f in zip(D["building"], D["zone"] % 10)])
    rng = np.random.default_rng(0)
    hc, he = strat_split(hy(hold), rng)
    for GF in (5, 6, 8, 10):
        for tF in (0.2, 0.3, 0.4):
            hv = HierarchicalVerifier(cfg["G"], GF, 10, cfg["theta"], tF, power=E).fit(
                fit["X"], fit["building"], fit["zone"] % 10)
            S = hscores(hv, hold["X"])
            r = evaluate_split(S[hc], hy(hold)[hc], S[he], hy(hold)[he], rng)
            sel.append(dict(GF=GF, thetaF=tF, EER=r["EER"], kstar=kstar_from_curve(r["cert"]),
                            cert1=r["cert"][1]))
    sel = pd.DataFrame(sel); sel.to_csv(RES / "selection_floor.csv", index=False)
    b = sel.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
    hv = HierarchicalVerifier(cfg["G"], int(b.GF), 10, cfg["theta"], float(b.thetaF), power=E).fit(
        tr["X"], tr["building"], fl_tr)
    S = hscores(hv, te["X"]); yz = hy(te)
    bf = fit_baselines(tr["X"], hy(tr))
    Bf = {n: baseline_scores(m, te["X"], len(zones)) for n, m in bf.items()}
    frows = []
    for r in range(R):
        rng = np.random.default_rng(2000 + r)
        cal, ev = strat_split(yz, rng)
        res = evaluate_split(S[cal], yz[cal], S[ev], yz[ev], rng)
        same_b = np.array([[zones[j][0] == te["building"][i] for j in range(len(zones))] for i in ev])
        G = np.zeros((len(ev), len(zones)), bool); G[np.arange(len(ev)), yz[ev]] = True
        hard = S[ev][(~G) & same_b]
        from core import certified_far_randomized
        frows.append(dict(split=r, model="AP-Certified (hierarchical)", FRR=res["FRR"], FAR=res["FAR"],
                          EER=res["EER"], kstar=kstar_from_curve(res["cert"]),
                          FAR_adjacent=certified_far_randomized(hard, res["tau"], 0),
                          cert1=res["cert"][1], cert2=res["cert"][2]))
        for name, Sb in Bf.items():
            rb = evaluate_split(Sb[cal], yz[cal], Sb[ev], yz[ev], rng, jitter=1e-9)
            frows.append(dict(split=r, model=name, FRR=rb["FRR"], FAR=rb["FAR"], EER=rb["EER"]))
    pd.DataFrame(frows).to_csv(RES / "floor_limit.csv", index=False)
    (RES / "config.json").write_text(json.dumps(dict(building=cfg, floor=dict(GF=int(b.GF),
                                     thetaF=float(b.thetaF), power=E), R=R, alpha=ALPHA,
                                     runtime_s=time.time() - T0), indent=2))


if __name__ == "__main__":
    st = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("splits", "partition", "attacks", "floor")
    main(stages=st)
