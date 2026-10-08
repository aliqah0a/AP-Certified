"""AP-Certified on the additional datasets (SODIndoorLoc, Tampere, two malls).

Protocol (same for every task)
- Selection: hold out 20 % of the training position groups (stratified by
  zone); fit candidates on the rest; select per rule family on the held-out
  groups (calibration and evaluation halves of that set).
- Refit selected configurations on the full training set.
- Deployment calibration: R splits of the test position groups into
  calibration / evaluation halves (stratified by zone).
- Attacks on split 0: group-flip greedy attack on the hybrid verifier (lower
  bound), beam search on LightGBM and MLP, gradient attack on the MLP.

Usage: python3 src/run_more.py <task> [<task> ...]
Tasks: sod_cetc_floor, sod_syl_region, sod_hcxy_region, tampere_floor, mall1_floor, mall2_floor
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from attacks import (beam_attack, cascade_greedy_attack, gradient_attack_mlp,  # noqa: E402
                     group_single_edits, zone_profiles)
from core import (GroupVerifier, certified_far_randomized, conformal_tau, eer,  # noqa: E402
                  frr_far, hybrid_scores, merge_scans, partition_spread)
from data_more import load_mall, load_sod, load_tampere  # noqa: E402
from experiments import (ALPHA, K_RANGE, baseline_scores, fit_baselines,  # noqa: E402
                         kstar_from_curve, split_claims)
from run_v3 import C_GRID, strat_half_groups  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results" / "more"
RES.mkdir(parents=True, exist_ok=True)
E = float(np.e)
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)

TASKS = {
    "sod_cetc_floor": lambda: load_sod("CETC331", "floor"),
    "sod_syl_region": lambda: load_sod("SYL", "region"),
    "sod_hcxy_region": lambda: load_sod("HCXY", "region"),
    "tampere_floor": load_tampere,
    "mall1_floor": lambda: load_mall(1),
    "mall2_floor": lambda: load_mall(2),
}
MULTI = {"sod_cetc_floor", "sod_syl_region", "sod_hcxy_region"}


def sod_logins(D, m):
    """Merge m consecutive samples (by SampleTimes) at one position and device."""
    X, y, g = [], [], []
    keys = np.array([f"{a}|{b}" for a, b in zip(D["g"], D["dev"])])
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        idx = idx[np.argsort(D["order"][idx])]
        for s in range(0, len(idx) - m + 1, m):
            w = idx[s:s + m]
            X.append(merge_scans(D["X"][w])); y.append(D["y"][w[0]]); g.append(D["g"][w[0]])
    return dict(X=np.array(X), y=np.array(y), g=np.array(g))


def eval_scores(Sc, yc, Se, ye, rng):
    gc, _ = split_claims(Sc, yc)
    ge, ie = split_claims(Se, ye)
    tau = conformal_tau(gc, ALPHA, rng)
    frr, far = frr_far(ge, ie, tau, rng)
    curve = [certified_far_randomized(ie, tau, k) for k in K_RANGE]
    return dict(tau=tau, FRR=frr, FAR=far, EER=eer(ge, ie), kstar=kstar_from_curve(curve), curve=curve)


def select(Xf, yf, Xs, ys, gs, Z, tag):
    rng = np.random.default_rng(0)
    sc, se = strat_half_groups(ys, gs, rng)
    rows = []
    for G, th, p in itertools.product((5, 10, 15, 20), (0.6, 0.7), (1.0, E)):
        gv = GroupVerifier(partition_spread(Xf, G, 0), Z, 10, th, power=p).fit(Xf, yf)
        V = gv.votes(Xs).astype(float)
        for c in C_GRID:
            S = hybrid_scores(V, c)
            r = eval_scores(S[sc], ys[sc], S[se], ys[se], rng)
            rows.append(dict(G=G, theta=th, power=p, c=c, kstar=r["kstar"], EER=r["EER"]))
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"selection_{tag}.csv", index=False)
    fam = {"one-sided": df[np.isposinf(df.c)], "symmetric": df[np.isneginf(df.c)], "hybrid": df}
    return {n: d.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0].to_dict() for n, d in fam.items()}


def run_task(task, m=1, R=5, n_attack=100, k_attack=6, attacks=True):
    d = TASKS[task]()
    tr, te, Z = d["train"], d["test"], d["Z"]
    if m > 1:
        tr, te = sod_logins(tr, m), sod_logins(te, m)
        if len(tr["y"]) == 0 or len(te["y"]) == 0:
            log(f"{task} m={m}: not enough repeated samples per position; skipped")
            return pd.DataFrame(), pd.DataFrame()
    tag = f"{task}_m{m}"
    # ---- selection on held-out training groups
    rng = np.random.default_rng(0)
    sel_idx = np.zeros(len(tr["y"]), bool)
    for z in np.unique(tr["y"]):
        gs = np.unique(tr["g"][tr["y"] == z])
        pick = rng.permutation(gs)[: max(1, int(0.2 * len(gs)))]
        sel_idx |= np.isin(tr["g"], pick) & (tr["y"] == z)
    best = select(tr["X"][~sel_idx], tr["y"][~sel_idx], tr["X"][sel_idx], tr["y"][sel_idx],
                  tr["g"][sel_idx], Z, tag)
    log(f"{tag}: selected " + "; ".join(f"{k}: G={int(v['G'])} th={v['theta']} p={v['power']:.2f} c={v['c']}"
                                         for k, v in best.items()))
    # ---- final models
    models = {}
    for fam, b in best.items():
        key = (int(b["G"]), float(b["theta"]), float(b["power"]))
        if key not in models:
            models[key] = GroupVerifier(partition_spread(tr["X"], key[0], 0), Z, 10, key[1],
                                        power=key[2]).fit(tr["X"], tr["y"])
    V = {key: gv.votes(te["X"]).astype(float) for key, gv in models.items()}
    base = fit_baselines(tr["X"], tr["y"]) if m == 1 else {}
    B = {n: baseline_scores(mdl, te["X"], Z) for n, mdl in base.items()}
    rows = []
    for r in range(R):
        rs = np.random.default_rng(100 + r)
        cal, ev = strat_half_groups(te["y"], te["g"], rs)
        for fam, b in best.items():
            key = (int(b["G"]), float(b["theta"]), float(b["power"]))
            S = hybrid_scores(V[key], b["c"])
            o = eval_scores(S[cal], te["y"][cal], S[ev], te["y"][ev], rs)
            rows.append(dict(task=task, m=m, split=r, model=f"AP-Certified ({fam})", FRR=o["FRR"],
                             FAR=o["FAR"], EER=o["EER"], kstar=o["kstar"], G=key[0], theta=key[1],
                             power=key[2], c=b["c"], **{f"k{k}": o["curve"][k] for k in K_RANGE}))
        for n, S in B.items():
            gc, _ = split_claims(S[cal], te["y"][cal]); ge, ie = split_claims(S[ev], te["y"][ev])
            t = conformal_tau(gc, ALPHA, rs, 1e-9)
            f1, f2 = frr_far(ge, ie, t, rs, 1e-9)
            rows.append(dict(task=task, m=m, split=r, model=n, FRR=f1, FAR=f2, EER=eer(ge, ie)))
    df = pd.DataFrame(rows)
    g = df.groupby("model")[["FRR", "FAR", "EER", "kstar"]].mean()
    log(f"{tag}:\n{g.round(4).to_string()}")
    # ---- attacks (split 0)
    att = []
    if attacks and m == 1:
        rs = np.random.default_rng(100)
        cal, ev = strat_half_groups(te["y"], te["g"], rs)
        prof = zone_profiles(tr["X"], tr["y"], Z)
        b = best["hybrid"]; key = (int(b["G"]), float(b["theta"]), float(b["power"])); gv = models[key]
        S = hybrid_scores(V[key], b["c"])
        tau = conformal_tau(split_claims(S[cal], te["y"][cal])[0], ALPHA, rs)
        imp = [(i, z) for i in ev for z in range(Z) if z != te["y"][i]]
        pick = [imp[j] for j in rs.choice(len(imp), min(n_attack, len(imp)), replace=False)]
        succ, cert = np.zeros(k_attack + 1), np.zeros(k_attack + 1)
        for i, z in pick:
            u = rs.uniform()
            acc = lambda X, z, u=u: hybrid_scores(gv.votes(X).astype(float), b["c"])[:, z] + u >= tau
            ed = group_single_edits(gv, te["X"][i], z, rate=prof[0])
            succ += cascade_greedy_attack(ed, te["X"][i], z, acc, lambda X, z: gv.soft_score(X, z), k_attack)
            cert += np.array([np.clip(S[i, z] + k + 1 - tau, 0, 1) for k in range(k_attack + 1)])
        att += [dict(task=task, model="AP-Certified (hybrid)", k=k, success=succ[k] / len(pick),
                     certified=cert[k] / len(pick)) for k in range(k_attack + 1)]
        for n in ("LightGBM", "MLP"):
            mdl = base[n]
            t = conformal_tau(split_claims(B[n][cal], te["y"][cal])[0], ALPHA, rs, 1e-9)
            sc_ = lambda X, z, mdl=mdl: baseline_scores(mdl, X, Z)[:, z]
            ac_ = lambda X, z, mdl=mdl, t=t: baseline_scores(mdl, X, Z)[:, z] >= t
            s = np.zeros(k_attack + 1)
            for i, z in pick:
                o = beam_attack(sc_, ac_, te["X"][i], z, prof, k_attack, width=3)
                if n == "MLP":
                    o = o | gradient_attack_mlp(mdl, t, te["X"][i], z, k_attack)
                s += o
            att += [dict(task=task, model=n, k=k, success=s[k] / len(pick)) for k in range(k_attack + 1)]
        a = pd.DataFrame(att)
        log(f"{tag} attacks:\n{a.pivot_table(index='k', columns='model', values='success').round(3).to_string()}")
    return df, pd.DataFrame(att)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    skip_m1 = "--skip-m1" in sys.argv
    for task in args:
        ms = (1, 3, 5) if task in MULTI else (1,)
        if skip_m1:
            ms = tuple(x for x in ms if x > 1)
        for m in ms:
            df, att = run_task(task, m=m)
            if len(df):
                df.to_csv(RES / f"{task}_m{m}.csv", index=False)
            if len(att):
                att.to_csv(RES / f"{task}_attacks.csv", index=False)
    log("finished")
