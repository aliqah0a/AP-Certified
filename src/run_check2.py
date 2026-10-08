"""Certifiability check with sweeps of different sizes, and population-level
(Clopper-Pearson) certified radii.

For each setting and split, the test positions are halved (calibration pool /
deployment data). A sweep of a fraction f of all test positions is drawn from
the calibration pool, stratified by zone. The sweep sets the conformal
threshold and estimates the radius (point estimate and Clopper-Pearson lower
estimate); the deployment half gives the achieved FRR and radius at that
threshold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import (GroupVerifier, certified_far_randomized, certified_far_upper,  # noqa: E402
                  conformal_tau, hybrid_scores, partition_spread)
from experiments import ALPHA, FAR_BUDGET, K_RANGE, kstar_from_curve, split_claims  # noqa: E402
from run_check import task_data  # noqa: E402
from run_v3 import strat_half_groups  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
FRACS = (0.05, 0.10, 0.25, 0.50)


def imp_with_groups(S, y, g):
    """Impostor scores and the position id of each impostor claim."""
    Z = S.shape[1]
    mask = np.ones_like(S, bool); mask[np.arange(len(y)), y] = False
    gg = np.repeat(g[:, None], Z, 1)
    return S[mask], gg[mask]


def kstar_upper(ups):
    return kstar_from_curve(ups)


def sub_positions(idx, y, g, frac_of_pool, rng):
    keep = []
    for z in np.unique(y[idx]):
        ii = idx[y[idx] == z]
        gs = np.unique(g[ii])
        n = max(1, int(round(frac_of_pool * len(gs))))
        pick = rng.permutation(gs)[:n]
        keep.append(ii[np.isin(g[ii], pick)])
    return np.sort(np.concatenate(keep))


def main(splits=5):
    rows = []
    for lab, (Xf, yf, Xt, yt, gt, Z, (G, th, p, c), _) in task_data().items():
        gv = GroupVerifier(partition_spread(Xf, G, 0), Z, 10, th, power=p).fit(Xf, yf)
        S = hybrid_scores(gv.votes(Xt).astype(float), c)
        gt = np.asarray(gt).astype(str)
        npos = len(np.unique(gt))
        for r in range(splits):
            rs = np.random.default_rng(100 + r)
            pool, ev = strat_half_groups(yt, gt, rs)
            ge, _ = split_claims(S[ev], yt[ev])
            ie, ige = imp_with_groups(S[ev], yt[ev], gt[ev])
            for f in FRACS:
                sw = sub_positions(pool, yt, gt, min(1.0, f / 0.5), rs)
                gc, ic = split_claims(S[sw], yt[sw])
                _, igc = imp_with_groups(S[sw], yt[sw], gt[sw])
                tau = conformal_tau(gc, ALPHA, rs)
                k_sw = kstar_from_curve([certified_far_randomized(ic, tau, k) for k in K_RANGE])
                k_sw_cp = kstar_upper(certified_far_upper(ic, tau, K_RANGE, rs))
                k_dep = kstar_from_curve([certified_far_randomized(ie, tau, k) for k in K_RANGE])
                k_dep_cp = kstar_upper(certified_far_upper(ie, tau, K_RANGE, rs))
                k_dep_cp_pos = kstar_upper(certified_far_upper(ie, tau, K_RANGE, rs, groups=ige))
                frr = float((ge + rs.uniform(0, 1, len(ge)) < tau).mean())
                rows.append(dict(setting=lab, split=r, frac=f, n_pos_sweep=len(np.unique(gt[sw])), n_pos=npos,
                                 n_gen_sweep=len(gc), n_imp_sweep=len(ic), n_imp_dep=len(ie),
                                 n_pos_dep=len(np.unique(ige)), tau=tau, FRR_dep=frr,
                                 k_sweep=k_sw, k_sweep_cp=k_sw_cp, k_dep=k_dep, k_dep_cp=k_dep_cp,
                                 k_dep_cp_pos=k_dep_cp_pos))
        d = pd.DataFrame(rows); d = d[d.setting == lab]
        print(lab, d.groupby("frac")[["k_sweep", "k_sweep_cp", "k_dep", "k_dep_cp", "k_dep_cp_pos", "FRR_dep",
                                      "n_pos_sweep"]].mean().round(2).to_string(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "check_sweepsize.csv", index=False)


if __name__ == "__main__":
    main()
