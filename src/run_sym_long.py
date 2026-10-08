"""Symmetric (FPA-style) rule vs one-sided rule on long-term floors, month 1."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from core import GroupVerifier, conformal_tau, partition_spread, frr_far, eer
from data_long import load_month, zones_from_coords
from experiments import ALPHA, K_RANGE, kstar_from_curve, split_claims
from run_long import load_tst_files

R = Path(__file__).resolve().parents[1] / "results/long/floor"
cfg = dict(G=10, theta=0.5, power=1.0)  # selected configuration (selection.csv)
Xtr, Ctr, _ = load_month(1, "trn"); ytr = zones_from_coords(Ctr, "floor")
t1 = load_tst_files(1)
Xs = np.vstack([t[0] for t in t1[:2]]); ys = zones_from_coords(np.vstack([t[1] for t in t1[:2]]), "floor")
Xe = np.vstack([t[0] for t in t1[2:]]); ye = zones_from_coords(np.vstack([t[1] for t in t1[2:]]), "floor")
rows = []
for theta in (0.6, 0.7):
    gv = GroupVerifier(partition_spread(Xtr, cfg["G"]), 2, 10, theta, power=cfg["power"]).fit(Xtr, ytr)
    def margin(V):
        return np.stack([V[:, z] - np.delete(V, z, 1).max(1) for z in range(V.shape[1])], 1).astype(float)
    Mc, Me = margin(gv.votes(Xs)), margin(gv.votes(Xe))
    rng = np.random.default_rng(3)
    tau = conformal_tau(split_claims(Mc, ys)[0], ALPHA, rng)
    g, i = split_claims(Me, ye)
    frr, far = frr_far(g, i, tau, rng)
    curve = [float(np.clip(i + 2 * k + 1 - tau, 0, 1).mean()) for k in K_RANGE]
    rows.append(dict(rule="symmetric", theta=theta, FRR=frr, FAR=far, EER=eer(g, i), kstar=kstar_from_curve(curve)))
pd.DataFrame(rows).to_csv(R / "symmetric_month1.csv", index=False)
print(pd.DataFrame(rows))
