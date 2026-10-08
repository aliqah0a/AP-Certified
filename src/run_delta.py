"""Hybrid (k, delta) certificates with the combined (interval AND pairwise)
delta test, for UJI building zones and long-term floor zones.

Thresholds: same conformal protocol as run_uji.py / run_long.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import GroupVerifier, conformal_tau, certified_far_randomized, partition_spread  # noqa: E402
from data import load_uji  # noqa: E402
from data_long import load_month, zones_from_coords  # noqa: E402
from experiments import ALPHA, K_RANGE, kstar_from_curve, split_claims  # noqa: E402
from run_long import load_tst_files  # noqa: E402
from run_uji import strat_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results"
DELTAS = [1.0, 2.0, 4.0]
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)


def vote_bounds(gv, X, delta):
    P = gv.possible_accept(X, delta) & gv.possible_accept_pairwise(X, delta)
    return P.sum(1).astype(float)


def uji(R=10):
    cfg = json.loads((ROOT / "uji/config.json").read_text())["building"]
    d = load_uji(); tr, te = d["train"], d["test"]
    gv = GroupVerifier(partition_spread(tr["X"], cfg["G"]), 3, 10, cfg["theta"],
                       power=cfg["power"]).fit(tr["X"], tr["building"])
    y = te["building"]
    V0 = gv.votes(te["X"]).astype(float)
    Vd = {}
    for dl in DELTAS:
        Vd[dl] = vote_bounds(gv, te["X"], dl)
        log(f"uji delta={dl} done")
    rows = []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        cal, ev = strat_split(y, rng)
        tau = conformal_tau(split_claims(V0[cal], y[cal])[0], ALPHA, rng)
        for dl in [0.0] + DELTAS:
            V = V0 if dl == 0 else Vd[dl]
            imp = split_claims(V[ev], y[ev])[1]
            curve = [certified_far_randomized(imp, tau, k) for k in K_RANGE]
            rows.append(dict(split=r, delta=dl, kstar=kstar_from_curve(curve),
                             **{f"k{k}": curve[k] for k in K_RANGE}))
    pd.DataFrame(rows).to_csv(ROOT / "uji/delta.csv", index=False)


def long_floor():
    cfg = json.loads((ROOT / "long/floor/config.json").read_text())["cfg"]
    Xtr, Ctr, _ = load_month(1, "trn"); ytr = zones_from_coords(Ctr, "floor")
    gv = GroupVerifier(partition_spread(Xtr, cfg["G"]), 2, 10, cfg["theta"], power=cfg["power"]).fit(Xtr, ytr)
    t1 = load_tst_files(1)
    Xs = np.vstack([t[0] for t in t1[:2]]); ys = zones_from_coords(np.vstack([t[1] for t in t1[:2]]), "floor")
    Xe = np.vstack([t[0] for t in t1[2:]]); ye = zones_from_coords(np.vstack([t[1] for t in t1[2:]]), "floor")
    rng = np.random.default_rng(1)
    tau = conformal_tau(split_claims(gv.votes(Xs).astype(float), ys)[0], ALPHA, rng)
    sub = np.random.default_rng(0).choice(len(Xe), 800, replace=False)
    rows = []
    V0 = gv.votes(Xe[sub]).astype(float)
    for dl in [0.0] + DELTAS:
        V = V0 if dl == 0 else vote_bounds(gv, Xe[sub], dl)
        imp = split_claims(V, ye[sub])[1]
        curve = [certified_far_randomized(imp, tau, k) for k in K_RANGE]
        rows.append(dict(delta=dl, kstar=kstar_from_curve(curve), **{f"k{k}": curve[k] for k in K_RANGE}))
        log(f"long delta={dl} kstar={rows[-1]['kstar']}")
    pd.DataFrame(rows).to_csv(ROOT / "long/floor/delta.csv", index=False)


if __name__ == "__main__":
    uji()
    long_floor()
    log("finished")
