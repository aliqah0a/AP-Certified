"""Pre-deployment certifiability check with a short deployment-condition sweep.

For each task: the selected hybrid verifier (from the earlier runs) is fitted
on the training data; the deployment calibration half (a sweep collected in
deployment conditions) gives the threshold AND an estimate of the certified
radius from its own impostor claims; the evaluation half gives the achieved
radius. Five splits per task.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import (GroupVerifier, certified_far_randomized, conformal_tau,  # noqa: E402
                  hybrid_scores, partition_spread)
from data import load_uji  # noqa: E402
from data_long import zones_from_coords  # noqa: E402
from data_more import load_mall, load_sod, load_tampere  # noqa: E402
from experiments import ALPHA, K_RANGE, kstar_from_curve, split_claims  # noqa: E402
from run_v3 import long_files, long_logins, strat_half_groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def best(sel):
    return sel.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]


def task_data():
    out = {}
    d = load_uji()
    s = best(pd.read_csv(R / "uji/selection_building.csv"))
    out["UJI building"] = (d["train"]["X"], d["train"]["building"], d["test"]["X"], d["test"]["building"],
                           np.arange(len(d["test"]["X"])).astype(str), 3, (int(s.G), float(s.theta), float(s.power), np.inf), s.kstar)
    for scheme, lab in (("floor", "Library floor"), ("quadrant", "Library half-floor")):
        Xf, yf, _ = long_logins(long_files(1, "trn"), 1, scheme)
        Xt, yt, gt = long_logins(long_files(1, "tst"), 1, scheme)
        s = best(pd.read_csv(R / f"v3/selection_long_{scheme}_m1.csv"))
        out[lab] = (Xf, yf, Xt, yt, gt, int(yf.max()) + 1, (int(s.G), float(s.theta), float(s.power), float(s.c)), s.kstar)
    for task, lab, fn in (("sod_cetc_floor", "SOD CETC331 floor", lambda: load_sod("CETC331", "floor")),
                          ("sod_syl_region", "SOD SYL region", lambda: load_sod("SYL", "region")),
                          ("sod_hcxy_region", "SOD HCXY region", lambda: load_sod("HCXY", "region")),
                          ("tampere_floor", "Tampere floor", load_tampere),
                          ("mall1_floor", "Mall 1 floor", lambda: load_mall(1)),
                          ("mall2_floor", "Mall 2 floor", lambda: load_mall(2))):
        dd = fn()
        s = best(pd.read_csv(R / f"more/selection_{task}_m1.csv"))
        out[lab] = (dd["train"]["X"], dd["train"]["y"], dd["test"]["X"], dd["test"]["y"], dd["test"]["g"], dd["Z"],
                    (int(s.G), float(s.theta), float(s.power), float(s.c)), s.kstar)
    return out


def main(splits=5):
    rows = []
    for lab, (Xf, yf, Xt, yt, gt, Z, (G, th, p, c), k_src) in task_data().items():
        gv = GroupVerifier(partition_spread(Xf, G, 0), Z, 10, th, power=p).fit(Xf, yf)
        S = hybrid_scores(gv.votes(Xt).astype(float), c)
        for r in range(splits):
            rs = np.random.default_rng(100 + r)
            cal, ev = strat_half_groups(yt, gt, rs)
            gc, ic = split_claims(S[cal], yt[cal])
            _, ie = split_claims(S[ev], yt[ev])
            tau = conformal_tau(gc, ALPHA, rs)
            k_cal = kstar_from_curve([certified_far_randomized(ic, tau, k) for k in K_RANGE])
            k_ev = kstar_from_curve([certified_far_randomized(ie, tau, k) for k in K_RANGE])
            rows.append(dict(setting=lab, split=r, source_kstar=float(k_src), sweep_kstar=k_cal, deploy_kstar=k_ev))
        d = pd.DataFrame(rows); d = d[d.setting == lab]
        print(f"{lab}: source {k_src}, sweep {d.sweep_kstar.mean():.1f}, deployment {d.deploy_kstar.mean():.1f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(R / "certifiability_sweep.csv", index=False)
    g = df.groupby("setting").mean(numeric_only=True)
    summ = dict(
        source=dict(pearson=float(np.corrcoef(g.source_kstar, g.deploy_kstar)[0, 1]),
                    mae=float((g.source_kstar - g.deploy_kstar).abs().mean()),
                    agree=float(((g.source_kstar >= 1) == (g.deploy_kstar >= 1)).mean())),
        sweep=dict(pearson=float(np.corrcoef(g.sweep_kstar, g.deploy_kstar)[0, 1]),
                   mae=float((g.sweep_kstar - g.deploy_kstar).abs().mean()),
                   agree=float(((g.sweep_kstar >= 1) == (g.deploy_kstar >= 1)).mean()),
                   per_split_agree=float(((df.sweep_kstar >= 1) == (df.deploy_kstar >= 1)).mean()),
                   per_split_exact_or_one=float(((df.sweep_kstar - df.deploy_kstar).abs() <= 1).mean())))
    (R / "certifiability_sweep.json").write_text(json.dumps(summ, indent=2))
    print(g.round(2).to_string()); print(summ)


if __name__ == "__main__":
    main()
