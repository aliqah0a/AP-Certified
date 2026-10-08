"""Recompute partition coverage tables (Proposition 1 quantile convention)
and measure RSSI fluctuation at fixed positions (long-term, month 1)."""
import collections, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from data import load_uji
from data_long import load_month
from experiments import partition_table
from run_long import load_tst_files

R = Path(__file__).resolve().parents[1] / "results"
d = load_uji()
pd.DataFrame(partition_table(d["test"]["X"], d["train"]["X"], [3, 5, 10, 15, 20, 30])).to_csv(R / "uji/partition.csv", index=False)
Xtr, Ctr, _ = load_month(1, "trn")
t1 = load_tst_files(1)
Xe1 = np.vstack([t[0] for t in t1[2:]])
pd.DataFrame(partition_table(Xe1, Xtr, [3, 5, 10, 15, 20, 30])).to_csv(R / "long/floor/partition.csv", index=False)

grp = collections.defaultdict(list)
for i, c in enumerate(Ctr):
    grp[tuple(c)].append(i)
sds, rng = [], []
for idx in grp.values():
    S = Xtr[idx]
    for j in range(S.shape[1]):
        v = S[:, j]; v = v[v > 0]
        if len(v) >= 5:
            sds.append(v.std()); rng.append(np.percentile(v, 95) - np.percentile(v, 5))
out = dict(positions=len(grp), scans_per_position=float(np.mean([len(v) for v in grp.values()])),
           median_std_dB=float(np.median(sds)), p90_std_dB=float(np.percentile(sds, 90)),
           median_5_95_range_dB=float(np.median(rng)))
(R / "long/rssi_fluctuation.json").write_text(json.dumps(out, indent=2))
print(out)
