"""Dataset overview table (paper Table I)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import load_uji  # noqa: E402
from data_more import load_mall, load_sod, load_tampere  # noqa: E402
from run_v3 import long_files, long_logins  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def heard(X):
    return float((X > 0).sum(1).mean())


rows = []
d = load_uji()
rows.append(("UJIIndoorLoc~\\cite{torres2014ujiindoorloc}", "University, 3 bldgs", d["train"]["X"].shape[1],
             len(d["train"]["X"]), len(d["test"]["X"]), "16 / 11", heard(d["test"]["X"]), "3 bldgs; 13 floors"))
Xf, _, _ = long_logins(long_files(1, "trn"), 1, "floor")
Xt, _, _ = long_logins(long_files(1, "tst"), 1, "floor")
rows.append(("UJI long-term~\\cite{mendoza2018longterm}", "Library, 25 months", Xf.shape[1],
             len(Xf), len(Xt), "1--2", heard(Xt), "2 floors; 4 half-floors"))
for b, env, zl in (("CETC331", "Offices", "3 floors"), ("HCXY", "Offices", "4 regions"), ("SYL", "Offices", "4 regions")):
    dd = load_sod(b, "floor" if b == "CETC331" else "region")
    devs = len(set(dd["train"]["dev"]) | set(dd["test"]["dev"]))
    rows.append((f"SODIndoorLoc {b}~\\cite{{bi2022sodindoorloc}}", env, dd["train"]["X"].shape[1],
                 len(dd["train"]["X"]), len(dd["test"]["X"]), str(devs), heard(dd["test"]["X"]), zl))
dd = load_tampere()
rows.append(("Tampere~\\cite{lohan2017tampere}", "University, crowdsourced", dd["train"]["X"].shape[1],
             len(dd["train"]["X"]), len(dd["test"]["X"]), str(len(set(dd["train"]["dev"]) | set(dd["test"]["dev"])))
             , heard(dd["test"]["X"]), "5 floors"))
for m, zl in ((1, "3 floors"), (2, "2 floors")):
    dd = load_mall(m)
    rows.append((f"Mall {m}~\\cite{{lopezpastor2020malls}}", "Shopping mall", dd["train"]["X"].shape[1],
                 len(dd["train"]["X"]), len(dd["test"]["X"]), "5 / 10", heard(dd["test"]["X"]), zl))
t = ["\\begin{tabular}{llrrrcrl}", "\\toprule",
     "Dataset & Environment & APs & Train & Test & Devices & Heard & Zones \\\\", "\\midrule"]
for r in rows:
    t.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]:,} & {r[4]:,} & {r[5]} & {r[6]:.1f} & {r[7]} \\\\".replace(",", "{,}").replace("{,} ", ", "))
t += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "paper/tables/datasets.tex").write_text("\n".join(t) + "\n")
print("\n".join(t))
