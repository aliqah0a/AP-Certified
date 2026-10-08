"""Table of uncertified defense baselines next to AP-Certified."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, TAB = ROOT / "results", ROOT / "paper/tables"
d = pd.read_csv(R / "defenses/defenses.csv")
ap = {"UJI building": pd.read_csv(R / "uji/attacks.csv"), "Library floor": pd.read_csv(R / "long/floor/attacks.csv")}
KS = (3, 6)
order = ["MLP", "Adversarially trained MLP", "LightGBM", "LightGBM + consistency filter"]
SHORT = {"MLP": "MLP", "Adversarially trained MLP": "Adv.-trained MLP", "LightGBM": "LightGBM", "LightGBM + consistency filter": "LightGBM + filter"}
t = ["\\begin{tabular}{l" + "c" * (2 * (len(KS) + 1)) + "}", "\\toprule",
     " & \\multicolumn{%d}{c}{UJI, buildings} & \\multicolumn{%d}{c}{Library, floors} \\\\" % (len(KS) + 1, len(KS) + 1),
     "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}" % (len(KS) + 2, len(KS) + 3, 2 * len(KS) + 3),
     "Verifier & FRR & " + " & ".join(f"$k{{=}}{k}$" for k in KS) + " & FRR & " + " & ".join(f"$k{{=}}{k}$" for k in KS) + " \\\\",
     "\\midrule"]
for m in order:
    cells = []
    for s in ("UJI building", "Library floor"):
        r = d[(d.setting == s) & (d.model == m)].iloc[0]
        cells += [f"{100*r.FRR:.1f}"] + [f"{100*r[f'att_k{k}']:.1f}" for k in KS]
    t.append(f"{SHORT[m]} & " + " & ".join(cells) + " \\\\")
cells = []
for s, frr in (("UJI building", "4.9"), ("Library floor", "1.7")):
    a = ap[s][ap[s].model == "AP-Certified"].set_index("k")
    cells += [frr] + [f"{100*a.success[k]:.1f} ({100*a.certified[k]:.1f})" for k in KS]
t += ["\\midrule", "AP-Certified & " + " & ".join(cells) + " \\\\", "\\bottomrule", "\\end{tabular}"]
(TAB / "defenses.tex").write_text("\n".join(t) + "\n")
print("\n".join(t))
