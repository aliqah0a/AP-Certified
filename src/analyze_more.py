"""Table for the additional datasets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, TAB = ROOT / "results/more", ROOT / "paper/tables"
INFO = {  # dataset, zones, mean heard APs (test), #zones
    "sod_cetc_floor": ("SODIndoorLoc CETC331 (offices)", "3 floors"),
    "sod_syl_region": ("SODIndoorLoc SYL (offices)", "4 regions, 1 floor"),
    "sod_hcxy_region": ("SODIndoorLoc HCXY (offices)", "4 regions, 1 floor"),
    "tampere_floor": ("Tampere (crowdsourced, 20 devices)", "5 floors"),
    "mall1_floor": ("Mall 1", "3 floors"),
    "mall2_floor": ("Mall 2", "2 floors"),
}
rows, summ = [], {}
for task, (name, zones) in INFO.items():
    for m in (1, 3, 5):
        p = R / f"{task}_m{m}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        h = d[d.model == "AP-Certified (hybrid)"]
        ks = h.kstar
        kk = f"{ks.mean():.1f} [{int(ks.min())}, {int(ks.max())}]"
        mlp = d[d.model == "MLP"].EER.mean() if (d.model == "MLP").any() else np.nan
        lgb = d[d.model == "LightGBM"].EER.mean() if (d.model == "LightGBM").any() else np.nan
        a3 = {"AP": "--", "LightGBM": "--", "MLP": "--"}
        ap = R / f"{task}_attacks.csv"
        if m == 1 and ap.exists():
            a = pd.read_csv(ap)
            for k_, lab in (("AP-Certified (hybrid)", "AP"), ("LightGBM", "LightGBM"), ("MLP", "MLP")):
                v = a[(a.model == k_) & (a.k == 3)].success
                a3[lab] = f"{100*v.iloc[0]:.0f}" if len(v) else "--"
        rows.append((name if m == 1 else "", zones if m == 1 else "", m,
                     f"{100*h.FRR.mean():.2f}", f"{100*h.FAR.mean():.2f}", f"{100*h.EER.mean():.2f}", kk,
                     "--" if np.isnan(mlp) else f"{100*mlp:.2f}", "--" if np.isnan(lgb) else f"{100*lgb:.2f}",
                     a3["AP"], a3["MLP"], a3["LightGBM"]))
        summ[f"{task}_m{m}"] = dict(FRR=h.FRR.mean(), FAR=h.FAR.mean(), EER=h.EER.mean(), kstar=ks.mean(),
                                    kmin=int(ks.min()), kmax=int(ks.max()), MLP_EER=mlp, LGB_EER=lgb,
                                    attack_k3=a3,
                                    other=d.groupby("model")[["FRR", "FAR", "EER", "kstar"]].mean().round(4).to_dict())
t = ["\\begin{tabular}{llcccccccccc}", "\\toprule",
     "Dataset & Zones & $m$ & FRR (\\%) & FAR (\\%) & EER (\\%) & $k^*$ & MLP EER & LGBM EER & \\multicolumn{3}{c}{Attack @ $k{=}3$ (\\%)} \\\\",
     " & & & \\multicolumn{4}{c}{AP-Certified (hybrid)} & (\\%) & (\\%) & AP-C. & MLP & LGBM \\\\", "\\midrule"]
t += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
t += ["\\bottomrule", "\\end{tabular}"]
(TAB / "more.tex").write_text("\n".join(t) + "\n")
(R / "summary_more.json").write_text(json.dumps(summ, indent=1, default=str))
print("\n".join(t))
