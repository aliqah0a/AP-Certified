"""Tables (paper/tables/*.tex) and figures (paper/figures/*.pdf) for the main study."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG, TAB = ROOT / "results", ROOT / "paper/figures", ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
                     "grid.linewidth": 0.5, "lines.linewidth": 1.4, "pdf.fonttype": 42})
pct = lambda s: f"{100*s.mean():.2f} $\\pm$ {100*s.std():.2f}"
summary = {}


def table(name, header, rows, cols):
    t = ["\\begin{tabular}{" + cols + "}", "\\toprule", header + " \\\\", "\\midrule"]
    t += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    t += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(t) + "\n")


# ------------------------------------------------------------------ UJI main
sp = pd.read_csv(R / "uji/splits.csv")
dl = pd.read_csv(R / "uji/delta.csv") if (R / "uji/delta.csv").exists() else None
order = [("kNN", "deployment"), ("LightGBM", "deployment"), ("MLP", "deployment"),
         ("FPA-style symmetric", "deployment"), ("AP-Certified (random)", "deployment"),
         ("AP-Certified (spread)", "deployment"), ("AP-Certified (spread)", "source")]
rows = []
for m, c in order:
    d = sp[(sp.model == m) & (sp.calib == c)]
    ks = d.kstar
    kk = "--" if ks.isna().all() else f"{ks.mean():.1f} [{int(ks.min())}, {int(ks.max())}]"
    label = m if c == "deployment" else m + ", source calib."
    rows.append((label, pct(d.EER), pct(d.FRR), pct(d.FAR), kk))
    summary[f"uji|{m}|{c}"] = dict(EER=d.EER.mean(), FRR=d.FRR.mean(), FAR=d.FAR.mean(), kstar=ks.mean())
table("uji_main", "Verifier & EER (\\%) & FRR (\\%) & FAR (\\%) & $k^*$", rows, "lcccc")
mond = sp[(sp.model == "AP-Certified (spread)") & (sp.calib == "deployment")].mondrian.dropna()
mf = pd.DataFrame([json.loads(x) for x in mond])
summary["uji|mondrian_FRR_mean"] = mf.mean().to_dict(); summary["uji|mondrian_FRR_max"] = mf.max().to_dict()

# certified curves UJI
cu = pd.read_csv(R / "uji/curves.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
sty = [("AP-Certified (spread)", C[0], "-", "o", "AP-Certified, spread"),
       ("FPA-style symmetric", C[1], "--", "s", "Symmetric rule, spread"),
       ("AP-Certified (random)", C[2], ":", "^", "AP-Certified, random")]
for m, col, ls, mk, lab in sty:
    g = cu[(cu.model == m) & (cu.delta == 0)].groupby("k").cert.agg(["mean", "min", "max"]).reset_index()
    ax.plot(g.k, 100 * g["mean"], ls=ls, marker=mk, ms=3.5, color=col, label=lab)
    ax.fill_between(g.k, 100 * g["min"], 100 * g["max"], color=col, alpha=0.12, lw=0)
if dl is not None:
    for dv, col in ((1.0, C[3]), (2.0, C[4])):
        g = dl[dl.delta == dv][[f"k{k}" for k in range(11)]].mean()
        ax.plot(range(11), 100 * g.values, color=col, ls="-.", lw=1.1, label=f"AP-Certified, $\\delta$={int(dv)} dB")
ax.axhline(1, color="#888888", lw=0.7, ls=(0, (1, 2)))
ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 100); ax.set_xlim(0, 10)
ax.set_xlabel("Manipulated APs $k$"); ax.set_ylabel("Certified FAR (%)")
ax.legend(frameon=False, fontsize=6.3, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "uji_certified.pdf"); fig.savefig(FIG / "uji_certified.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ attacks
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, (name, path) in zip(axes, (("UJIIndoorLoc, building zones", R / "uji/attacks.csv"),
                                    ("Long-term, floor zones", R / "long/floor/attacks.csv"))):
    if not path.exists():
        continue
    a = pd.read_csv(path)
    for m, col, mk in (("kNN", C[1], "s"), ("LightGBM", C[3], "D"), ("MLP", C[2], "^"), ("AP-Certified", C[0], "o")):
        r = a[a.model == m]
        ax.plot(r.k, 100 * r.success, marker=mk, ms=3.5, color=col, label=f"{m} (attack)")
    r = a[a.model == "AP-Certified"]
    ax.plot(r.k, 100 * r.certified, color=C[0], ls="--", label="AP-Certified (certified bound)")
    ax.set_title(name, fontsize=8); ax.set_xlabel("Manipulated APs $k$")
    summary[f"attack|{name}"] = a.pivot_table(index="k", columns="model", values="success").round(4).to_dict()
    summary[f"attack_cert|{name}"] = r.set_index("k").certified.round(4).to_dict()
axes[0].set_ylabel("Impostor acceptance (%)")
axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "attacks.pdf"); fig.savefig(FIG / "attacks.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ drift
mo = pd.read_csv(R / "long/floor/months.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
for (m, c), col, ls, lab in ((("AP-Certified (spread)", "static"), C[1], "--", "AP-Certified, static"),
                             (("AP-Certified (spread)", "recalibrated"), C[0], "-", "AP-Certified, recalibrated"),
                             (("AP-Certified (spread)", "retrained"), C[2], ":", "AP-Certified, retrained"),
                             (("MLP", "static"), C[3], "-.", "MLP, static")):
    d = mo[(mo.model == m) & (mo.calib == c)].sort_values("month")
    ax.plot(d.month, 100 * d.FRR, color=col, ls=ls, marker="o", ms=2.5, label=lab)
ax.axhline(5, color="#888888", lw=0.7, ls=(0, (1, 2)))
ax.set_yscale("log"); ax.set_ylim(0.8, 100)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
ax.set_xlabel("Month"); ax.set_ylabel("FRR (%)")
ax.legend(frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
ax = axes[1]
for c, col, ls, lab in (("recalibrated", C[0], "-", "recalibrated"), ("retrained", C[2], ":", "retrained"),
                        ("static", C[1], "--", "static")):
    d = mo[(mo.model == "AP-Certified (spread)") & (mo.calib == c)].sort_values("month")
    ax.step(d.month, d.kstar, where="mid", color=col, ls=ls, label=lab)
ax.set_xlabel("Month"); ax.set_ylabel("Certified radius $k^*$"); ax.set_yticks([-1, 0, 1, 2, 3, 4])
ax.legend(frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
fig.tight_layout(); fig.savefig(FIG / "drift.pdf"); fig.savefig(FIG / "drift.png", dpi=200); plt.close(fig)
g = mo.groupby(["model", "calib"])
summary["long|monthly"] = g[["FRR", "FAR", "EER", "kstar"]].mean().round(4).reset_index().to_dict("records")
summary["long|months_FRR_gt_5"] = {f"{a}|{b}": int((d.FRR > 0.05).sum()) for (a, b), d in g}
summary["long|month1"] = mo[mo.month == 1].round(4).to_dict("records")
rows = []
for (m, c) in (("kNN", "recalibrated"), ("LightGBM", "recalibrated"), ("MLP", "recalibrated"),
               ("MLP", "static"), ("AP-Certified (random)", "recalibrated"),
               ("AP-Certified (spread)", "static"), ("AP-Certified (spread)", "recalibrated"),
               ("AP-Certified (spread)", "retrained")):
    d = mo[(mo.model == m) & (mo.calib == c)]
    d1 = d[d.month == 1]
    k1 = "--" if d.kstar.isna().all() or len(d1) == 0 else f"{int(d1.kstar.iloc[0])}"
    kall = "--" if d.kstar.isna().all() else f"{d.kstar.mean():.1f}"
    f1 = f"{100*d1.FRR.iloc[0]:.2f}" if len(d1) else "--"
    rows.append((f"{m}, {c}", f1, f"{100*d.FRR.mean():.2f}", f"{(d.FRR > 0.05).sum()}/{len(d)}",
                 f"{100*d.FAR.mean():.2f}", k1, kall))
table("long_main", "Verifier & FRR m1 (\\%) & FRR avg (\\%) & months FRR$>$5\\% & FAR avg (\\%) & $k^*$ m1 & $k^*$ avg",
      rows, "lcccccc")

# ------------------------------------------------------------------ partition
rows = []
for name, path, cfgp in (("UJI", R / "uji/partition.csv", None), ("Long-term", R / "long/floor/partition.csv", None)):
    p = pd.read_csv(path)
    for _, r in p.iterrows():
        rows.append((name, int(r.G), int(r.bound_q05), f"{r.random_expected_q05:.1f}", int(r.random_q05),
                     int(r.spread_q05), f"{r.bound_mean:.1f}", f"{r.random_mean:.1f}", f"{r.spread_mean:.1f}"))
table("partition", "Data & $G$ & Bound & Rand. (exp.) & Rand. & Spread & Bound & Rand. & Spread", rows, "lcccccccc")

# ------------------------------------------------------------------ delta and limits
rows = []
if dl is not None:
    for dv in sorted(dl.delta.unique()):
        d = dl[dl.delta == dv]
        rows.append(("UJI, building", f"{dv:.0f}", f"{d.kstar.mean():.1f} [{int(d.kstar.min())}, {int(d.kstar.max())}]",
                     f"{100*d.k1.mean():.2f}", f"{100*d.k3.mean():.2f}"))
lp = R / "long/floor/delta.csv"
if lp.exists():
    ld = pd.read_csv(lp)
    for _, r in ld.iterrows():
        rows.append(("Long-term, floor", f"{r.delta:.0f}", f"{int(r.kstar)}", f"{100*r.k1:.2f}", f"{100*r.k3:.2f}"))
table("delta", "Data & $\\delta$ (dB) & $k^*$ & FAR$_\\mathrm{cert}(1)$ (\\%) & FAR$_\\mathrm{cert}(3)$ (\\%)", rows, "lcccc")

rows = []
fl = R / "uji/floor_limit.csv"
if fl.exists():
    f = pd.read_csv(fl)
    for m in ("kNN", "LightGBM", "MLP", "AP-Certified (hierarchical)"):
        d = f[f.model == m]
        rows.append(("UJI, 13 floors", m, pct(d.EER), pct(d.FRR), pct(d.FAR),
                     "--" if "kstar" not in d or d.kstar.isna().all() else f"{d.kstar.mean():.1f}"))
    h = f[f.model == "AP-Certified (hierarchical)"]
    summary["uji_floor|adjacent_FAR"] = h.FAR_adjacent.mean()
qp = R / "long/quadrant/months.csv"
if qp.exists():
    q = pd.read_csv(qp)
    for m in ("kNN", "LightGBM", "MLP", "AP-Certified (spread)"):
        d = q[(q.model == m) & (q.calib == "recalibrated")]
        rows.append(("Long-term, 4 half-floors", m, f"{100*d.EER.iloc[0]:.2f}", f"{100*d.FRR.iloc[0]:.2f}",
                     f"{100*d.FAR.iloc[0]:.2f}", "--" if d.kstar.isna().all() else f"{int(d.kstar.iloc[0])}"))
table("limits", "Task & Verifier & EER (\\%) & FRR (\\%) & FAR (\\%) & $k^*$", rows, "llcccc")

(R / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps(summary, indent=1, default=float)[:6000])
