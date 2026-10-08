"""Figures and tables for the pre-deployment certifiability check, sweep sizes
and population-level radii (results of run_check.py and run_check2.py)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG, TAB = ROOT / "results", ROOT / "paper/figures", ROOT / "paper/tables"
src = pd.read_csv(R / "certifiability_sweep.csv").groupby("setting").source_kstar.first()
d = pd.read_csv(R / "check_sweepsize.csv")
h = d[d.frac == 0.5]
order = ["SOD CETC331 floor", "Library floor", "UJI building", "SOD HCXY region", "Library half-floor",
         "SOD SYL region", "Mall 1 floor", "Mall 2 floor", "Tampere floor"]
g = h.groupby("setting").agg(sweep=("k_sweep", "mean"), deploy=("k_dep", "mean")).loc[order]
g["source"] = src.loc[order]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# ---------------------------------------------------------------- Fig: check
fig, ax = plt.subplots(figsize=(3.45, 2.9))
y = np.arange(len(g))[::-1]
ax.axvspan(-1.6, 0.5, color="#888888", alpha=0.08, lw=0)
ax.text(-0.55, len(g) - 0.15, "not certifiable", ha="center", va="center", fontsize=6.0, color="#666666")
for yi, (_, r) in zip(y, g.iterrows()):
    ax.plot([min(r.source, r.deploy), max(r.source, r.deploy)], [yi, yi], color="#cccccc", lw=1.2, zorder=1)
ax.scatter(g.source, y + 0.18, marker="o", facecolor="white", edgecolor=ORANGE, lw=1.1, s=22, zorder=3,
           label="Selection data (source)")
ax.scatter(g.sweep, y - 0.18, marker="s", color=BLUE, s=16, zorder=3, label="Deployment sweep")
ax.scatter(g.deploy, y, marker="|", color="black", s=90, lw=1.6, zorder=4, label="Achieved on deployment")
ax.set_yticks(y); ax.set_yticklabels(g.index, fontsize=7)
ax.set_xlim(-1.6, 10.6); ax.set_xticks(range(-1, 11))
ax.set_xticklabels(["none"] + [str(i) for i in range(0, 11)], fontsize=7)
ax.set_xlabel("Certified radius $k^*$ (APs)")
ax.set_ylim(-0.6, len(g) + 0.15)
ax.legend(loc="center right", fontsize=6.5, frameon=True, framealpha=0.95, edgecolor="none")
fig.tight_layout(); fig.savefig(FIG / "sweep_check.pdf"); fig.savefig(FIG / "sweep_check.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------- Table: check accuracy
def stats(est, dep):
    return dict(r=np.corrcoef(est, dep)[0, 1], mae=np.abs(est - dep).mean())


go = h.k_dep >= 1
s_src, s_sw = stats(g.source, g.deploy), stats(g.sweep, g.deploy)
hh = h.merge(src.rename("source").reset_index(), on="setting")
rows = [("Selection data (source)", s_src, ((hh.source >= 1) == go.values).mean(), ((hh.source >= 1) & ~go.values).mean()),
        ("Deployment sweep, point estimate", s_sw, ((h.k_sweep >= 1) == go).mean(), ((h.k_sweep >= 1) & ~go).mean()),
        ("Deployment sweep, lower confidence", stats(h.groupby("setting").k_sweep_cp.mean().loc[order], g.deploy),
         ((h.k_sweep_cp >= 1) == go).mean(), ((h.k_sweep_cp >= 1) & ~go).mean())]
t = ["\\begin{tabular}{lcccc}", "\\toprule",
     "Radius estimate & Pearson $r$ & MAE (APs) & Go/no-go correct & False go \\\\", "\\midrule"]
for name, st, acc, fg in rows:
    t.append(f"{name} & {st['r']:.2f} & {st['mae']:.2f} & {100*acc:.0f}\\% & {100*fg:.0f}\\% \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(TAB / "sweep_check.tex").write_text("\n".join(t) + "\n")
print("\n".join(t))

# ---------------------------------------------------------------- Fig: sweep size
def per_frac(q):
    go = q.k_dep >= 1
    return pd.Series(dict(
        acc=((q.k_sweep >= 1) == go).mean(), fg=((q.k_sweep >= 1) & ~go).mean(),
        acc_cp=((q.k_sweep_cp >= 1) == go).mean(), fg_cp=((q.k_sweep_cp >= 1) & ~go).mean(),
        mae=(q.k_sweep - q.k_dep).abs().mean(),
        frr_med=q.FRR_dep.median(), frr_q90=q.FRR_dep.quantile(0.9), frr_gt10=(q.FRR_dep > 0.10).mean(),
        pos=q.n_pos_sweep.median()))


pf = d.groupby("frac").apply(per_frac)
pf.to_csv(R / "check_sweepsize_summary.csv")
print(pf.round(3))
fr = np.arange(len(pf)); frl = [f"{int(round(100*f))}" for f in pf.index.values]
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.2), gridspec_kw=dict(wspace=0.5))
ax = axes[0]
ax.plot(fr, 100 * pf.acc, marker="o", ms=3, color=BLUE, label="Point estimate")
ax.plot(fr, 100 * pf.acc_cp, marker="s", ms=3, color=GREEN, label="Lower confidence")
ax.plot(fr, 100 * pf.fg, marker="o", ms=3, color=BLUE, ls="--")
ax.plot(fr, 100 * pf.fg_cp, marker="s", ms=3, color=GREEN, ls="--")
ax.set_xticks(fr); ax.set_xticklabels(frl); ax.set_xlim(-0.3, len(fr) - 0.7)
ax.set_xlabel("Sweep size (% of positions)"); ax.set_ylabel("Splits (%)")
ax.set_ylim(-4, 104); ax.set_title("(a) Go/no-go decisions", fontsize=7)
ax.text(1.5, 88, "correct", fontsize=6, color="#555555", ha="center")
ax.text(1.5, 10, "false go", fontsize=6, color="#555555", ha="center")
ax.legend(fontsize=5.8, loc="center left", frameon=False, bbox_to_anchor=(0.0, 0.33))
ax = axes[1]
ax.plot(fr, 100 * pf.frr_med, marker="o", ms=3, color=ORANGE, label="Median")
ax.plot(fr, 100 * pf.frr_q90, marker="^", ms=3, color=ORANGE, ls="--", label="90th percentile")
ax.axhline(5, color="#888888", lw=0.8, ls=":")
ax.set_xticks(fr); ax.set_xticklabels(frl); ax.set_xlim(-0.3, len(fr) - 0.7)
ax.set_xlabel("Sweep size (% of positions)"); ax.set_ylabel("Achieved FRR (%)")
ax.set_title("(b) Achieved FRR", fontsize=7)
ax.legend(fontsize=5.8, loc="upper right", frameon=False)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.88)
fig.savefig(FIG / "sweep_size.pdf", bbox_inches="tight"); fig.savefig(FIG / "sweep_size.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Table: population radii
p = h.groupby("setting").agg(k=("k_dep", "mean"), kcp=("k_dep_cp", "mean"), kpos=("k_dep_cp_pos", "mean"),
                             nimp=("n_imp_dep", "mean"), npos=("n_pos_dep", "mean")).loc[order]
fmt = lambda v: "none" if v <= -1 + 1e-9 else f"{v:.1f}"
t = ["\\begin{tabular}{lrrccc}", "\\toprule",
     "Setting & Claims & Positions & $k^*$ (sample) & $k^*_{\\mathrm{pop}}$ (claims) & $k^*_{\\mathrm{pop}}$ (positions) \\\\",
     "\\midrule"]
for s_, r in p.iterrows():
    t.append(f"{s_} & {int(round(r.nimp)):,} & {int(round(r.npos)):,} & {fmt(r.k)} & {fmt(r.kcp)} & {fmt(r.kpos)} \\\\"
             .replace(",", "{,}"))
t += ["\\bottomrule", "\\end{tabular}"]
(TAB / "population.tex").write_text("\n".join(t) + "\n")
print("\n".join(t))
json.dump(dict(per_frac=pf.round(4).to_dict()), open(R / "check_sweepsize_summary.json", "w"), indent=1)
