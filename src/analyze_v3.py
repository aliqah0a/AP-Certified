"""Tables and figures for the v3 additions (multi-scan, hybrid, cascade, online)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG, TAB = ROOT / "results/v3", ROOT / "paper/figures", ROOT / "paper/tables"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
                     "grid.linewidth": 0.5, "lines.linewidth": 1.4, "pdf.fonttype": 42})


def table(name, header, rows, cols):
    t = ["\\begin{tabular}{" + cols + "}", "\\toprule", header + " \\\\", "\\midrule"]
    t += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    t += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(t) + "\n")


ms = pd.read_csv(R / "multiscan.csv")
ms = ms[ms.dataset == "long-term"]
rows = []
for zones, zl in (("floor", "Floors"), ("quadrant", "Half-floors")):
    for m in (1, 3, 5):
        for rule in ("one-sided", "symmetric", "hybrid", "one-sided weighted"):
            d = ms[(ms.zones == zones) & (ms.m == m) & (ms.rule == rule)]
            if not len(d):
                continue
            r = d.iloc[0]
            cc = "$\\infty$" if np.isposinf(r.c) else ("$-\\infty$" if np.isneginf(r.c) else f"{r.c:g}")
            rows.append((zl if (m == 1 and rule == "one-sided") else "", m if rule == "one-sided" else "",
                         rule, int(r.G), f"{r.theta:g}", cc, f"{100*r.FRR:.2f}", f"{100*r.FAR:.2f}",
                         f"{100*r.EER:.2f}", int(r.kstar), int(r.bound)))
table("multiscan", "Zones & $m$ & Rule & $G$ & $\\theta$ & $c$ & FRR (\\%) & FAR (\\%) & EER (\\%) & $k^*$ & Bound",
      rows, "llllcccccccc"[:11])

# cascade
cp = R / "cascade.csv"
if cp.exists():
    ca = pd.read_csv(cp)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, m in zip(axes, (1, 3)):
        d = ca[(ca.m == m) & ca.k.notna()]
        for name, col, mk in (("AP-Certified alone", C[0], "o"), ("cascade", C[1], "s")):
            r = d[d.model == name]
            ax.plot(r.k, 100 * r.attack, color=col, marker=mk, ms=3.5, label=f"{name} (attack)")
            ax.plot(r.k, 100 * r.certified, color=col, ls="--", lw=1.0, label=f"{name} (certified)")
        ax.set_title(f"Library floors, $m = {m}$", fontsize=8); ax.set_xlabel("Manipulated APs $k$")
    axes[0].set_ylabel("Impostor acceptance (%)")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "cascade.pdf"); fig.savefig(FIG / "cascade.png", dpi=200); plt.close(fig)

# online
for m in (1, 3):
    op = R / f"online_m{m}.csv"
    if not op.exists():
        continue
    o = pd.read_csv(op)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    sty = {"static": (C[1], "--"), "online": (C[2], ":"), "online+floor": (C[3], "-."),
           "online+floor+retrain": (C[0], "-")}
    for n, (col, ls) in sty.items():
        d = o[o.method == n]
        axes[0].plot(d.month, 100 * d.FRR, color=col, ls=ls, marker="o", ms=2.5, label=n)
        axes[1].plot(d.month, 100 * d.certFAR_k0, color=col, ls=ls, marker="o", ms=2.5, label=n)
    axes[0].axhline(5, color="#888888", lw=0.7, ls=(0, (1, 2)))
    axes[1].axhline(1, color="#888888", lw=0.7, ls=(0, (1, 2)))
    for ax, yl in zip(axes, ("FRR (%)", "Certified FAR at $k_0 = 1$ (%)")):
        ax.set_xlabel("Month"); ax.set_ylabel(yl)
    axes[0].set_yscale("log"); axes[0].set_ylim(0.8, 100)
    axes[0].set_yticks([1, 2, 5, 10, 20, 50]); axes[0].set_yticklabels(["1", "2", "5", "10", "20", "50"])
    axes[1].set_yscale("symlog", linthresh=0.1); axes[1].set_ylim(0, 20)
    axes[1].set_yticks([0, 0.1, 1, 10]); axes[1].set_yticklabels(["0", "0.1", "1", "10"])
    axes[0].legend(frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(FIG / f"online_m{m}.pdf"); fig.savefig(FIG / f"online_m{m}.png", dpi=200)
    plt.close(fig)

summary = {}
for m in (1, 3):
    op = R / f"online_m{m}.csv"
    if op.exists():
        o = pd.read_csv(op)
        g = o.groupby("method")
        summary[f"online_m{m}"] = dict(
            FRR_weighted=(g.apply(lambda d: (d.FRR * d.n).sum() / d.n.sum())).round(4).to_dict(),
            months_FRR_gt5=g.apply(lambda d: int((d.FRR > 0.05).sum())).to_dict(),
            FAR_mean=g.FAR.mean().round(4).to_dict(),
            cert_mean=g.certFAR_k0.mean().round(4).to_dict(),
            months_cert_gt1=g.apply(lambda d: int((d.certFAR_k0 > 0.01).sum())).to_dict(),
            bind_mean=g.floor_bind.mean().round(4).to_dict(),
            meta=json.loads((R / f"online_m{m}.json").read_text()) if (R / f"online_m{m}.json").exists() else None)
        rows = []
        for n in ("static", "online", "online+floor", "online+floor+retrain"):
            d = o[o.method == n]
            frr = (d.FRR * d.n).sum() / d.n.sum()
            rows.append((n, f"{100*frr:.2f}", f"{100*d.FRR.max():.1f}", f"{100*d.FAR.mean():.2f}",
                         f"{100*d.certFAR_k0.mean():.2f}", f"{(d.certFAR_k0 > 0.01).sum()}/25",
                         f"{100*d.floor_bind.mean():.1f}"))
        table(f"online_m{m}", "Method & FRR (\\%) & max monthly FRR (\\%) & FAR (\\%) & FAR$_\\mathrm{cert}(1)$ (\\%) & months cert.$>$1\\% & floor active (\\%)",
              rows, "lcccccc")
if cp.exists():
    s = ca[ca.model == "cascade-summary"]
    summary["cascade"] = s.round(4).to_dict("records")
    summary["cascade_attack"] = ca[ca.k.notna()].round(4).to_dict("records")
(R / "summary_v3.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str)[:5000])
