"""Figure and summary for the physically grounded attacks on SODIndoorLoc."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R, FIG = ROOT / "results/physics", ROOT / "paper/figures"
COL = {"AP-Certified": "#2a78d6", "LightGBM": "#eb6834", "MLP": "#1baf7a", "kNN": "#eda100"}
LAB = {"AP-Certified": "AP-Certified", "LightGBM": "LightGBM", "MLP": "MLP", "kNN": "$K$-NN"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.5, "pdf.fonttype": 42})

sites = [("CETC331", "", "CETC331, floor zones")] + [
    (b, f"_{b}", f"{b}, region zones") for b in ("HCXY", "SYL") if (R / f"physics_attacks_{b}.csv").exists()]
fig0, ax = plt.subplots(figsize=(3.45, 2.4))

# (a) path-loss fit, CETC331 2.4 GHz
pl = json.loads((R / "pathloss.json").read_text())
pts = pd.read_csv(R / "pathloss_points.csv")
pts = pts[pts.band.astype(str) == "2.4"]
fc = ["#2a78d6", "#eb6834", "#1baf7a"]
rng = np.random.default_rng(0)
for f in (0, 1, 2):
    q = pts[pts.dfloor == f]
    q = q.iloc[rng.choice(len(q), min(1500, len(q)), replace=False)]
    ax.scatter(np.maximum(q.d, 1), q.rss, s=2, alpha=0.25, color=fc[f], lw=0, rasterized=True)
    dd = np.logspace(0, np.log10(80), 50)
    p = pl["2.4"]
    ax.plot(dd, p["P0"] - 10 * p["n"] * np.log10(dd) - p["FAF"] * f, color=fc[f], lw=1.3,
            label=f"$\\Delta f = {f}$")
ax.set_xscale("log"); ax.set_xlabel("Distance to AP (m)"); ax.set_ylabel("RSS (dBm)")
ax.set_title(f"CETC331, 2.4 GHz: $n = {p['n']:.2f}$, FAF $= {p['FAF']:.1f}$ dB, $R^2 = {p['r2']:.2f}$", fontsize=7.5)
ax.legend(fontsize=6.5, loc="lower left", frameon=False, handlelength=1.2)
fig0.tight_layout(); fig0.savefig(FIG / "pathloss.pdf", dpi=300); fig0.savefig(FIG / "pathloss.png", dpi=200); plt.close(fig0)

W = 7.16 if len(sites) > 2 else 3.45
fig, axes = plt.subplots(1, len(sites), figsize=(W, 2.15), sharey=True, gridspec_kw=dict(wspace=0.12))
axes = np.atleast_1d(axes)

summary = {}
for ax, (b, suf, title), letter in zip(axes, sites, "abc"):
    d = pd.read_csv(R / f"physics_attacks{suf}.csv")
    for n in ["LightGBM", "MLP", "kNN", "AP-Certified"]:
        for mode, ls, mk in (("spoof", "-", "o"), ("compromise", "--", "s")):
            q = d[(d.model == n) & (d.attack == mode)].sort_values("k")
            ax.plot(q.k, 100 * q.success, ls=ls, marker=mk, ms=2.5, color=COL[n], lw=1.1)
    c = d[(d.model == "AP-Certified") & (d.attack == "spoof")].sort_values("k")
    ax.plot(c.k, 100 * c.certified, color="black", lw=1.0, ls=":", label="Certified bound")
    summary[b] = {f"{n}_{m}": d[(d.model == n) & (d.attack == m)].sort_values("k").success.round(3).tolist()
                  for n in COL for m in ("spoof", "compromise")}
    summary[b]["certified"] = c.certified.round(4).tolist()
    ax.set_xlabel("Manipulated BSSIDs $k$")
    if letter == "a":
        ax.set_ylabel("Impostor acceptance (%)")
    ax.set_ylim(-3, 103); ax.set_title(f"({letter}) {title}", fontsize=7.5)
h = [plt.Line2D([], [], color=COL[n], lw=1.3, label=LAB[n]) for n in COL]
h += [plt.Line2D([], [], color="gray", ls="-", marker="o", ms=2.5, label="Rogue spoofers"),
      plt.Line2D([], [], color="gray", ls="--", marker="s", ms=2.5, label="Compromised APs"),
      plt.Line2D([], [], color="black", ls=":", label="Certified bound")]
nc = 7 if len(sites) > 2 else 4
fig.legend(handles=h, loc="lower center", ncol=nc, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02 if nc == 7 else -0.12))
fig.subplots_adjust(bottom=0.3 if nc == 7 else 0.36, top=0.9, left=0.08, right=0.99)
fig.savefig(FIG / "physics.pdf", bbox_inches="tight"); fig.savefig(FIG / "physics.png", dpi=200, bbox_inches="tight")
(R / "summary_physics.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
