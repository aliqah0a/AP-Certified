"""Physics-grounded rogue-AP attacks on SODIndoorLoc CETC331 (three office floors).

1. Fit a log-distance path-loss model with a floor attenuation factor to the
   training scans, using the surveyed positions of the 26 installed dual-band
   APs:  RSS = P0_b - 10 n_b log10(max(d, 1 m)) - FAF_b * |floor difference|.
2. Attacks on impostor claims (scan from one floor, claim of another):
   - Rogue spoofers: the impostor carries up to k small transmitters, each
     broadcasting the BSSID of one installed AP of the claimed floor at
     distance 1-3 m with EIRP offset -10..+10 dB relative to the fitted P0.
     The phone reports the stronger of the genuine and the spoofed signal.
   - Compromised APs: the attacker controls up to k installed BSSIDs and
     changes their transmit power by -10..+10 dB or switches them off; the
     reading at the impostor's position follows the fitted model.
   Each attack edits at most k BSSIDs, so it lies inside the (k, 0) threat
   model, and the certificate must hold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import GroupVerifier, conformal_tau, hybrid_scores, partition_spread  # noqa: E402
from data_more import RAW, SOD_FILES, load_sod  # noqa: E402
from experiments import ALPHA, baseline_scores, fit_baselines, split_claims  # noqa: E402
from run_v3 import strat_half_groups  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results" / "physics"
RES.mkdir(parents=True, exist_ok=True)
K_MAX = 6


def ap_table(building="CETC331"):
    a = pd.read_excel(RAW / "sod/ap_info.xlsx", sheet_name=building)
    rows = []
    for _, r in a.iterrows():
        for band, col in (("2.4", "Attribute_2.4"), ("5", "Attribute_5")):
            if isinstance(r[col], str):
                rows.append(dict(col=int(r[col][3:]) - 1, x=r.ECoord, y=r.NCoord, floor=int(r.FloorID), band=band))
    return pd.DataFrame(rows).sort_values("col").reset_index(drop=True)


def fit_pathloss(df_tr, aps):
    """Least squares per band on all heard (scan, AP) pairs."""
    macs = [c for c in df_tr.columns if c.startswith("MAC")]
    R = df_tr[macs].values
    out = {}
    for band in ("2.4", "5"):
        A, b = [], []
        if not (aps.band == band).any():
            continue
        for _, ap in aps[aps.band == band].iterrows():
            rss = R[:, ap.col]
            h = rss != 100
            d = np.hypot(df_tr.ECoord.values[h] - ap.x, df_tr.NCoord.values[h] - ap.y)
            df = np.abs(df_tr.FloorID.values[h] - ap.floor)
            A.append(np.c_[np.ones(h.sum()), -10 * np.log10(np.maximum(d, 1.0)), -df]); b.append(rss[h])
        A, b = np.vstack(A), np.concatenate(b)
        theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = b - A @ theta
        out[band] = dict(P0=float(theta[0]), n=float(theta[1]), FAF=float(theta[2]),
                         sigma=float(res.std()), r2=float(1 - res.var() / b.var()), n_obs=int(len(b)))
    return out


def predict(pl, band, d, dfloor):
    p = pl[band]
    return p["P0"] - 10 * p["n"] * np.log10(np.maximum(d, 1.0)) - p["FAF"] * dfloor


def to_strength(rss):
    return float(np.clip(rss + 105.0, 1.0, 105.0)) if rss > -105 else 0.0


TASKS = {"CETC331": ("floor", "sod_cetc_floor", ""), "HCXY": ("region", "sod_hcxy_region", "_HCXY"),
         "SYL": ("region", "sod_syl_region", "_SYL")}


def main(building="CETC331", n_claims=150, seed=0):
    zones, task, suf = TASKS[building]
    d = load_sod(building, zones)
    tr, te, Z = d["train"], d["test"], d["Z"]
    df_tr = pd.read_csv(RAW / "sod" / SOD_FILES[building][0])
    df_te = pd.read_csv(RAW / "sod" / SOD_FILES[building][1])
    aps = ap_table(building)
    pl = fit_pathloss(df_tr, aps)
    (RES / f"pathloss{suf}.json").write_text(json.dumps(pl, indent=2))
    print("path loss:", json.dumps(pl, indent=1))
    floors = sorted(df_tr.FloorID.unique())
    if zones == "floor":
        aps["zone"] = [floors.index(f) for f in aps.floor]
    else:  # zone of the nearest reference point on the AP's floor
        P = df_tr[["ECoord", "NCoord"]].values
        aps["zone"] = [int(tr["y"][np.argmin(np.hypot(P[:, 0] - a.x, P[:, 1] - a.y))]) for _, a in aps.iterrows()]

    sel = pd.read_csv(Path(__file__).resolve().parents[1] / f"results/more/selection_{task}_m1.csv")
    b = sel.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
    gv = GroupVerifier(partition_spread(tr["X"], int(b.G), 0), Z, 10, float(b.theta),
                       power=float(b.power)).fit(tr["X"], tr["y"])
    c = float(b.c)
    base = fit_baselines(tr["X"], tr["y"])
    rs = np.random.default_rng(100)
    cal, ev = strat_half_groups(te["y"], te["g"], rs)
    S = hybrid_scores(gv.votes(te["X"]).astype(float), c)
    tauA = conformal_tau(split_claims(S[cal], te["y"][cal])[0], ALPHA, rs)
    B = {n: baseline_scores(m, te["X"], Z) for n, m in base.items()}
    tauB = {n: conformal_tau(split_claims(Sb[cal], te["y"][cal])[0], ALPHA, rs, 1e-9) for n, Sb in B.items()}

    rng = np.random.default_rng(seed)
    imp = [(i, z) for i in ev for z in range(Z) if z != te["y"][i]]
    pick = [imp[j] for j in rng.choice(len(imp), min(n_claims, len(imp)), replace=False)]

    def scorers(u):
        out = {"AP-Certified": (lambda X, z: hybrid_scores(gv.votes(X).astype(float), c)[:, z] + 1e-3 * gv.soft_score(X, z),
                                lambda X, z: hybrid_scores(gv.votes(X).astype(float), c)[:, z] + u >= tauA)}
        for n, m in base.items():
            out[n] = (lambda X, z, m=m: baseline_scores(m, X, Z)[:, z],
                      lambda X, z, m=m, n=n: baseline_scores(m, X, Z)[:, z] >= tauB[n])
        return out

    def candidates(x, i, z, mode):
        """Single physical actions -> (bssid, new strength)."""
        pos = (df_te.ECoord.values[i], df_te.NCoord.values[i], df_te.FloorID.values[i])
        cands = []
        if mode == "spoof":
            for _, ap in aps[aps.zone == z].iterrows():
                for dr in (1.0, 2.0, 3.0):
                    for dP in (-10, -5, 0, 5, 10):
                        v = max(x[ap.col], to_strength(predict(pl, ap.band, dr, 0) + dP))
                        if v != x[ap.col]:
                            cands.append((int(ap.col), v))
        else:  # compromised AP: power change or switch-off
            for _, ap in aps.iterrows():
                dist = np.hypot(pos[0] - ap.x, pos[1] - ap.y)
                base_rss = (x[ap.col] - 105.0) if x[ap.col] > 0 else predict(pl, ap.band, dist, abs(pos[2] - ap.floor))
                if x[ap.col] > 0:
                    cands.append((int(ap.col), 0.0))
                for dP in (-10, -5, 5, 10):
                    v = to_strength(base_rss + dP) if base_rss + dP > -100 else 0.0
                    if v != x[ap.col]:
                        cands.append((int(ap.col), v))
        return cands

    rows = []
    cert = np.zeros(K_MAX + 1)
    for i, z in pick:
        cert += np.array([np.clip(S[i, z] + k + 1 - tauA, 0, 1) for k in range(K_MAX + 1)])
    for mode in ("spoof", "compromise"):
        succ = {n: np.zeros(K_MAX + 1) for n in ["AP-Certified"] + list(base)}
        for i, z in pick:
            u = rng.uniform()
            for n, (score, acc) in scorers(u).items():
                x = te["X"][i].copy(); used = set()
                s = np.zeros(K_MAX + 1, bool); s[0] = acc(x[None], z)[0]
                for k in range(1, K_MAX + 1):
                    if s[k - 1]:
                        s[k:] = True; break
                    cs = [(j, v) for j, v in candidates(x, i, z, mode) if j not in used]
                    if not cs:
                        break
                    C = np.repeat(x[None], len(cs), 0)
                    for r, (j, v) in enumerate(cs):
                        C[r, j] = v
                    a = acc(C, z)
                    if a.any():
                        s[k:] = True; break
                    bi = int(np.argmax(score(C, z)))
                    x = C[bi]; used.add(cs[bi][0])
                succ[n] += s
        for n, s in succ.items():
            for k in range(K_MAX + 1):
                rows.append(dict(attack=mode, model=n, k=k, success=s[k] / len(pick),
                                 certified=cert[k] / len(pick) if n == "AP-Certified" else np.nan))
        print(mode, {n: (s / len(pick)).round(3).tolist() for n, s in succ.items()}, flush=True)
    pd.DataFrame(rows).to_csv(RES / f"physics_attacks{suf}.csv", index=False)
    # data for the path-loss figure
    macs = [cc for cc in df_tr.columns if cc.startswith("MAC")]
    R = df_tr[macs].values
    pts = []
    for _, ap in aps.iterrows():
        h = R[:, ap.col] != 100
        dd = np.hypot(df_tr.ECoord.values[h] - ap.x, df_tr.NCoord.values[h] - ap.y)
        pts.append(pd.DataFrame(dict(band=ap.band, d=dd, dfloor=np.abs(df_tr.FloorID.values[h] - ap.floor),
                                     rss=R[h, ap.col])))
    pd.concat(pts).to_csv(RES / f"pathloss_points{suf}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "CETC331")
