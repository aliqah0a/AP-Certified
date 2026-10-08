"""v3 experiments: (A) multi-scan logins and the hybrid / weighted rules,
(B) two-layer cascade with kNN, (C) online calibration under drift.

Usage: python3 src/run_v3.py A|B|C
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from attacks import (cascade_greedy_attack, group_single_edits, zone_profiles)  # noqa: E402
from core import (GroupVerifier, OnlineThreshold, certified_far_randomized,  # noqa: E402
                  certified_far_weighted, conformal_tau, eer, frr_far,
                  hybrid_scores, merge_scans, partition_spread, weighted_votes)
from data_long import DB, _load, load_month, zones_from_coords  # noqa: E402
from experiments import ALPHA, K_RANGE, kstar_from_curve, q_alpha, split_claims  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v3"
RES.mkdir(parents=True, exist_ok=True)
E = float(np.e)
C_GRID = [np.inf, 6.0, 4.0, 2.0, 0.0, -2.0, -np.inf]
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)


# =========================================================================
# Login construction
# =========================================================================
def windows(idx, m):
    return [idx[i:i + m] for i in range(0, len(idx) - m + 1, m)]


def long_logins(files, m, scheme):
    """Logins of m consecutive scans at one position within one file."""
    X, y, P = [], [], []
    for f in files:
        S, C = _load([f])
        _, inv = np.unique(C, axis=0, return_inverse=True)
        z = zones_from_coords(C, scheme)
        for p in np.unique(inv):
            idx = np.where(inv == p)[0]
            # bursts: runs of consecutive rows (each position is visited in
            # separate bursts minutes apart); a login never spans two bursts
            bursts = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for w in (w for b in bursts for w in windows(b, m)):
                X.append(merge_scans(S[w])); y.append(z[w[0]]); P.append(f"{f}|{p}")
    return np.array(X), np.array(y), np.array(P)


def long_files(month, kind):
    import glob
    return sorted(glob.glob(str(DB / f"{month:02d}" / f"{kind}*rss.csv")))


def uji_logins(df, m, level):
    from data import N_AP, to_strength
    keys = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID"]
    X, y, ph, P = [], [], [], []
    S = to_strength(df.iloc[:, :N_AP].values)
    for key, g in df.groupby(keys):
        idx = g.sort_values("TIMESTAMP").index.values
        rows = df.index.get_indexer(idx)
        for w in windows(rows, m):
            X.append(merge_scans(S[w]))
            b, f = int(df.BUILDINGID.iloc[w[0]]), int(df.FLOOR.iloc[w[0]])
            y.append(b if level == "building" else b * 10 + f)
            ph.append(int(df.PHONEID.iloc[w[0]])); P.append(str(key))
    y = np.array(y)
    if level == "floor":
        u = {v: i for i, v in enumerate(sorted(set(y)))}
        y = np.array([u[v] for v in y])
    return np.array(X), y, np.array(ph), np.array(P)


def strat_half_groups(y, groups, rng):
    """Half of the position groups (stratified by zone) for calibration."""
    cal = np.zeros(len(y), bool)
    for z in np.unique(y):
        gs = np.unique(groups[y == z])
        pick = rng.permutation(gs)[: len(gs) // 2]
        cal |= np.isin(groups, pick) & (y == z)
    return np.where(cal)[0], np.where(~cal)[0]


# =========================================================================
# Rule evaluation
# =========================================================================
def eval_scores(Sc, yc, Se, ye, rng, w=None):
    gc, _ = split_claims(Sc, yc)
    ge, ie = split_claims(Se, ye)
    tau = conformal_tau(gc, ALPHA, rng)
    frr, far = frr_far(ge, ie, tau, rng)
    if w is None:
        curve = [certified_far_randomized(ie, tau, k) for k in K_RANGE]
    else:
        curve = [certified_far_weighted(ie, tau, w, k) for k in K_RANGE]
    return dict(tau=tau, FRR=frr, FAR=far, EER=eer(ge, ie), kstar=kstar_from_curve(curve), curve=curve)


def group_weights(gv, X, y):
    """w_g = 2 for the more reliable half of the groups, else 1 (selection data)."""
    A = gv.accept(X)
    rel = np.zeros(gv.G)
    for g in range(gv.G):
        heard = A[:, g, :].any(1)
        if heard.sum() == 0:
            continue
        right = A[np.arange(len(y)), g, y][heard].mean()
        wrong = (A[:, g, :].sum(1) - A[np.arange(len(y)), g, y])[heard].mean()
        rel[g] = right - wrong
    w = np.ones(gv.G)
    w[np.argsort(-rel)[: gv.G // 2]] = 2.0
    return w


def select_and_eval(Xf, yf, Xs, ys, gs, Xc, yc, Xe, ye, Z, tag, seed=0):
    """Selection on (Xs, ys) halves; thresholds on (Xc, yc); test on (Xe, ye)."""
    rng = np.random.default_rng(seed)
    sc, se = strat_half_groups(ys, gs, rng)
    cands = []
    for G, th, p in itertools.product((5, 10, 15, 20), (0.6, 0.7), (1.0, E)):
        gv = GroupVerifier(partition_spread(Xf, G, seed), Z, 10, th, power=p).fit(Xf, yf)
        V = gv.votes(Xs).astype(float)
        for c in C_GRID:
            S = hybrid_scores(V, c)
            r = eval_scores(S[sc], ys[sc], S[se], ys[se], rng)
            cands.append(dict(G=G, theta=th, power=p, c=c, kstar=r["kstar"], EER=r["EER"]))
    cd = pd.DataFrame(cands)
    cd.to_csv(RES / f"selection_{tag}.csv", index=False)
    fam = {"one-sided": cd[np.isposinf(cd.c)], "symmetric": cd[np.isneginf(cd.c)], "hybrid": cd}
    out, models = [], {}
    for name, d in fam.items():
        b = d.sort_values(["kstar", "EER"], ascending=[False, True]).iloc[0]
        key = (int(b.G), float(b.theta), float(b.power))
        if key not in models:
            models[key] = GroupVerifier(partition_spread(Xf, key[0], seed), Z, 10, key[1],
                                        power=key[2]).fit(Xf, yf)
        gv = models[key]
        Vc, Ve = gv.votes(Xc).astype(float), gv.votes(Xe).astype(float)
        r = eval_scores(hybrid_scores(Vc, b.c), yc, hybrid_scores(Ve, b.c), ye, np.random.default_rng(seed + 1))
        cov = q_alpha(np.minimum(key[0], (Xe > 0).sum(1)))
        out.append(dict(tag=tag, rule=name, G=key[0], theta=key[1], power=key[2], c=float(b.c),
                        FRR=r["FRR"], FAR=r["FAR"], EER=r["EER"], kstar=r["kstar"], bound=cov,
                        **{f"k{k}": r["curve"][k] for k in K_RANGE}))
        if name == "hybrid":
            best = (gv, float(b.c), key)
            w = group_weights(gv, Xs, ys)
            Ac, Ae = gv.accept(Xc), gv.accept(Xe)
            rw = eval_scores(weighted_votes(Ac, w), yc, weighted_votes(Ae, w), ye,
                             np.random.default_rng(seed + 2), w=w)
            out.append(dict(tag=tag, rule="one-sided weighted", G=key[0], theta=key[1], power=key[2],
                            c=np.inf, FRR=rw["FRR"], FAR=rw["FAR"], EER=rw["EER"], kstar=rw["kstar"],
                            bound=cov, **{f"k{k}": rw["curve"][k] for k in K_RANGE}))
    for o in out[-4:]:
        log(f"{tag}: {o['rule']:<20} G={o['G']} th={o['theta']} c={o['c']} FRR={o['FRR']:.3f} "
            f"FAR={o['FAR']:.4f} EER={o['EER']:.4f} k*={o['kstar']} bound={o['bound']}")
    return out, best


# =========================================================================
# Part A
# =========================================================================
def part_A():
    rows, best = [], {}
    # ---- long-term library, month 1 ---------------------------------------
    trn, tst = long_files(1, "trn"), long_files(1, "tst")
    for scheme in ("floor", "quadrant"):
        for m in (1, 3, 5):
            Xf, yf, _ = long_logins(trn, m, scheme)
            Xs, ys, gs = long_logins(tst[:2], m, scheme)
            Xe, ye, _ = long_logins(tst[2:], m, scheme)
            Z = int(yf.max()) + 1
            o, b = select_and_eval(Xf, yf, Xs, ys, gs, Xs, ys, Xe, ye, Z, f"long_{scheme}_m{m}")
            for r in o:
                r.update(dataset="long-term", zones=scheme, m=m, n_eval=len(ye))
            rows += o; best[f"long_{scheme}_m{m}"] = b
            pd.DataFrame(rows).to_csv(RES / "multiscan.csv", index=False)
    rows += part_A_uji()
    pd.DataFrame(rows).to_csv(RES / "multiscan.csv", index=False)
    log("part A done")


def uji_phone_split(df):
    """Test phones: a fixed triple whose removal keeps every building-floor
    zone in the training phones while the test phones still cover all 13
    zones. Selection phones: the pair from the rest that keeps all zones in
    the fit phones and covers the most zones."""
    z = df.BUILDINGID * 10 + df.FLOOR
    allz = set(z)
    test_ph = np.array([7, 14, 19])
    rest = [p for p in sorted(df.PHONEID.unique()) if p not in test_ph]
    best = None
    for a, b in itertools.combinations(rest, 2):
        fit = ~df.PHONEID.isin(list(test_ph) + [a, b])
        if set(z[fit]) != allz:
            continue
        cov = len(set(z[df.PHONEID.isin([a, b])]))
        if best is None or cov > best[0]:
            best = (cov, np.array([a, b]))
    return test_ph, best[1]


def part_A_uji():
    rows = []
    df = pd.read_csv(ROOT / "data/raw/uji/UJIndoorLoc/trainingData.csv")
    test_ph, sel_ph = uji_phone_split(df)
    for level in ("building", "floor"):
        for m in (1, 3, 5):
            X, y, ph, P = uji_logins(df, m, level)
            Z = int(y.max()) + 1
            fit = ~np.isin(ph, test_ph) & ~np.isin(ph, sel_ph)
            sel = np.isin(ph, sel_ph)
            tst_ = np.isin(ph, test_ph)
            cal, ev = strat_half_groups(y[tst_], P[tst_], np.random.default_rng(1))
            Xt, yt = X[tst_], y[tst_]
            o, _ = select_and_eval(X[fit], y[fit], X[sel], y[sel], P[sel], Xt[cal], yt[cal],
                                   Xt[ev], yt[ev], Z, f"uji_{level}_m{m}")
            for r in o:
                r.update(dataset="UJIIndoorLoc", zones=level, m=m, n_eval=len(ev))
            rows += o
    (RES / "uji_phone_split.json").write_text(json.dumps(dict(test=[int(p) for p in test_ph],
                                                              sel=[int(p) for p in sel_ph])))
    return rows


# =========================================================================
# Part B: cascade with kNN
# =========================================================================
def part_B(n_attack=200, k_max=8):
    sel = pd.read_csv(RES / "multiscan.csv")
    rows = []
    trn, tst = long_files(1, "trn"), long_files(1, "tst")
    for m in (1, 3):
        Xf, yf, _ = long_logins(trn, m, "floor")
        Xc, yc, _ = long_logins(tst[:2], m, "floor")
        Xe, ye, _ = long_logins(tst[2:], m, "floor")
        Z = 2
        h = sel[(sel.tag == f"long_floor_m{m}") & (sel.rule == "hybrid")].iloc[0]
        gv = GroupVerifier(partition_spread(Xf, int(h.G), 0), Z, 10, float(h.theta),
                           power=float(h.power)).fit(Xf, yf)
        c = float(h.c)
        knn = KNeighborsClassifier(n_neighbors=10).fit(Xf / 105.0, yf)
        kp = lambda X: knn.predict_proba(X / 105.0)
        rng = np.random.default_rng(5)
        Sc = hybrid_scores(gv.votes(Xc).astype(float), c)
        Se = hybrid_scores(gv.votes(Xe).astype(float), c)
        Kc, Ke = kp(Xc), kp(Xe)
        tA = conformal_tau(split_claims(Sc, yc)[0], ALPHA / 2, rng)
        tB = conformal_tau(split_claims(Kc, yc)[0], ALPHA / 2, rng, 1e-9)
        tA1 = conformal_tau(split_claims(Sc, yc)[0], ALPHA, rng)
        ge, ie = split_claims(Se, ye)
        gk, ik = split_claims(Ke, ye)
        ug, ui = rng.uniform(size=len(ge)), rng.uniform(size=len(ie))
        vg, vi = 1e-9 * rng.uniform(size=len(gk)), 1e-9 * rng.uniform(size=len(ik))
        frr_c = np.mean((ge + ug < tA) | (gk + vg < tB))
        far_c = np.mean((ie + ui >= tA) & (ik + vi >= tB))
        cert_c = [certified_far_randomized(ie, tA, k) for k in K_RANGE]
        cert_1 = [certified_far_randomized(ie, tA1, k) for k in K_RANGE]
        # attacks
        prof = zone_profiles(Xf, yf, Z)
        imp = [(i, z) for i in range(len(ye)) for z in range(Z) if z != ye[i]]
        pick = [imp[j] for j in rng.choice(len(imp), min(n_attack, len(imp)), replace=False)]
        res = {"AP-Certified alone": np.zeros(k_max + 1), "cascade": np.zeros(k_max + 1)}
        log(f"m={m}: tA={tA:.3f} tB={tB:.6f} tA1={tA1:.3f}")
        for i, z in pick:
            u, u2 = rng.uniform(), 1e-9 * rng.uniform()
            accA1 = lambda X, z, u=u: hybrid_scores(gv.votes(X).astype(float), c)[:, z] + u >= tA1
            accC = lambda X, z, u=u, u2=u2: (hybrid_scores(gv.votes(X).astype(float), c)[:, z] + u >= tA) & (kp(X)[:, z] + u2 >= tB)
            ed = group_single_edits(gv, Xe[i], z, rate=prof[0])
            s2 = lambda X, z: kp(X)[:, z] + 1e-3 * gv.soft_score(X, z)
            res["AP-Certified alone"] += cascade_greedy_attack(ed, Xe[i], z, accA1, s2, k_max)
            res["cascade"] += cascade_greedy_attack(ed, Xe[i], z, accC, s2, k_max)
        for name, s in res.items():
            for k in range(k_max + 1):
                rows.append(dict(m=m, model=name, k=k, attack=s[k] / len(pick),
                                 certified=(cert_1 if name.startswith("AP") else cert_c)[k] if k < len(K_RANGE) else np.nan))
        rows.append(dict(m=m, model="cascade-summary", FRR=frr_c, FAR=far_c, kstar=kstar_from_curve(cert_c),
                         kstar_single=kstar_from_curve(cert_1)))
        log(f"cascade m={m}: FRR={frr_c:.3f} FAR={far_c:.4f} k*={kstar_from_curve(cert_c)} | "
            + " ".join(f"{k}:{res['AP-Certified alone'][k]/len(pick):.2f}/{res['cascade'][k]/len(pick):.2f}"
                       for k in range(k_max + 1)))
        pd.DataFrame(rows).to_csv(RES / "cascade.csv", index=False)
    log("part B done")


# =========================================================================
# Part C: online calibration under drift
# =========================================================================
def part_C(m=1, eta=0.05, k0=1, win=500, alarm_rate=0.10, refresh=25):
    """Stream months 1..25 in order. Each confirmed genuine attempt also yields
    an impostor claim (its scan claiming the other floor), so the server can
    estimate the certified FAR at radius k0 on recent scans without attack data.

    Methods
      static          : month-1 conformal threshold
      online          : quantile tracking (Theorem 4), FRR only
      online+floor    : threshold never below tau_need, the smallest threshold
                        with certified FAR(k0) <= 1% on the last `win` scans
      online+floor+retrain : as above; if the floor binds on more than
                        `alarm_rate` of the last `win` attempts, retrain on the
                        month-1 sweep + half of the current month's sweep and
                        recalibrate on the other half
    """
    sel = pd.read_csv(RES / "multiscan.csv")
    h = sel[(sel.tag == f"long_floor_m{m}") & (sel.rule == "hybrid")].iloc[0]
    G, th, p, c = int(h.G), float(h.theta), float(h.power), float(h.c)
    Z = 2
    trn1, tst1 = long_files(1, "trn"), long_files(1, "tst")
    Xf, yf, _ = long_logins(trn1, m, "floor")
    Xc, yc, _ = long_logins(tst1[:2], m, "floor")

    def fit(X, y):
        return GroupVerifier(partition_spread(X, G, 0), Z, 10, th, power=p).fit(X, y)

    def tau_need(imp):
        lo, hi = -G, 2.0 * G + 2
        if np.clip(imp + k0 + 1 - lo, 0, 1).mean() <= 0.01:
            return lo
        for _ in range(30):
            mid = (lo + hi) / 2
            if np.clip(imp + k0 + 1 - mid, 0, 1).mean() <= 0.01:
                hi = mid
            else:
                lo = mid
        return hi

    gv0 = fit(Xf, yf)
    rng = np.random.default_rng(11)
    S1 = hybrid_scores(gv0.votes(Xc).astype(float), c)
    tau_static = conformal_tau(split_claims(S1, yc)[0], ALPHA, rng)
    imp0 = list(split_claims(S1, yc)[1][-win:])
    names = ["static", "online", "online+floor", "online+floor+retrain"]
    st = {n: dict(tau=tau_static, gv=gv0, imp=list(imp0), floor=tau_need(np.array(imp0)),
                  hist=[], retrains=[]) for n in names}
    rows = []
    for month in range(1, 26):
        files = long_files(month, "tst")[2:] if month == 1 else long_files(month, "tst")
        Xe, ye, _ = long_logins(files, m, "floor")
        S_base = hybrid_scores(gv0.votes(Xe).astype(float), c)
        for n in names:
            d = st[n]
            Sm = S_base if d["gv"] is gv0 else hybrid_scores(d["gv"].votes(Xe).astype(float), c)
            err = np.zeros(len(ye)); fa = np.zeros(len(ye)); cert = np.zeros(len(ye)); bind = 0
            for t in range(len(ye)):
                tau = d["tau"]
                sg, si = Sm[t, ye[t]] + rng.uniform(), Sm[t, 1 - ye[t]]
                err[t] = sg < tau
                fa[t] = si + rng.uniform() >= tau
                cert[t] = np.clip(si + k0 + 1 - tau, 0, 1)
                if n == "static":
                    continue
                new = tau - eta * (err[t] - ALPHA)
                d["imp"].append(si); d["imp"] = d["imp"][-win:]
                if n != "online":
                    if t % refresh == 0:
                        d["floor"] = tau_need(np.array(d["imp"]))
                    b = new < d["floor"]
                    bind += b
                    new = max(new, d["floor"])
                    d["hist"].append(b); d["hist"] = d["hist"][-win:]
                d["tau"] = new
                if n == "online+floor+retrain" and len(d["hist"]) >= win and np.mean(d["hist"]) > alarm_rate:
                    Xr, yr, gr = long_logins(long_files(month, "trn"), m, "floor")
                    h1, h2 = strat_half_groups(yr, gr, np.random.default_rng(month))
                    d["gv"] = fit(np.vstack([Xf, Xr[h1]]), np.r_[yf, yr[h1]])
                    Sr = hybrid_scores(d["gv"].votes(Xr[h2]).astype(float), c)
                    d["tau"] = conformal_tau(split_claims(Sr, yr[h2])[0], ALPHA, rng)
                    d["imp"] = list(split_claims(Sr, yr[h2])[1])[-win:]
                    d["floor"] = tau_need(np.array(d["imp"]))
                    d["hist"] = []
                    d["retrains"].append((month, t))
                    Sm = hybrid_scores(d["gv"].votes(Xe).astype(float), c)
            rows.append(dict(month=month, method=n, FRR=err.mean(), FAR=fa.mean(), certFAR_k0=cert.mean(),
                             floor_bind=bind / len(ye), n=len(ye), tau_end=d["tau"]))
        log(f"month {month}: " + " | ".join(f"{r['method']}: FRR={r['FRR']:.3f} FAR={r['FAR']:.4f} "
                                            f"cert={r['certFAR_k0']:.4f} bind={r['floor_bind']:.2f}"
                                            for r in rows[-4:]))
        pd.DataFrame(rows).to_csv(RES / f"online_m{m}.csv", index=False)
    meta = dict(m=m, eta=eta, k0=k0, win=win, alarm_rate=alarm_rate, G=G, theta=th, power=p, c=c,
                tau_static=float(tau_static),
                retrains=[(int(a), int(b)) for a, b in st["online+floor+retrain"]["retrains"]])
    (RES / f"online_m{m}.json").write_text(json.dumps(meta, indent=2))
    log(f"part C done: {meta}")


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "A":
        part_A()
    elif part == "Auji":
        prev = pd.read_csv(RES / "multiscan.csv")
        prev = prev[prev.dataset != "UJIIndoorLoc"]
        new = pd.DataFrame(part_A_uji())
        pd.concat([prev, new]).to_csv(RES / "multiscan.csv", index=False)
    elif part == "Along":
        prev = pd.read_csv(RES / "multiscan.csv")
        uji = prev[prev.dataset == "UJIIndoorLoc"]
        rows = []
        trn, tst = long_files(1, "trn"), long_files(1, "tst")
        for scheme in ("floor", "quadrant"):
            for m in (1, 3, 5):
                Xf, yf, _ = long_logins(trn, m, scheme)
                Xs, ys, gs = long_logins(tst[:2], m, scheme)
                Xe, ye, _ = long_logins(tst[2:], m, scheme)
                o, _ = select_and_eval(Xf, yf, Xs, ys, gs, Xs, ys, Xe, ye, int(yf.max()) + 1, f"long_{scheme}_m{m}")
                for r in o:
                    r.update(dataset="long-term", zones=scheme, m=m, n_eval=len(ye),
                             heard_eval=float((Xe > 0).sum(1).mean()))
                rows += o
        pd.concat([pd.DataFrame(rows), uji]).to_csv(RES / "multiscan.csv", index=False)
    elif part == "B":
        part_B()
    elif part == "C":
        part_C(m=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
