"""Uncertified defense baselines under the same adaptive attacks.

D1  Adversarially trained MLP: the MLP of the main study is refined on
    training scans perturbed by a first-order L0 attack towards a random
    other zone (1-6 AP edits, label unchanged), following the adversarial
    training principle of Madry et al. (2018).
D2  Consistency-filtered LightGBM: a threshold-based malicious-AP check.
    A heard AP is inconsistent with the claimed zone if it is (almost) never
    heard there or its reading lies outside the zone's 1st-99th percentile
    range (+-5 dB). Claims whose fraction of inconsistent APs exceeds a
    calibrated threshold are rejected; otherwise LightGBM scores the scan with
    the inconsistent APs removed. Both thresholds are calibrated at alpha/2.

Both are attacked with the beam search of the main study, which evaluates the
complete pipeline and is therefore adaptive to the filter; the adversarially
trained MLP is also attacked with the gradient attack. Settings and impostor
claims are those of the main attack experiments (UJIIndoorLoc building zones,
split 0; library floors, month 1).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
from attacks import beam_attack, gradient_attack_mlp, mlp_logp_and_grad, zone_profiles  # noqa: E402
from core import conformal_tau, eer  # noqa: E402
from data import load_uji  # noqa: E402
from data_long import load_month, zones_from_coords  # noqa: E402
from experiments import ALPHA, SCALE, baseline_scores, fit_baselines, split_claims  # noqa: E402
from run_long import load_tst_files  # noqa: E402
from run_uji import strat_split  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results" / "defenses"
RES.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda s: print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
K_ATT = 6


# ------------------------------------------------------------------ D1
def l0_adv_batch(mlp, X, y, Z, prof, rng, k_max=6):
    """First-order L0 attack towards a random wrong zone (vectorised)."""
    rate, _, q90 = prof
    Xa = X.copy()
    tgt = (y + rng.integers(1, Z, len(y))) % Z
    ks = rng.integers(1, k_max + 1, len(y))
    used = np.zeros_like(X, bool)
    for step in range(k_max):
        act = np.where(ks > step)[0]
        if len(act) == 0:
            break
        for z in np.unique(tgt[act]):
            ii = act[tgt[act] == z]
            _, g = mlp_logp_and_grad(mlp, Xa[ii] / SCALE, z)
            up = np.where(rate[z] > 0, q90[z], 0.0)[None, :]
            gain_up = g * (up - Xa[ii]) / SCALE
            gain_dn = np.where(Xa[ii] > 0, g * (0.0 - Xa[ii]) / SCALE, -np.inf)
            best = np.maximum(gain_up, gain_dn)
            best[used[ii]] = -np.inf
            j = best.argmax(1)
            newv = np.where(gain_up[np.arange(len(ii)), j] >= gain_dn[np.arange(len(ii)), j], up[0, j], 0.0)
            Xa[ii, j] = newv
            used[ii, j] = True
    return Xa


def adv_train_mlp(X, y, Z, prof, seed=0, epochs=15, batch=4000):
    rng = np.random.default_rng(seed)
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, early_stopping=True,
                        random_state=seed).fit(X / SCALE, y)
    mlp.early_stopping = False
    mlp.best_loss_ = np.inf
    for e in range(epochs):
        idx = rng.choice(len(X), min(batch, len(X)), replace=False)
        Xa = l0_adv_batch(mlp, X[idx], y[idx], Z, prof, rng)
        for _ in range(3):
            mlp.partial_fit(np.vstack([X[idx], Xa]) / SCALE, np.r_[y[idx], y[idx]])
        acc = (mlp.predict(Xa / SCALE) == y[idx]).mean()
        log(f"  adv epoch {e}: accuracy on fresh adversarial scans {acc:.3f}")
    return mlp


# ------------------------------------------------------------------ D2
class ConsistencyFilter:
    def __init__(self, X, y, Z, margin=5.0, min_rate=0.01):
        self.Z = Z
        self.rate = np.zeros((Z, X.shape[1])); self.lo = np.zeros_like(self.rate); self.hi = np.zeros_like(self.rate)
        for z in range(Z):
            Xz = X[y == z]; H = Xz > 0
            self.rate[z] = H.mean(0)
            for j in np.where(H.any(0))[0]:
                v = Xz[H[:, j], j]
                self.lo[z, j], self.hi[z, j] = np.percentile(v, 1) - margin, np.percentile(v, 99) + margin
        self.min_rate = min_rate

    def inconsistent(self, X, z):
        H = X > 0
        bad = H & ((self.rate[z] < self.min_rate)[None, :] | (X < self.lo[z]) | (X > self.hi[z]))
        return bad

    def anomaly(self, X, z):
        H = (X > 0).sum(1)
        return self.inconsistent(X, z).sum(1) / np.maximum(H, 1)

    def clean(self, X, z):
        Xc = X.copy(); Xc[self.inconsistent(X, z)] = 0.0
        return Xc


def filtered_scores(flt, model, X, Z):
    """(anomaly, classifier score) matrices over all claimed zones."""
    A = np.zeros((len(X), Z)); P = np.zeros((len(X), Z))
    for z in range(Z):
        A[:, z] = flt.anomaly(X, z)
        P[:, z] = baseline_scores(model, flt.clean(X, z), Z)[:, z]
    return A, P


# ------------------------------------------------------------------ evaluation
def evaluate(name, score_fn, accept_fn, Xc, yc, Xe, ye, Z, pick, prof, mlp=None, thr=None):
    """Clean FRR/FAR/EER on the evaluation scans and attack success on `pick`."""
    Se = np.stack([score_fn(Xe, z) for z in range(Z)], 1)
    Ae = np.stack([accept_fn(Xe, z) for z in range(Z)], 1)
    g, i = split_claims(Se, ye)
    ga, ia = split_claims(Ae.astype(float), ye)
    row = dict(model=name, FRR=1 - ga.mean(), FAR=ia.mean(), EER=eer(g, i))
    S = np.zeros(K_ATT + 1)
    for idx, z in pick:
        s = beam_attack(score_fn, accept_fn, Xe[idx], z, prof, K_ATT)
        if mlp is not None:
            s = s | gradient_attack_mlp(mlp, thr, Xe[idx], z, K_ATT)
        S += s
    for k in range(K_ATT + 1):
        row[f"att_k{k}"] = S[k] / len(pick)
    log(f"{name}: FRR={row['FRR']:.3f} FAR={row['FAR']:.4f} EER={row['EER']:.4f} attack " +
        " ".join(f"{S[k]/len(pick):.3f}" for k in range(K_ATT + 1)))
    return row


def run_setting(label, Xtr, ytr, Xc, yc, Xe, ye, Z, pick):
    prof = zone_profiles(Xtr, ytr, Z)
    rng = np.random.default_rng(11)
    base = fit_baselines(Xtr, ytr)
    rows = []
    # reference: plain MLP and LightGBM, calibrated at alpha on the calibration scans
    for n in ("MLP", "LightGBM"):
        m = base[n]
        t = conformal_tau(split_claims(baseline_scores(m, Xc, Z), yc)[0], ALPHA, rng, 1e-9)
        rows.append(evaluate(n, lambda X, z, m=m: baseline_scores(m, X, Z)[:, z],
                             lambda X, z, m=m, t=t: baseline_scores(m, X, Z)[:, z] >= t,
                             Xc, yc, Xe, ye, Z, pick, prof, mlp=m if n == "MLP" else None, thr=t))
    # D1
    log(f"{label}: adversarial training")
    at = adv_train_mlp(Xtr, ytr, Z, prof)
    t = conformal_tau(split_claims(baseline_scores(at, Xc, Z), yc)[0], ALPHA, rng, 1e-9)
    rows.append(evaluate("Adversarially trained MLP", lambda X, z: baseline_scores(at, X, Z)[:, z],
                         lambda X, z: baseline_scores(at, X, Z)[:, z] >= t, Xc, yc, Xe, ye, Z, pick, prof,
                         mlp=at, thr=t))
    # D2
    flt = ConsistencyFilter(Xtr, ytr, Z)
    lg = base["LightGBM"]
    Ac, Pc = filtered_scores(flt, lg, Xc, Z)
    ga, _ = split_claims(Ac, yc); gp, _ = split_claims(Pc, yc)
    rho = -conformal_tau(-ga, ALPHA / 2, rng, 1e-9)          # upper quantile of genuine anomaly
    tp = conformal_tau(gp, ALPHA / 2, rng, 1e-9)

    def sc2(X, z):
        a = flt.anomaly(X, z); p = baseline_scores(lg, flt.clean(X, z), Z)[:, z]
        return p - np.maximum(a - rho, 0) * 10

    def ac2(X, z):
        return (flt.anomaly(X, z) <= rho) & (baseline_scores(lg, flt.clean(X, z), Z)[:, z] >= tp)
    rows.append(evaluate("LightGBM + consistency filter", sc2, ac2, Xc, yc, Xe, ye, Z, pick, prof))
    rows[-1]["rho"] = rho
    for r in rows:
        r["setting"] = label
    return rows


def main():
    rows = []
    # UJIIndoorLoc building zones: split and impostor claims of the main attack experiment
    d = load_uji(); tr, te = d["train"], d["test"]
    yb_tr, yb_te = tr["building"], te["building"]
    rng = np.random.default_rng(1000)
    cal, ev = strat_split(yb_te, rng)
    n_gen = len(cal)
    for _ in range(4):                       # same random-number use as the main experiment
        rng.uniform(0, 1, n_gen)
    imp = [(i, z) for i in ev for z in range(3) if z != yb_te[i]]
    pick = [imp[j] for j in rng.choice(len(imp), 200, replace=False)]
    # attack indices refer to the evaluation set
    pos = {i: n for n, i in enumerate(ev)}
    pick = [(pos[i], z) for i, z in pick]
    rows += run_setting("UJI building", tr["X"], yb_tr, te["X"][cal], yb_te[cal], te["X"][ev], yb_te[ev], 3, pick)
    pd.DataFrame(rows).to_csv(RES / "defenses.csv", index=False)
    # library floors, month 1
    Xtr, Ctr, _ = load_month(1, "trn"); ytr = zones_from_coords(Ctr, "floor")
    t1 = load_tst_files(1)
    Xs = np.vstack([t[0] for t in t1[:2]]); ys = zones_from_coords(np.vstack([t[1] for t in t1[:2]]), "floor")
    Xe1 = np.vstack([t[0] for t in t1[2:]]); ye1 = zones_from_coords(np.vstack([t[1] for t in t1[2:]]), "floor")
    rng = np.random.default_rng(7)
    rng.uniform(0, 1, len(ys))               # same random-number use as the main experiment
    imp = [(i, z) for i in range(len(ye1)) for z in range(2) if z != ye1[i]]
    pick = [imp[j] for j in rng.choice(len(imp), 200, replace=False)]
    rows += run_setting("Library floor", Xtr, ytr, Xs, ys, Xe1, ye1, 2, pick)
    pd.DataFrame(rows).to_csv(RES / "defenses.csv", index=False)
    log("finished")


if __name__ == "__main__":
    main()
