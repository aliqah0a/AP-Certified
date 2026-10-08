"""Brute-force soundness of the hybrid score, weighted votes, multi-scan
merge, and the online-threshold FRR bound."""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import (GroupVerifier, hybrid_scores, merge_scans, topk_weight,
                  weighted_votes, OnlineThreshold)
from test_core import world

GRID = [0.0, 15.0, 40.0, 70.0, 100.0]


def test_hybrid_and_weighted():
    viol = checked = 0
    for seed, c in itertools.product(range(3), (-np.inf, -2.0, 0.0, 3.0, np.inf)):
        X, y, Z = world(seed)
        gv = GroupVerifier(np.array([0, 1, 2, 0, 1, 2]), Z, K=5, theta=0.6).fit(X, y)
        w = np.array([2.0, 1.0, 3.0])
        rng = np.random.default_rng(seed)
        T = np.clip(X[rng.choice(len(X), 5, replace=False)] + rng.normal(0, 5, (5, 6)), 0, 105)
        A0 = gv.accept(T)
        S0 = hybrid_scores(A0.sum(1), c)
        W0 = weighted_votes(A0, w)
        for i, z, k in itertools.product(range(5), range(Z), (1, 2)):
            for aps in itertools.combinations(range(6), k):
                for vals in itertools.product(GRID, repeat=k):
                    xa = T[i].copy(); xa[list(aps)] = vals
                    A = gv.accept(xa[None])
                    checked += 1
                    viol += hybrid_scores(A.sum(1), c)[0, z] > S0[i, z] + k + 1e-9
                    viol += weighted_votes(A, w)[0, z] > W0[i, z] + topk_weight(w, k) + 1e-9
    assert checked and viol == 0, (viol, checked)
    return checked


def test_merge_l0():
    """Editing k APs in any scans changes at most k merged entries; offsets of
    delta per scan move each merged entry by at most delta."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        S = rng.uniform(0, 105, (4, 8)) * (rng.uniform(size=(4, 8)) > 0.4)
        base = merge_scans(S)
        k = rng.integers(1, 3)
        aps = rng.choice(8, k, replace=False)
        S2 = S.copy()
        for j in aps:
            rows = rng.uniform(size=4) > 0.5
            S2[rows, j] = rng.uniform(0, 105, rows.sum())
        assert (merge_scans(S2) != base).sum() <= k
        d = 3.0
        S3 = np.where(S > 0, np.clip(S + rng.uniform(-d, d, S.shape), 1, 105), 0)
        assert np.all(np.abs(merge_scans(S3) - base) <= d + 1e-4)
    return True


def test_online_threshold():
    """Long-run FRR within the deterministic bound for an adversarial drift."""
    rng = np.random.default_rng(1)
    B, eta, alpha, T = 10, 0.05, 0.05, 20000
    ot = OnlineThreshold(5.0, alpha, eta)
    errs = []
    for t in range(T):
        mu = 8 if t < T // 3 else (3 if t < 2 * T // 3 else 9)   # abrupt drift
        s = np.clip(rng.normal(mu, 1.5), 0, B) // 1 + rng.uniform()
        e = s < ot.tau
        errs.append(e); ot.update(e)
    gap = abs(np.mean(errs) - alpha)
    assert gap <= (B + 1 + 2 * eta) / (eta * T) + 1e-12, gap
    return np.mean(errs)


if __name__ == "__main__":
    print("hybrid/weighted checked:", test_hybrid_and_weighted())
    print("merge:", test_merge_l0())
    print("online FRR:", test_online_threshold())
