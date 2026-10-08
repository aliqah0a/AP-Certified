"""Soundness tests for core.py certificates and conformal calibration."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from core import (GroupVerifier, HierarchicalVerifier, conformal_tau,  # noqa: E402
                  conformal_tau_pac)


def world(seed, n_aps=6, Z=3, n=40):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=(Z, n_aps)) * (rng.uniform(size=(Z, n_aps)) > 0.3)
    X, y = [], []
    for z in range(Z):
        for _ in range(n):
            x = centers[z] + rng.normal(0, 6, n_aps)
            x[centers[z] == 0] = 0
            X.append(np.clip(x, 0, 105)); y.append(z)
    return np.array(X, dtype=np.float32), np.array(y), Z


def test_k_certificate_any_theta():
    """v + k < tau  =>  no k-AP change reaches tau (theta below and above 1/2)."""
    viol = checked = 0
    grid = [0.0, 15.0, 40.0, 70.0, 100.0]
    for seed, theta, power in itertools.product(range(3), (0.3, 0.5, 0.8), (1.0, 2.718)):
        X, y, Z = world(seed)
        gv = GroupVerifier(np.array([0, 1, 2, 0, 1, 2]), Z, K=5, theta=theta, power=power).fit(X, y)
        rng = np.random.default_rng(seed)
        T = np.clip(X[rng.choice(len(X), 6, replace=False)] + rng.normal(0, 5, (6, 6)), 0, 105)
        V = gv.votes(T)
        for i, z, k in itertools.product(range(len(T)), range(Z), (1, 2)):
            tau = V[i, z] + k + 1          # tightest tau the certificate covers
            for aps in itertools.combinations(range(6), k):
                for vals in itertools.product(grid, repeat=k):
                    xa = T[i].copy(); xa[list(aps)] = vals
                    checked += 1
                    viol += gv.votes(xa[None])[0, z] >= tau
    assert checked > 0 and viol == 0, (viol, checked)
    return checked


def test_delta_certificate():
    """If possible_accept is False, no x' within the delta box makes the group accept."""
    viol = checked = 0
    for seed, delta, power in itertools.product(range(3), (2.0, 5.0, 10.0), (1.0, 2.718)):
        X, y, Z = world(seed)
        gv = GroupVerifier(np.array([0, 1, 2, 0, 1, 2]), Z, K=5, theta=0.6, power=power).fit(X, y)
        rng = np.random.default_rng(100 + seed)
        T = np.clip(X[rng.choice(len(X), 10, replace=False)] + rng.normal(0, 8, (10, 6)), 0, 105)
        P = gv.possible_accept(T, delta) & gv.possible_accept_pairwise(T, delta)
        for i in range(len(T)):
            heard = T[i] > 0
            lo = np.where(heard, np.maximum(T[i] - delta, 1), 0)
            hi = np.where(heard, np.minimum(T[i] + delta, 105), 0)
            # corners of the box plus random interior points
            S = [np.where(np.array(c), hi, lo) for c in itertools.product([0, 1], repeat=6)]
            S += list(rng.uniform(lo, hi, size=(200, 6)))
            S = np.array(S, dtype=np.float32)
            A = gv.accept(S)
            for g, z in itertools.product(range(gv.G), range(Z)):
                if not P[i, g, z]:
                    checked += len(S)
                    viol += int(A[:, g, z].sum())
    assert checked > 0 and viol == 0, (viol, checked)
    return checked


def test_delta_zero_superset():
    X, y, Z = world(1)
    gv = GroupVerifier(np.array([0, 1, 2, 0, 1, 2]), Z, K=5, theta=0.6).fit(X, y)
    A = gv.accept(X)
    P = gv.possible_accept(X, 0.0)
    assert not (A & ~P).any()
    P2 = gv.possible_accept_pairwise(X, 0.0)
    assert not (A & ~P2).any()


def test_conformal_coverage():
    """Empirical FRR over many exchangeable splits stays <= alpha (+ MC noise)."""
    rng = np.random.default_rng(0)
    alpha, frr, frr_pac, bad_pac = 0.05, [], [], 0
    for _ in range(2000):
        s = rng.binomial(8, 0.7, size=300).astype(float)   # integer scores with ties
        cal, test = s[:150], s[150:]
        tau = conformal_tau(cal, alpha, rng)
        f = np.mean(test + rng.uniform(0, 1, len(test)) < tau)
        frr.append(f)
        tp = conformal_tau_pac(cal, alpha, 0.1, rng)
        fp = np.mean(test + rng.uniform(0, 1, len(test)) < tp)
        frr_pac.append(fp)
    assert np.mean(frr) <= alpha + 0.005, np.mean(frr)
    return np.mean(frr), np.mean(frr_pac)


def test_hierarchical_k_certificate():
    rng = np.random.default_rng(3)
    X, b = [], []
    Xa, ya, _ = world(5, n_aps=8, Z=2, n=50)
    Xb, yb, _ = world(6, n_aps=8, Z=2, n=50)
    Xa[:, 4:] = 0; Xb[:, :4] = 0                     # disjoint AP sets per building
    X = np.vstack([Xa, Xb]); building = np.r_[np.zeros(100, int), np.ones(100, int)]
    floor = np.r_[ya, yb]
    hv = HierarchicalVerifier(2, 2, K=5, thetaB=0.5, thetaF=0.4).fit(X, building, floor)
    T = np.clip(X[rng.choice(len(X), 5, replace=False)] + rng.normal(0, 5, (5, 8)), 0, 105)
    S = hv.claim_scores(T)
    viol = checked = 0
    for i, bb, ff in itertools.product(range(5), range(2), range(2)):
        tau = S[i, bb, ff] + 2                       # k = 1
        for j in range(8):
            for v in (0.0, 20.0, 60.0, 100.0):
                xa = T[i].copy(); xa[j] = v
                checked += 1
                viol += hv.claim_scores(xa[None])[0, bb, ff] >= tau
    assert viol == 0, viol
    return checked


def test_pairwise_tighter():
    X, y, Z = world(2)
    gv = GroupVerifier(np.array([0, 1, 2, 0, 1, 2]), Z, K=5, theta=0.6).fit(X, y)
    a = gv.possible_accept(X[:40], 6.0).sum()
    b = (gv.possible_accept(X[:40], 6.0) & gv.possible_accept_pairwise(X[:40], 6.0)).sum()
    return a, b


if __name__ == "__main__":
    print("possible accepts (interval, combined):", test_pairwise_tighter())
    print("k-certificate checked:", test_k_certificate_any_theta())
    print("delta-certificate checked:", test_delta_certificate())
    test_delta_zero_superset(); print("delta=0 superset: ok")
    print("conformal mean FRR (marginal, PAC):", test_conformal_coverage())
    print("hierarchical checked:", test_hierarchical_k_certificate())
