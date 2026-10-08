"""Brute-force check of the one-sided and symmetric certificates.

Small synthetic world: 6 APs, 3 groups, 3 zones. For every test scan and
claimed zone that a certificate marks as rejected at radius k, we enumerate
every way to change up to k AP readings (values from a finite grid) and
confirm the verifier never accepts.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import (GroupVoteVerifier, one_sided_certified_reject,  # noqa: E402
                    symmetric_certified_reject, symmetric_margin)


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    n_aps, Z = 6, 3
    centers = rng.uniform(0, 40, size=(Z, n_aps)) * (rng.uniform(size=(Z, n_aps)) > 0.3)
    X, y = [], []
    for z in range(Z):
        for _ in range(30):
            x = centers[z] + rng.normal(0, 3, n_aps)
            x[centers[z] == 0] = 0
            X.append(np.clip(x, 0, None)); y.append(z)
    return np.array(X), np.array(y), Z


def check(seed, theta=None):
    X, y, Z = make_world(seed)
    groups = np.array([0, 1, 2, 0, 1, 2])
    clf = GroupVoteVerifier(groups, Z, n_neighbors=3, theta=theta).fit(X, y)
    rng = np.random.default_rng(seed + 100)
    test = X[rng.choice(len(X), 8, replace=False)] + rng.normal(0, 2, (8, 6))
    test = np.clip(test, 0, None)
    grid = [0.0, 10.0, 25.0, 40.0]
    V0 = clf.votes(test)
    violations, checked = 0, 0
    for i in range(len(test)):
        for z in range(Z):
            for k in (1, 2):
                for tau in (2, 3):
                    one = one_sided_certified_reject(V0[i, z], tau, k)
                    for m in (0, 1, 2):
                        sym = symmetric_certified_reject(V0[i:i + 1], np.array([z]), m, k)[0]
                        if not (one or sym):
                            continue
                        for aps in itertools.combinations(range(6), k):
                            for vals in itertools.product(grid, repeat=k):
                                xa = test[i].copy()
                                xa[list(aps)] = vals
                                Va = clf.votes(xa[None])
                                checked += 1
                                if one and Va[0, z] >= tau:
                                    violations += 1
                                if sym:
                                    mg, _ = symmetric_margin(Va, np.array([z]))
                                    if mg[0] >= m:
                                        violations += 1
    return violations, checked


def test_certificates_hold():
    total_v, total_c = 0, 0
    for s in range(3):
        for th in (None, 0.7):
            v, c = check(s, th)
            total_v += v; total_c += c
    assert total_c > 0
    assert total_v == 0, f"{total_v} violations"


if __name__ == "__main__":
    for s in range(3):
        for th in (None, 0.7):
            print("seed", s, "theta", th, "violations/checked =", check(s, th))
