"""Coverage test of the population-level certified FAR bound (Proposition 3)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import certified_far_upper  # noqa: E402


def test_coverage(trials=2000, n=800, beta=0.05):
    rng = np.random.default_rng(0)
    ks = list(range(0, 4))
    tau = 6.0
    miss = 0
    for _ in range(trials):
        v = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.6, 0.2, 0.1, 0.07, 0.03])
        ups = certified_far_upper(v, tau, ks, rng, beta)
        # true population certified FAR: E[clip(v + k + 1 - tau, 0, 1)]
        vals = np.array([0, 1, 2, 3, 4]); pv = np.array([0.6, 0.2, 0.1, 0.07, 0.03])
        true = [float((np.clip(vals + k + 1 - tau, 0, 1) * pv).sum()) for k in ks]
        miss += any(t > u + 1e-12 for t, u in zip(true, ups))
    rate = miss / trials
    assert rate <= beta + 0.015, rate
    print(f"simultaneous miss rate {rate:.4f} (target <= {beta})")


if __name__ == "__main__":
    test_coverage()
