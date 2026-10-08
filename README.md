# AP-Certified

Code for the paper *AP-Certified: Certified Zero-Effort Wi-Fi Two-Factor Authentication Against
Rogue and Compromised Access Points*.

AP-Certified is a Wi-Fi fingerprint verifier for zero-effort second-factor authentication. It
partitions the access points (APs) into disjoint groups, classifies each group separately and
accepts a claimed zone only when enough groups vote for it. This gives

* a certified bound on the false acceptance rate against any adversary that controls `k` APs and
  shifts all other readings by at most `delta` dB,
* a conformal bound on the false rejection rate at the same threshold, and an online variant with
  a security floor for drift,
* a population-level version of the certificate (Clopper-Pearson), and
* a pre-deployment check that estimates, from a short sweep, whether a building can be certified.

## Repository layout

| Path | Content |
|---|---|
| `src/core.py` | AP partitions (random, co-occurrence spreading), group verifier, `(k, delta)` certificates, hybrid and weighted scores, multi-scan merging, conformal and online thresholds, hierarchical verifier |
| `src/attacks.py` | beam search, MLP gradient attack, group-wise search, cascade attack |
| `src/experiments.py` | baselines (kNN, LightGBM, MLP), evaluation helpers |
| `src/data*.py` | dataset loaders |
| `src/run_*.py` | experiments (see `reproduce.sh` for the order) |
| `src/run_physics.py` | path-loss model fitted at the surveyed AP positions; rogue-spoofer and compromised-AP attacks |
| `src/run_check.py`, `src/run_check2.py` | pre-deployment certifiability check, sweep sizes, population-level (Clopper-Pearson) radii |
| `src/run_defenses.py` | uncertified defense baselines (adversarially trained MLP, consistency-filtered LightGBM) |
| `src/analyze_*.py`, `src/dataset_table.py` | tables and figures |
| `tests/` | brute-force soundness tests of every certificate |
| `results/` | CSV/JSON outputs used in the paper |

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 tests/test_core.py          # soundness tests, no data needed
```

To reproduce the paper, place the datasets as described in `data/README.md` and run

```bash
./reproduce.sh
```

The tables and figures are written to `paper/tables/` and `paper/figures/`. The precomputed outputs
in `results/` let the analysis scripts regenerate all tables and figures without rerunning the
experiments.

## Minimal use

```python
import numpy as np  # run from inside src/
from core import GroupVerifier, partition_spread, hybrid_scores, conformal_tau, certified_far_randomized

# X_fit: (n, n_APs) strengths (RSSI + 105, zero = not heard); y_fit: zone labels
groups = partition_spread(X_fit, G=10, seed=0)
gv = GroupVerifier(groups, n_zones, K=10, theta=0.6, power=np.e).fit(X_fit, y_fit)
S = hybrid_scores(gv.votes(X_cal).astype(float), c=6)          # (n, n_zones) scores
genuine = S[np.arange(len(y_cal)), y_cal]
tau = conformal_tau(genuine, alpha=0.05, rng=np.random.default_rng(0))
# certified FAR at radius k on impostor scores s_imp (claims of other zones):
# certified_far_randomized(s_imp, tau, k)
```

## Citation

Please cite the paper if you use this code (reference to be added on publication).

## License

MIT, see `LICENSE`.
