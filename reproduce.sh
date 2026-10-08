#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper and the supplement.
# Run from the repository root after placing the datasets as described in data/README.md.
# Total run time on two CPU cores: roughly 8-10 hours (most of it in the attack experiments).
set -euo pipefail
export PYTHONHASHSEED=0
mkdir -p results paper/tables paper/figures

echo "== soundness tests"
python3 tests/test_certificate.py
python3 tests/test_core.py
python3 tests/test_v3.py
python3 tests/test_population.py

echo "== UJIIndoorLoc, building and floor zones"
python3 src/run_uji.py
echo "== UJI long-term dataset, floors and half-floors over 25 months"
python3 src/run_long.py
python3 src/run_delta.py
python3 src/run_sym_long.py
python3 src/run_partition_and_noise.py

echo "== decision rules, multi-scan logins, two-layer verifier, online calibration"
python3 src/run_v3.py Along
python3 src/run_v3.py B
python3 src/run_v3.py C 1
python3 src/run_v3.py C 3

echo "== additional datasets"
python3 src/run_more.py sod_cetc_floor sod_syl_region sod_hcxy_region tampere_floor mall1_floor mall2_floor

echo "== physically grounded attacks (SODIndoorLoc)"
python3 src/run_physics.py CETC331
python3 src/run_physics.py HCXY
python3 src/run_physics.py SYL

echo "== pre-deployment certifiability check, sweep sizes and population radii"
python3 src/run_check.py
python3 src/run_check2.py

echo "== uncertified defense baselines"
python3 src/run_defenses.py

echo "== tables and figures"
python3 src/dataset_table.py
python3 src/analyze_main.py
python3 src/analyze_v3.py
python3 src/analyze_more.py
python3 src/analyze_physics.py
python3 src/analyze_sweep.py
python3 src/analyze_defenses.py
echo "done"
