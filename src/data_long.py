"""UJI long-term Wi-Fi dataset (library, floors 3 and 5, 25 months).

Each month m has 'trn' files (reference sweep) and 'tst' files.
RSSI 100 = not heard. Strength s = RSSI + 105, 0 = not heard.
"""
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/raw/long/db"


def _load(files):
    R = np.vstack([np.loadtxt(f, delimiter=",", ndmin=2) for f in files])
    C = np.vstack([np.loadtxt(f.replace("rss", "crd"), delimiter=",", ndmin=2) for f in files])
    S = (R + 105.0).astype(np.float32)
    S[R == 100] = 0.0
    return S, C


def load_month(m, kind):
    files = sorted(glob.glob(str(DB / f"{m:02d}" / f"{kind}*rss.csv")))
    return _load(files) + (files,)


def zones_from_coords(C, scheme="floor"):
    """'floor': 2 zones (floor 3, floor 5).
    'quadrant': floor x 2 halves along the long axis (y) -> 4 zones; the
    halves meet at y = 23 m, so neighbouring zones are metres apart."""
    fl = (C[:, 2] == 5).astype(int)
    if scheme == "floor":
        return fl
    half = (C[:, 1] >= 23.0).astype(int)
    return fl * 2 + half
