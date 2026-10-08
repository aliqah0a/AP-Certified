"""Data loading and task construction for AP-Certified.

UJIIndoorLoc: RSSI in dBm in [-104, 0]; the value 100 means "AP not detected".
We map readings to a non-negative "strength" s = RSSI + 105 (so -104 dBm -> 1,
0 dBm -> 105) and "not detected" -> 0. A zone is (BUILDINGID, FLOOR).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UJI = ROOT / "data/raw/uji/UJIndoorLoc"
N_AP = 520
NOT_DETECTED = 100


def to_strength(rssi: np.ndarray) -> np.ndarray:
    s = rssi.astype(np.float32) + 105.0
    s[rssi == NOT_DETECTED] = 0.0
    return s


def load_uji():
    tr = pd.read_csv(UJI / "trainingData.csv")
    va = pd.read_csv(UJI / "validationData.csv")
    out = {}
    for name, d in (("train", tr), ("test", va)):
        X = to_strength(d.iloc[:, :N_AP].values)
        zone = (d.BUILDINGID.values * 10 + d.FLOOR.values).astype(int)
        out[name] = dict(X=X, zone=zone, phone=d.PHONEID.values,
                         user=d.USERID.values, ts=d.TIMESTAMP.values,
                         building=d.BUILDINGID.values)
    zones = np.unique(out["train"]["zone"])
    zmap = {z: i for i, z in enumerate(zones)}
    for name in out:
        out[name]["y"] = np.array([zmap[z] for z in out[name]["zone"]])
    out["zones"] = zones
    return out


def split_fit_calib(train: dict, calib_frac: float = 0.2, seed: int = 0):
    """Hold out whole phones for threshold calibration (device-disjoint)."""
    rng = np.random.default_rng(seed)
    phones = np.unique(train["phone"])
    n_cal = max(2, int(round(calib_frac * len(phones))))
    cal_ph = rng.choice(phones, n_cal, replace=False)
    m = np.isin(train["phone"], cal_ph)
    sub = lambda mask: {k: v[mask] for k, v in train.items()}
    return sub(~m), sub(m), cal_ph


def verification_pairs(y: np.ndarray, n_zones: int):
    """Every scan claims every zone once.
    Returns (idx, claim, genuine) arrays. Genuine iff claim == true zone."""
    idx = np.repeat(np.arange(len(y)), n_zones)
    claim = np.tile(np.arange(n_zones), len(y))
    return idx, claim, (claim == y[idx])
