"""Loaders for the additional datasets: SODIndoorLoc, Tampere crowdsourced,
and the two-mall dataset. Every loader returns train/test dicts with
X (strength = RSSI + 105, 0 = not heard), y (zone index), g (position group),
dev (device id), plus zone names.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data/raw"


def _strength_from_dbm(R):
    S = (np.asarray(R, dtype=np.float32) + 105.0)
    S[np.asarray(R) == 100] = 0.0
    return np.clip(S, 0, 105)


def _regions(coords_tr, coords_te, k, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(coords_tr)
    return km.predict(coords_tr), km.predict(coords_te)


# -------------------------------------------------------------------- SOD
SOD_FILES = {"CETC331": ("CETC331/Training_CETC331.csv", "CETC331/Testing_CETC331.csv"),
             "HCXY": ("HCXY/Training_HCXY_All_30.csv", "HCXY/Testing_HCXY_All.csv"),
             "SYL": ("SYL/Training_SYL_All_30.csv", "SYL/Testing_SYL_All.csv")}


def load_sod(building, zones="floor", k_regions=4):
    tr = pd.read_csv(RAW / "sod" / SOD_FILES[building][0])
    te = pd.read_csv(RAW / "sod" / SOD_FILES[building][1])
    macs = [c for c in tr.columns if c.startswith("MAC")]
    out = {}
    if zones == "floor":
        ztr, zte = tr.FloorID.values, te.FloorID.values
        u = {v: i for i, v in enumerate(sorted(set(ztr)))}
        ztr, zte = np.array([u[v] for v in ztr]), np.array([u[v] for v in zte])
    else:
        ztr, zte = _regions(tr[["ECoord", "NCoord"]].values, te[["ECoord", "NCoord"]].values, k_regions)
    for name, d, z in (("train", tr, ztr), ("test", te, zte)):
        g = (d.ECoord.round(2).astype(str) + "_" + d.NCoord.round(2).astype(str) + "_" + d.FloorID.astype(str)).values
        out[name] = dict(X=_strength_from_dbm(d[macs].values), y=z, g=g, dev=d.PhoneID.values,
                         order=d.SampleTimes.values)
    out["Z"] = int(max(ztr.max(), zte.max())) + 1
    return out


# -------------------------------------------------------------------- Tampere
def load_tampere():
    base = RAW / "tampere/FINGERPRINTING_DB"
    out = {}
    floors = None
    for name, pre in (("train", "Training"), ("test", "Test")):
        R = np.loadtxt(base / f"{pre}_rss_21Aug17.csv", delimiter=",")
        C = np.loadtxt(base / f"{pre}_coordinates_21Aug17.csv", delimiter=",")
        D = np.loadtxt(base / f"{pre}_device_21Aug17.csv", delimiter=",", dtype=str)
        if floors is None:
            floors = sorted(set(np.round(C[:, 2], 1)))
        y = np.array([floors.index(round(v, 1)) for v in C[:, 2]])
        g = np.array([f"{a:.2f}_{b:.2f}_{c:.1f}" for a, b, c in C])
        out[name] = dict(X=_strength_from_dbm(R), y=y, g=g, dev=D)
    out["Z"] = len(floors)
    return out


# -------------------------------------------------------------------- Malls
def load_mall(mall, max_test_scans=8000, min_ap_count=20, seed=0):
    base = RAW / "malls/data"
    z = pd.read_csv(base / "zone.csv", sep=";")[["id", "building_id", "floor_id"]].rename(columns={"id": "zone_id"})
    t = pd.read_csv(base / "training_sample.csv", sep=";").merge(z, on="zone_id")
    t = t[t.building_id == mall]
    r = pd.read_csv(base / "route_sample.csv", sep=";").merge(z, on="zone_id")
    r = r[r.building_id == mall]
    cnt = t.groupby("ap").size()
    aps = sorted(cnt[cnt >= min_ap_count].index)            # APs unknown to the server are ignored
    col = {a: i for i, a in enumerate(aps)}
    floors = sorted(t.floor_id.unique())
    fidx = {f: i for i, f in enumerate(floors)}

    def wide(df, key):
        df = df[df.ap.isin(col)]
        keys = df.groupby([key, "timestamp"]).ngroup().values
        n = keys.max() + 1
        X = np.zeros((n, len(aps)), dtype=np.float32)
        X[keys, df.ap.map(col).values] = np.clip(105.0 - df.rssi.values, 1, 105)
        meta = df.assign(k=keys).groupby("k").agg(floor=("floor_id", "first"), grp=(key, "first"),
                                                  zone=("zone_id", "first"))
        return X, np.array([fidx[f] for f in meta.floor.values]), meta.grp.values.astype(str), meta.zone.values

    Xtr, ytr, gtr, ztr = wide(t, "training_id")
    Xte, yte, gte, zte = wide(r, "route_id")
    rng = np.random.default_rng(seed)
    if len(yte) > max_test_scans:
        keep = np.sort(rng.choice(len(yte), max_test_scans, replace=False))
        Xte, yte, gte, zte = Xte[keep], yte[keep], gte[keep], zte[keep]
    # training position groups: shop zone within a calibration session
    gtr = np.array([f"{a}_{b}" for a, b in zip(gtr, ztr)])
    return dict(train=dict(X=Xtr, y=ytr, g=gtr, dev=gtr), test=dict(X=Xte, y=yte, g=gte, dev=gte),
                Z=len(floors), n_aps=len(aps))
