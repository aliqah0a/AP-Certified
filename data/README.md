# Datasets

The datasets are public but not redistributed here. Download them from the original sources and
place them as shown. All loaders read from `data/raw/`.

| Dataset | Source | Expected location |
|---|---|---|
| UJIIndoorLoc | UCI Machine Learning Repository, dataset 310 (https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | `data/raw/uji/UJIndoorLoc/trainingData.csv`, `validationData.csv` |
| UJI long-term Wi-Fi fingerprinting | Zenodo, record 3748719 (Mendoza-Silva et al., Data 2018) | `data/raw/long/db/01/ ... 25/` (`trn*rss.csv`, `tst*rss.csv`, coordinate files) |
| SODIndoorLoc | GitHub repository of Bi et al., Satellite Navigation 2022 (https://github.com/renwudao24/SODIndoorLoc) | `data/raw/sod/CETC331/`, `HCXY/`, `SYL/`, and `data/raw/sod/ap_info.xlsx` |
| Tampere crowdsourced Wi-Fi | Zenodo, doi:10.5281/zenodo.889798 (Lohan et al., Data 2017) | `data/raw/tampere/FINGERPRINTING_DB/` |
| Two shopping malls | Zenodo, doi:10.5281/zenodo.3698238 (Lopez-Pastor et al., 2020) | `data/raw/malls/data/` (`training.csv`, `route.csv`, `zone.csv`, `floor.csv`) |

File names inside each folder follow the original archives; see `src/data.py`, `src/data_long.py`
and `src/data_more.py` for the exact paths.
