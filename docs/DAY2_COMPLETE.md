# ✅ Day 2 Complete — Feature Engineering Pipeline

**Date:** 2026-03-10
**Branch:** `feature/day2-feature-pipeline`
**PR:** → `develop`
**Tag:** `day2-complete`

---

## What Was Built

A full feature engineering pipeline that merges 15 per-shape raw CSVs from two Kaggle subdirectories into a single dataset, cleans and validates it, engineers three derived features, and outputs `train.parquet`, `test.parquet`, and `pipeline.joblib` to `data/processed/`. The sklearn `ColumnTransformer` applies ordinal encoding to 6 quality columns, one-hot encoding to `Shape`, and KNN imputation to all numerical columns — fitted strictly on training data to prevent leakage. An abbreviation expansion step normalises all abbreviated categorical values (e.g. `EX → Excellent`, `N → None`) before encoding.

---

## Files Changed

| File                         | Action   | Notes                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/constants.py`           | Modified | Updated `CUT_ORDER`, `COLOR_ORDER`, `CLARITY_ORDER`, `POLISH_ORDER`, `SYMMETRY_ORDER`, `FLUORESCENCE_ORDER` to match real dataset; added abbreviation maps (`CUT_ABBREV`, `POLISH_ABBREV`, `SYMMETRY_ABBREV`, `FLUORESCENCE_ABBREV`); added `LENGTH_COL`, `WIDTH_COL`, `DEPTH_MM_COL`, `LOG_PRICE_COL`, engineered feature name constants, path constants; replaced `RAW_DATA_PATH` with `RAW_DATA_DIR` |
| `src/features.py`            | Created  | `validate_raw_data`, `clean_raw_data`, `_merge_raw_csvs`, `build_pipeline`, `load_pipeline`, `run_feature_engineering` with full type hints                                                                                                                                                                                                                                                             |
| `src/tests/__init__.py`      | Created  | Empty — required for pytest discovery                                                                                                                                                                                                                                                                                                                                                                   |
| `src/tests/test_features.py` | Created  | 18 tests across 6 test classes; 81% coverage; integration tests auto-skip if raw CSVs absent                                                                                                                                                                                                                                                                                                            |

---

## Done Criteria

- [x] `make features` runs without error and produces Parquet files in `data/processed/`
- [x] Pipeline serialises and deserialises without data loss
- [x] `make lint` still passes
- [x] `make test` passes with ≥ 80% coverage (81.29%)

---

## Decisions Made

| Decision                         | Choice                                                          | Reason                                                                                                        |
| -------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Raw data input                   | Merge 15 per-shape CSVs at runtime via `_merge_raw_csvs()`      | `data/raw/diamonds.csv` does not exist — raw data is split across `Diamonds/` and `Diamonds2/` subdirectories |
| `Data Url` column                | Dropped silently via `errors="ignore"`                          | Present only in `Diamonds/` subset; not a feature                                                             |
| `CUT_ORDER`                      | `["Fair", "Good", "Very Good", "Excellent"]` — no Premium/Ideal | Dataset contains only `FR/GD/VG/EX`; original PRD order does not match the actual data                        |
| `COLOR_ORDER`                    | Extended to 20 grades including FANCY, K–Z, and range grades    | Full distribution found in data; FANCY kept at worst end rather than dropped                                  |
| `CLARITY_ORDER`                  | Extended with `I3`, `I2`, `FL`                                  | All three grades present in dataset; `FL` (Flawless) sits above `IF` per industry standard                    |
| `FLUORESCENCE_ORDER`             | Extended with `Slight`, `Very Slight` for `SL`/`VSL`            | Two additional abbreviations found in data not covered by original order                                      |
| Measurement separator            | Parse both `-` and `×`/`x`                                      | CSV uses `"5.05-4.35×2.94"` format, not the `"L x W x D"` format documented in the PRD                        |
| `RAW_DATA_PATH` → `RAW_DATA_DIR` | Directory-based input                                           | Reflects actual data layout; `_merge_raw_csvs` uses `Path.rglob("data_*.csv")`                                |

---

## Deferred / Known Issues

- `Diamonds/` contains a `data_radiant.csv` shape not present in `Diamonds2/` and not in `SHAPE_CATEGORIES`. Radiant diamonds will be one-hot encoded as unknown and silently zeroed out by `handle_unknown="ignore"`. Deferred to v1.1 — requires a PRD decision on whether to add `Radiant` as a supported shape.
- Price column in `Diamonds2/` CSVs has no currency symbol (plain float string), while `Diamonds/` has none either — `_parse_price` handles both via `pd.to_numeric` with `errors="coerce"`.
- Integration tests in `TestRunFeatureEngineering` auto-skip in CI since `data/raw/` is gitignored. Full end-to-end coverage requires a local run with the Kaggle download present.

---

## Next Day Preview

Day 3 trains CatBoost, XGBoost, LightGBM, and GradientBoosting against the processed parquet files, logs all runs to MLflow, runs 50 Optuna trials on CatBoost, and registers the best model as `Diamond/Production`.
