"""
src/features.py
Feature engineering pipeline for the Diamond price prediction project.

Public API:
    validate_raw_data(df)        — raises ValueError on bad schema / dtypes
    clean_raw_data(df)           — parse, expand abbreviations, remove outliers
    build_pipeline()             — returns unfitted sklearn Pipeline
    run_feature_engineering()    — orchestrator: read → clean → fit → save
    load_pipeline(path)          — load a saved joblib pipeline
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.constants import (
    CARAT_PER_VOLUME_COL,
    CLARITY_ORDER,
    COLOR_ORDER,
    CUT_ABBREV,
    CUT_ORDER,
    DEPTH_MM_COL,
    FLUORESCENCE_ABBREV,
    FLUORESCENCE_ORDER,
    LENGTH_COL,
    LOG_PRICE_COL,
    LOG_WEIGHT_COL,
    MIN_SPLIT_SAMPLES,
    OUTLIER_MAX_PRICE,
    OUTLIER_MAX_WEIGHT,
    OUTLIER_MIN_DEPTH_MM,
    OUTLIER_MIN_LENGTH,
    PIPELINE_PATH,
    POLISH_ABBREV,
    POLISH_ORDER,
    PRICE_COL,
    PROCESSED_DIR,
    RANDOM_STATE,
    RAW_COLUMNS,
    RAW_DATA_DIR,
    SHAPE_CATEGORIES,
    SYMMETRY_ABBREV,
    SYMMETRY_ORDER,
    TEST_SIZE,
    VOLUME_COL,
    VOLUME_EPSILON,
    WIDTH_COL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groups used by the ColumnTransformer
# ---------------------------------------------------------------------------
ORDINAL_COLS: list[str] = [
    "Cut",
    "Colour",
    "Clarity",
    "Polish",
    "Symmetry",
    "Fluorescence",
]
NOMINAL_COLS: list[str] = ["Shape"]
NUMERIC_COLS: list[str] = ["Weight", LENGTH_COL, WIDTH_COL, DEPTH_MM_COL]
ENGINEERED_NUMERIC_COLS: list[str] = [VOLUME_COL, CARAT_PER_VOLUME_COL, LOG_WEIGHT_COL]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Validate that *df* has the expected columns from the raw Kaggle CSV.

    Raises
    ------
    ValueError
        If any expected column is missing.
    """
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    logger.info(
        "Raw data validation passed — all %d columns present.", len(RAW_COLUMNS)
    )


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------


def _parse_price(series: pd.Series) -> pd.Series:
    """Strip currency symbols / commas and cast to float64."""
    cleaned = series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def _parse_measurements(series: pd.Series) -> pd.DataFrame:
    """
    Parse the 'Messurements' column into three float columns.

    Supported formats
    -----------------
    - "5.05-4.35×2.94"   (dash + unicode ×)
    - "5.05-4.35x2.94"   (dash + lowercase x)
    - "5.05 x 4.35 x 2.94"  (spaces + x)
    """

    def _split(val: object) -> tuple[float, float, float]:
        s = str(val).strip()
        # Replace all separators with a single pipe for uniform splitting
        s = re.sub(r"[×xX]", "|", s)
        s = re.sub(r"-", "|", s)
        s = re.sub(r"\s+", "", s)
        parts = s.split("|")
        if len(parts) != 3:  # noqa: PLR2004
            return float("nan"), float("nan"), float("nan")
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            return float("nan"), float("nan"), float("nan")

    parsed = series.apply(_split)
    result = pd.DataFrame(
        parsed.tolist(),
        columns=[LENGTH_COL, WIDTH_COL, DEPTH_MM_COL],
        index=series.index,
    )
    return result.astype("float64")


def _expand_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """Expand abbreviated categorical values to their full English labels."""
    df = df.copy()
    df["Cut"] = df["Cut"].map(CUT_ABBREV).fillna(df["Cut"])
    df["Polish"] = df["Polish"].map(POLISH_ABBREV).fillna(df["Polish"])
    df["Symmetry"] = df["Symmetry"].map(SYMMETRY_ABBREV).fillna(df["Symmetry"])
    df["Fluorescence"] = (
        df["Fluorescence"].map(FLUORESCENCE_ABBREV).fillna(df["Fluorescence"])
    )
    # Shape: all-caps in CSV → Title Case to match SHAPE_CATEGORIES
    df["Shape"] = df["Shape"].str.title()
    return df


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw diamonds DataFrame.

    Steps
    -----
    1. Expand categorical abbreviations to full English labels.
    2. Parse Price → float64.
    3. Parse Messurements → length, width, depth_mm.
    4. Remove outliers (from notebook 03_eda).
    5. Remove duplicate rows (excluding Id column).

    Parameters
    ----------
    df:
        Raw DataFrame as loaded from the Kaggle CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with Id and Messurements dropped.
    """
    df = df.copy()
    rows_before = len(df)

    # 1. Expand abbreviations
    df = _expand_abbreviations(df)

    # Validate shapes against allowed categories
    invalid_shapes = set(df["Shape"].dropna()) - set(SHAPE_CATEGORIES)
    if invalid_shapes:
        logger.error("Unknown shapes detected: %s", invalid_shapes)
        raise ValueError(f"Unknown diamond shapes found: {invalid_shapes}")

    # 2. Parse Price
    df[PRICE_COL] = _parse_price(df[PRICE_COL])
    df["Weight"] = pd.to_numeric(df["Weight"], errors="raise").astype("float64")

    # 3. Parse Messurements → 3 float columns
    measurements = _parse_measurements(df["Messurements"])
    df = pd.concat([df.drop(columns=["Messurements"]), measurements], axis=1)

    # 4. Remove outliers
    mask = (
        (df[PRICE_COL] >= 0)
        & (df[PRICE_COL] <= OUTLIER_MAX_PRICE)
        & (df["Weight"] > 0)
        & (df["Weight"] <= OUTLIER_MAX_WEIGHT)
        & (df[LENGTH_COL] >= OUTLIER_MIN_LENGTH)
        & (df[WIDTH_COL] > 0)
        & (df[DEPTH_MM_COL] >= OUTLIER_MIN_DEPTH_MM)
    )
    df = df[mask].copy()
    logger.info("Outlier removal: %d → %d rows.", rows_before, len(df))

    # 5. Remove duplicates (exclude Id — it's a row identifier)
    cols_for_dedup = [c for c in df.columns if c != "Id"]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=cols_for_dedup).copy()
    logger.info("Deduplication: %d → %d rows.", before_dedup, len(df))

    # Drop Id — not a feature
    df = df.drop(columns=["Id"], errors="ignore")

    logger.info("Cleaning complete. Final shape: %s.", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature engineering (engineered columns added before pipeline)
# ---------------------------------------------------------------------------


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume, carat_per_volume, and log_weight columns."""
    df = df.copy()
    df[VOLUME_COL] = df[LENGTH_COL] * df[WIDTH_COL] * df[DEPTH_MM_COL]
    df[CARAT_PER_VOLUME_COL] = df["Weight"] / (df[VOLUME_COL] + VOLUME_EPSILON)
    df[LOG_WEIGHT_COL] = np.log1p(df["Weight"])
    return df


def _extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target.

    Returns
    -------
    X: pd.DataFrame
        Feature matrix (Price column removed).
    y: pd.Series
        log1p-transformed price vector.
    """
    y = pd.Series(np.log1p(df[PRICE_COL]), index=df.index).rename(LOG_PRICE_COL)
    X = df.drop(columns=[PRICE_COL])
    return X, y


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """
    Build and return an *unfitted* sklearn Pipeline.

    Architecture
    ------------
    ColumnTransformer
    ├── ordinal  → SimpleImputer(most_frequent) + OrdinalEncoder
    ├── nominal  → SimpleImputer(most_frequent) + OneHotEncoder
    └── numeric  → KNNImputer(n_neighbors=5)
    """
    ordinal_orders = [
        CUT_ORDER,
        COLOR_ORDER,
        CLARITY_ORDER,
        POLISH_ORDER,
        SYMMETRY_ORDER,
        FLUORESCENCE_ORDER,
    ]

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_orders,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[SHAPE_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
        ]
    )

    all_numeric = NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_transformer, ORDINAL_COLS),
            ("nominal", nominal_transformer, NOMINAL_COLS),
            ("numeric", numeric_transformer, all_numeric),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipeline


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def load_pipeline(path: str = PIPELINE_PATH) -> Pipeline:
    """Load and return a fitted pipeline from *path*."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _merge_raw_csvs(raw_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Discover and merge all per-shape CSV files from *raw_dir*.

    Handles two subdirectory layouts found in the Kaggle download:
    - ``Diamonds/Diamonds/data_*.csv``  — has an extra ``Data Url`` column
    - ``Diamonds2/data_*.csv``          — standard 11-column schema

    Returns
    -------
    pd.DataFrame
        Concatenated raw DataFrame with only the standard RAW_COLUMNS kept.
    """
    csv_files = list(Path(raw_dir).rglob("data_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No data_*.csv files found under {raw_dir!r}. "
            "Run 'make download' to fetch the raw dataset."
        )

    frames: list[pd.DataFrame] = []
    for path in sorted(csv_files):
        df = pd.read_csv(path)
        # Drop the extra 'Data Url' column present in the Diamonds/ subset
        df = df.drop(columns=["Data Url"], errors="ignore")
        missing = [col for col in RAW_COLUMNS if col not in df.columns]
        if missing:
            logger.error(f"{path} is missing expected columns: {missing}")
            raise ValueError(f"{path} is missing expected columns: {missing}")
        frames.append(df[RAW_COLUMNS].copy())

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "Merged %d CSV files → %d rows, %d columns.",
        len(csv_files),
        len(merged),
        len(merged.columns),
    )
    return merged


def run_feature_engineering(
    raw_dir: str = RAW_DATA_DIR,
    processed_dir: str = PROCESSED_DIR,
    pipeline_path: str | None = None,
) -> None:
    """
    Full feature engineering orchestrator.

    1. Discover and merge all per-shape CSVs from *raw_dir*.
    2. Validate + clean.
    3. Engineer features.
    4. Train/test split.
    5. Fit pipeline on train only.
    6. Save train.parquet, test.parquet, pipeline.joblib to *processed_dir*.

    Parameters
    ----------
    raw_dir:
        Directory containing the Kaggle download subfolders.
    processed_dir:
        Directory for output artefacts.
    pipeline_path:
        Full path for the saved pipeline joblib file.
    """
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    if pipeline_path is None:
        pipeline_path = os.fspath(Path(processed_dir) / Path(PIPELINE_PATH).name)
    Path(pipeline_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Merge raw CSVs
    df_raw = _merge_raw_csvs(raw_dir)

    # 2. Validate + clean
    validate_raw_data(df_raw)
    df_clean = clean_raw_data(df_raw)

    # 3. Engineer features
    df_feat = _add_engineered_features(df_clean)
    X, y = _extract_target(df_feat)

    if len(X) < MIN_SPLIT_SAMPLES:
        logger.error("Dataset too small for train/test split: %s", len(X))
        raise ValueError("Dataset too small for train/test split")

    # 4. Split — stratification not needed for regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Split: %d train rows, %d test rows.", len(X_train), len(X_test))

    # 5. Fit pipeline on train only (no leakage)
    pipeline = build_pipeline()
    pipeline.fit(X_train)
    logger.info("Pipeline fitted on training data.")

    # 6. Save parquet files (include target)
    train_df = X_train.copy()
    train_df[LOG_PRICE_COL] = y_train
    test_df = X_test.copy()
    test_df[LOG_PRICE_COL] = y_test

    train_out = os.path.join(processed_dir, "train.parquet")
    test_out = os.path.join(processed_dir, "test.parquet")
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)
    logger.info("Saved train → %s", train_out)
    logger.info("Saved test  → %s", test_out)

    # 7. Save pipeline
    joblib.dump(pipeline, pipeline_path)
    logger.info("Saved pipeline → %s", pipeline_path)

    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_feature_engineering()
