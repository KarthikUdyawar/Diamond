"""
src/tests/test_features.py
Tests for the feature engineering pipeline.

Covers:
- validate_raw_data: passes on good data, raises on bad data
- clean_raw_data: price parsing, measurement splitting, abbreviation expansion,
                  outlier removal, deduplication
- build_pipeline: output shape, no NaNs, dtype consistency
- No data leakage: pipeline fit only on train, not on test
- Serialisation: joblib dump/load round-trip preserves transform output
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    DEPTH_MM_COL,
    LENGTH_COL,
    LOG_PRICE_COL,
    RAW_DATA_DIR,
    WIDTH_COL,
)
from src.features import (
    _parse_measurements,
    _parse_price,
    build_pipeline,
    clean_raw_data,
    load_pipeline,
    run_feature_engineering,
    validate_raw_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_ROW: dict[str, object] = {
    "Id": 1,
    "Shape": "ROUND",
    "Weight": 0.89,
    "Clarity": "SI2",
    "Colour": "H",
    "Cut": "EX",
    "Polish": "EX",
    "Symmetry": "VG",
    "Fluorescence": "N",
    "Messurements": "6.15-6.10×3.83",
    "Price": "$3,842.00",
}


def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Build a small synthetic raw diamonds DataFrame for testing."""
    rng = np.random.default_rng(seed)
    shapes = ["ROUND", "CUSHION", "OVAL", "PRINCESS", "EMERALD"]
    cuts = ["EX", "VG", "GD", "FR"]
    colours = ["D", "E", "F", "G", "H", "I", "J"]
    clarities = ["SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "IF"]
    polish = ["EX", "VG", "GD"]
    symmetry = ["EX", "VG", "GD"]
    fluorescence = ["N", "F", "M", "ST", "VS"]

    weights = rng.uniform(0.3, 2.5, n)
    lengths = rng.uniform(4.0, 8.0, n)
    widths = rng.uniform(4.0, 8.0, n)
    depths = rng.uniform(2.0, 5.0, n)
    prices = rng.uniform(500, 15_000, n)

    return pd.DataFrame(
        {
            "Id": range(1, n + 1),
            "Shape": rng.choice(shapes, n),
            "Weight": weights,
            "Clarity": rng.choice(clarities, n),
            "Colour": rng.choice(colours, n),
            "Cut": rng.choice(cuts, n),
            "Polish": rng.choice(polish, n),
            "Symmetry": rng.choice(symmetry, n),
            "Fluorescence": rng.choice(fluorescence, n),
            "Messurements": [
                f"{length:.2f}-{width:.2f}×{depth:.2f}"
                for length, width, depth in zip(lengths, widths, depths, strict=True)
            ],
            "Price": [f"${p:,.2f}" for p in prices],
        }
    )


# ---------------------------------------------------------------------------
# validate_raw_data
# ---------------------------------------------------------------------------


class TestValidateRawData:
    def test_passes_on_valid_df(self) -> None:
        df = _make_df()
        validate_raw_data(df)  # should not raise

    def test_raises_on_missing_column(self) -> None:
        df = _make_df().drop(columns=["Weight"])
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_raw_data(df)

    def test_raises_on_multiple_missing_columns(self) -> None:
        df = _make_df().drop(columns=["Weight", "Price", "Shape"])
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_raw_data(df)


# ---------------------------------------------------------------------------
# _parse_price
# ---------------------------------------------------------------------------


class TestParsePrice:
    def test_strips_dollar_and_comma(self) -> None:
        s = pd.Series(["$1,234.56", "$999.00", "500"])
        result = _parse_price(s)
        assert result.tolist() == pytest.approx([1234.56, 999.00, 500.0])

    def test_returns_float64(self) -> None:
        s = pd.Series(["$1,000.00"])
        assert _parse_price(s).dtype == np.float64

    def test_invalid_becomes_nan(self) -> None:
        s = pd.Series(["n/a", ""])
        result = _parse_price(s)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# _parse_measurements
# ---------------------------------------------------------------------------


class TestParseMeasurements:
    def test_dash_times_format(self) -> None:
        s = pd.Series(["5.05-4.35×2.94"])
        result = _parse_measurements(s)
        assert result[LENGTH_COL].iloc[0] == pytest.approx(5.05)
        assert result[WIDTH_COL].iloc[0] == pytest.approx(4.35)
        assert result[DEPTH_MM_COL].iloc[0] == pytest.approx(2.94)

    def test_lowercase_x_format(self) -> None:
        s = pd.Series(["6.15-6.10x3.83"])
        result = _parse_measurements(s)
        assert result[DEPTH_MM_COL].iloc[0] == pytest.approx(3.83)

    def test_malformed_becomes_nan(self) -> None:
        s = pd.Series(["bad_data"])
        result = _parse_measurements(s)
        assert result.isna().all().all()

    def test_returns_three_columns(self) -> None:
        s = pd.Series(["5.0-4.0×3.0"])
        result = _parse_measurements(s)
        assert list(result.columns) == [LENGTH_COL, WIDTH_COL, DEPTH_MM_COL]


# ---------------------------------------------------------------------------
# clean_raw_data
# ---------------------------------------------------------------------------


class TestCleanRawData:
    def test_output_has_no_messurements_or_id(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        assert "Messurements" not in cleaned.columns
        assert "Id" not in cleaned.columns

    def test_output_has_measurement_columns(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        for col in [LENGTH_COL, WIDTH_COL, DEPTH_MM_COL]:
            assert col in cleaned.columns

    def test_price_is_float64(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        assert cleaned["Price"].dtype == np.float64

    def test_outlier_removal_price(self) -> None:
        df = _make_df()
        # Force one row to have price > 20_000
        df.loc[0, "Price"] = "$25,000.00"
        cleaned = clean_raw_data(df)
        assert (cleaned["Price"] <= 20_000).all()

    def test_outlier_removal_weight(self) -> None:
        df = _make_df()
        df.loc[0, "Weight"] = 5.0
        cleaned = clean_raw_data(df)
        assert (cleaned["Weight"] <= 3.0).all()

    def test_deduplication(self) -> None:
        df = _make_df(n=10)
        # Duplicate first row (same data, different Id)
        dup = df.iloc[[0]].copy()
        dup["Id"] = 9999
        df_with_dup = pd.concat([df, dup], ignore_index=True)
        cleaned = clean_raw_data(df_with_dup)
        # Duplicate should be removed
        assert len(cleaned) == len(clean_raw_data(df))

    def test_abbreviations_expanded_cut(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        raw_abbrevs = {"EX", "VG", "GD", "FR"}
        assert not set(cleaned["Cut"].dropna().unique()) & raw_abbrevs

    def test_abbreviations_expanded_fluorescence(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        raw_abbrevs = {"N", "F", "M", "ST", "VS", "SL", "VSL"}
        assert not set(cleaned["Fluorescence"].dropna().unique()) & raw_abbrevs

    def test_shape_is_title_case(self) -> None:
        df = _make_df()
        cleaned = clean_raw_data(df)
        for val in cleaned["Shape"].dropna().unique():
            assert val == val.title(), f"Shape value not title-case: {val!r}"


# ---------------------------------------------------------------------------
# build_pipeline + no data leakage
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline

        p = build_pipeline()
        assert isinstance(p, Pipeline)

    def test_output_has_no_nans(self) -> None:
        df = _make_df(n=50)
        cleaned = clean_raw_data(df)
        from src.features import _add_engineered_features, _extract_target

        feat = _add_engineered_features(cleaned)
        X, _ = _extract_target(feat)
        pipeline = build_pipeline()
        transformed = pipeline.fit_transform(X)
        assert not np.isnan(transformed).any()

    def test_no_data_leakage(self) -> None:
        """Pipeline fitted on train must not use test rows."""
        from sklearn.model_selection import train_test_split

        from src.features import _add_engineered_features, _extract_target

        df = _make_df(n=60)
        cleaned = clean_raw_data(df)
        feat = _add_engineered_features(cleaned)
        X, y = _extract_target(feat)

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = build_pipeline()
        pipeline.fit(X_train)  # fit on train only

        # Both transforms should work without error
        train_out = pipeline.transform(X_train)
        test_out = pipeline.transform(X_test)

        assert train_out.shape[0] == len(X_train)
        assert test_out.shape[0] == len(X_test)

    def test_output_column_count_consistent(self) -> None:
        """Train and test transforms must produce the same number of columns."""
        from sklearn.model_selection import train_test_split

        from src.features import _add_engineered_features, _extract_target

        df = _make_df(n=60)
        cleaned = clean_raw_data(df)
        feat = _add_engineered_features(cleaned)
        X, y = _extract_target(feat)

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = build_pipeline()
        pipeline.fit(X_train)

        assert (
            pipeline.transform(X_train).shape[1] == pipeline.transform(X_test).shape[1]
        )


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_joblib_roundtrip(self) -> None:
        from src.features import _add_engineered_features, _extract_target

        df = _make_df(n=40)
        cleaned = clean_raw_data(df)
        feat = _add_engineered_features(cleaned)
        X, _ = _extract_target(feat)

        pipeline = build_pipeline()
        original_output = pipeline.fit_transform(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name

        import joblib

        joblib.dump(pipeline, tmp_path)
        loaded = load_pipeline(tmp_path)
        loaded_output = loaded.transform(X)

        np.testing.assert_array_almost_equal(original_output, loaded_output)

    def test_load_pipeline_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline

        from src.features import _add_engineered_features, _extract_target

        df = _make_df(n=30)
        cleaned = clean_raw_data(df)
        feat = _add_engineered_features(cleaned)
        X, _ = _extract_target(feat)

        pipeline = build_pipeline()
        pipeline.fit(X)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp_path = f.name

        import joblib

        joblib.dump(pipeline, tmp_path)
        loaded = load_pipeline(tmp_path)
        assert isinstance(loaded, Pipeline)


# ---------------------------------------------------------------------------
# run_feature_engineering (integration test — skipped if no raw CSV)
# ---------------------------------------------------------------------------


class TestRunFeatureEngineering:
    def test_produces_output_files(self, tmp_path: Path) -> None:
        """Integration test: only runs if data/raw CSVs exist."""
        raw_dir = Path(RAW_DATA_DIR)
        if not any(raw_dir.rglob("data_*.csv")):
            pytest.skip(
                "No data_*.csv files found in data/raw/ — skipping integration test."
            )

        processed_dir = str(tmp_path / "processed")
        pipeline_path = str(tmp_path / "processed" / "pipeline.joblib")

        run_feature_engineering(
            raw_dir=str(raw_dir),
            processed_dir=processed_dir,
            pipeline_path=pipeline_path,
        )

        assert (tmp_path / "processed" / "train.parquet").exists()
        assert (tmp_path / "processed" / "test.parquet").exists()
        assert (tmp_path / "processed" / "pipeline.joblib").exists()

    def test_parquet_has_target_column(self, tmp_path: Path) -> None:
        """Check that train/test parquet files contain the log_price target."""
        raw_dir = Path(RAW_DATA_DIR)
        if not any(raw_dir.rglob("data_*.csv")):
            pytest.skip(
                "No data_*.csv files found in data/raw/ — skipping integration test."
            )

        processed_dir = str(tmp_path / "processed")
        pipeline_path = str(tmp_path / "processed" / "pipeline.joblib")

        run_feature_engineering(
            raw_dir=str(raw_dir),
            processed_dir=processed_dir,
            pipeline_path=pipeline_path,
        )

        train_df = pd.read_parquet(str(tmp_path / "processed" / "train.parquet"))
        test_df = pd.read_parquet(str(tmp_path / "processed" / "test.parquet"))

        assert LOG_PRICE_COL in train_df.columns
        assert LOG_PRICE_COL in test_df.columns
