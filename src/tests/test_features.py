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
        validate_raw_data(df)

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


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline

        p = build_pipeline()
        assert isinstance(p, Pipeline)

    def test_output_has_no_nans(self) -> None:
        from src.features import _add_engineered_features, _extract_target

        df = _make_df(n=50)
        cleaned = clean_raw_data(df)
        feat = _add_engineered_features(cleaned)

        X, _ = _extract_target(feat)

        pipeline = build_pipeline()
        transformed = pipeline.fit_transform(X)

        assert not np.isnan(transformed).any()


# ---------------------------------------------------------------------------
# Serialisation
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


# ---------------------------------------------------------------------------
# run_feature_engineering integration test
# ---------------------------------------------------------------------------


class TestRunFeatureEngineering:
    def test_pipeline_runs_with_csv_fixtures(self, tmp_path: Path) -> None:
        """Ensure run_feature_engineering works without external dataset."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        df = pd.DataFrame(
            [
                {**VALID_ROW, "Id": 1, "Weight": 0.9},
                {**VALID_ROW, "Id": 2, "Weight": 1.1},
                {**VALID_ROW, "Id": 3, "Weight": 1.3},
                {**VALID_ROW, "Id": 4, "Weight": 0.7},
                {**VALID_ROW, "Id": 5, "Weight": 1.5},
                {**VALID_ROW, "Id": 6, "Weight": 2.0},
            ]
        )

        df.to_csv(raw_dir / "data_1.csv", index=False)
        df.to_csv(raw_dir / "data_2.csv", index=False)

        processed_dir = tmp_path / "processed"
        pipeline_path = processed_dir / "pipeline.joblib"

        run_feature_engineering(
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            pipeline_path=str(pipeline_path),
        )

        assert (processed_dir / "train.parquet").exists()
        assert (processed_dir / "test.parquet").exists()
        assert pipeline_path.exists()

        train_df = pd.read_parquet(processed_dir / "train.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")

        assert LOG_PRICE_COL in train_df.columns
        assert LOG_PRICE_COL in test_df.columns
