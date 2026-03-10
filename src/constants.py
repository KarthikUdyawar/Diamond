"""
Diamond project — shared constants.

All column names, category orders, and magic numbers live here.
Nothing in src/ or api/ should contain raw strings for these values.
"""

# ---------------------------------------------------------------------------
# Raw Kaggle CSV column names
# Dataset: harshitlakhani/natural-diamonds-prices-images
# ---------------------------------------------------------------------------
RAW_COLUMNS: list[str] = [
    "Id",
    "Shape",
    "Weight",  # carats — float64
    "Clarity",
    "Colour",
    "Cut",
    "Polish",
    "Symmetry",
    "Fluorescence",
    "Messurements",  # intentional typo in source data
    "Price",  # object dtype in CSV — contains currency symbols
]

# Parsed measurement column names (split from Messurements)
LENGTH_COL: str = "length"
WIDTH_COL: str = "width"
DEPTH_MM_COL: str = "depth_mm"

# Target column (raw and log-transformed)
PRICE_COL: str = "Price"
LOG_PRICE_COL: str = "log_price"

# Engineered feature names
VOLUME_COL: str = "volume"
CARAT_PER_VOLUME_COL: str = "carat_per_volume"
LOG_WEIGHT_COL: str = "log_weight"

# Small constant to avoid division by zero in carat_per_volume
VOLUME_EPSILON: float = 1e-6

# ---------------------------------------------------------------------------
# Abbreviation expansion maps  (raw CSV → full English label)
# Applied during cleaning BEFORE ordinal encoding
# ---------------------------------------------------------------------------

CUT_ABBREV: dict[str, str] = {
    "FR": "Fair",
    "GD": "Good",
    "VG": "Very Good",
    "EX": "Excellent",
}

POLISH_ABBREV: dict[str, str] = {
    "FR": "Fair",
    "GD": "Good",
    "VG": "Very Good",
    "EX": "Excellent",
}

SYMMETRY_ABBREV: dict[str, str] = {
    "FR": "Fair",
    "GD": "Good",
    "VG": "Very Good",
    "EX": "Excellent",
}

FLUORESCENCE_ABBREV: dict[str, str] = {
    "VS": "Very Strong",
    "ST": "Strong",
    "SL": "Slight",
    "M": "Medium",
    "F": "Faint",
    "VSL": "Very Slight",
    "N": "None",
}

# Shape values are all-caps in CSV — title-case them to match SHAPE_CATEGORIES
SHAPE_TITLE_CASE: bool = True  # "CUSHION" → "Cushion"

# ---------------------------------------------------------------------------
# Ordinal category orders  (must be exact — used by OrdinalEncoder)
# Ordered from worst/least → best/most
# ---------------------------------------------------------------------------

# Cut: Fair (worst) → Excellent (best)
# NOTE: CSV contains FR/GD/VG/EX — no Premium or Ideal in this dataset
CUT_ORDER: list[str] = ["Fair", "Good", "Very Good", "Excellent"]

# Colour: FANCY (non-standard) → D (best/most colourless)
# Extended to cover all grades present in the Kaggle dataset
COLOR_ORDER: list[str] = [
    "FANCY",
    "Y-Z",
    "W-X",
    "W",
    "U-V",
    "S-T",
    "Q-R",
    "O-P",
    "O",
    "N",
    "M",
    "L",
    "K",
    "J",
    "I",
    "H",
    "G",
    "F",
    "E",
    "D",
]

# Clarity: I3 (worst/most included) → FL (best/flawless)
# Extended to cover I2, I3, FL found in the Kaggle dataset
CLARITY_ORDER: list[str] = [
    "I3",
    "I2",
    "I1",
    "SI2",
    "SI1",
    "VS2",
    "VS1",
    "VVS2",
    "VVS1",
    "IF",
    "FL",
]

# Polish: Fair (worst) → Excellent (best)
# NOTE: dataset has no "Ideal" — matches CUT_ORDER pattern
POLISH_ORDER: list[str] = ["Fair", "Good", "Very Good", "Excellent"]

# Symmetry: Fair (worst) → Excellent (best)
SYMMETRY_ORDER: list[str] = ["Fair", "Good", "Very Good", "Excellent"]

# Fluorescence: Very Strong (most) → None (least)
# Extended to cover Slight and Very Slight found in the Kaggle dataset
FLUORESCENCE_ORDER: list[str] = [
    "Very Strong",
    "Strong",
    "Slight",
    "Medium",
    "Faint",
    "Very Slight",
    "None",
]

# Shape is nominal (no natural order) — one-hot encoded
SHAPE_CATEGORIES: list[str] = [
    "Cushion",
    "Emerald",
    "Heart",
    "Marquise",
    "Oval",
    "Pear",
    "Princess",
    "Round",
]

# ---------------------------------------------------------------------------
# EDA outlier thresholds  (from notebook 03_eda)
# ---------------------------------------------------------------------------
OUTLIER_MIN_LENGTH: float = 2.0
OUTLIER_MAX_PRICE: float = 20_000.0
OUTLIER_MAX_WEIGHT: float = 3.0
OUTLIER_MIN_DEPTH_MM: float = 1.0

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_MODEL_NAME: str = "Diamond"
MLFLOW_MODEL_STAGE: str = "Production"

# ---------------------------------------------------------------------------
# Confidence range  (PRD Q1 resolved: ±8% heuristic)
# ---------------------------------------------------------------------------
CONFIDENCE_MARGIN: float = 0.08

# ---------------------------------------------------------------------------
# Paths  (relative to project root)
# ---------------------------------------------------------------------------
RAW_DATA_DIR: str = "data/raw"
PROCESSED_DIR: str = "data/processed"
TRAIN_PARQUET_PATH: str = "data/processed/train.parquet"
TEST_PARQUET_PATH: str = "data/processed/test.parquet"
PIPELINE_PATH: str = "data/processed/pipeline.joblib"
