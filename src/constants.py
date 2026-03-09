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

# ---------------------------------------------------------------------------
# Ordinal category orders  (must be exact — used by OrdinalEncoder)
# ---------------------------------------------------------------------------

# Cut: Fair (worst) → Ideal (best)
CUT_ORDER: list[str] = ["Fair", "Good", "Very Good", "Premium", "Ideal"]

# Colour: J (worst) → D (best)
COLOR_ORDER: list[str] = ["J", "I", "H", "G", "F", "E", "D"]

# Clarity: I1 (worst) → IF (best)
CLARITY_ORDER: list[str] = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

# Polish: Fair (worst) → Ideal (best)
POLISH_ORDER: list[str] = ["Fair", "Good", "Very Good", "Excellent", "Ideal"]

# Symmetry: Fair (worst) → Ideal (best)
SYMMETRY_ORDER: list[str] = ["Fair", "Good", "Very Good", "Excellent", "Ideal"]

# Fluorescence: Very Strong (most) → None (least)
FLUORESCENCE_ORDER: list[str] = ["Very Strong", "Strong", "Medium", "Faint", "None"]

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
