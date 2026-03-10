# 💎 Diamond Price Prediction Pipeline Notebook

A complete **data science pipeline** for analyzing diamond characteristics and predicting diamond prices.  
The project covers the full workflow from **data collection → cleaning → exploration → transformation → statistical analysis → model selection → ML pipeline construction**.

The dataset contains **~6,300 diamonds** with attributes such as shape, weight, clarity, colour, cut, polish, symmetry, fluorescence, and physical measurements.

---

# 📊 Project Overview

This project investigates how different **diamond properties affect price** and builds a **machine learning model to predict diamond prices**.

The workflow follows a standard **end-to-end ML pipeline**:

```
Fetch Data
↓
Clean Data
↓
Exploratory Data Analysis
↓
Data Transformation
↓
Hypothesis Testing
↓
Model Selection
↓
ML Pipeline Construction
```

---

# 📂 Project Structure

```
notebooks/
│
├── 01_fetch_data.ipynb
├── 02_clean_data.ipynb
├── 03_eda.ipynb
├── 04_hypothesis_testing.ipynb
├── 05_data_transformation.ipynb
├── 06_model_selection.ipynb
└── 07_pipeline.ipynb

data/
├── raw/
├── processed/
└── transformed/

README.md
```

---

# ⚙️ Pipeline Stages

## 1️⃣ Data Fetching (`01_fetch_data`)

Downloads the **diamond dataset from Kaggle** and merges multiple files into a single dataset.

### Steps
- Download dataset using `opendatasets`
- Load CSV files for different diamond shapes:
  - cushion
  - emerald
  - heart
  - marquise
  - oval
  - pear
  - princess
  - round
- Combine all files into one dataframe
- Inspect structure and missing values
- Save dataset as:

```
diamonds.csv
```

Dataset size:
- **6339 rows**
- **11 columns**

---

# 🧹 Data Cleaning (`02_clean_data`)

Prepares the dataset for analysis.

### Steps

**Handle Missing Values**
- Filled using **mode of each column**

**Remove Duplicates**
- Removed **348 duplicate rows**

**Fix Data Types**
- Converted `Price` from string → float

**Split Measurement Column**

Original column:
```
Measurements
```

Split into:

```
Length
Width
Depth
```

**Memory Optimization**

Converted datatypes:

- `float16`
- `float32`
- `category`

Memory reduced:

```
0.64 MB → 0.20 MB
```

### Output

```
clean_diamonds.csv
clean_diamonds.br
```

---

# 🔍 Exploratory Data Analysis (`03_eda`)

EDA was performed to understand relationships between features and diamond prices.

### Univariate Analysis
Distribution of individual features:

- Price
- Weight
- Depth

### Categorical Analysis

Comparison across categories:

- Shape
- Clarity
- Colour

### Multivariate Analysis

Used visualization techniques such as:

- `pairplot`
- `jointplot`

### Outlier Detection

Detected abnormal values such as:

- Length < 2
- Price > 20,000
- Weight > 3
- Depth < 1

Outliers were removed based on visual inspection.

### Output

```
filtered_diamonds.csv
filtered_diamonds.br
```

---

# 🔄 Data Transformation (`05_data_transformation`)

Transforms data into a format suitable for machine learning.

### Steps

**Drop Irrelevant Columns**

Remove columns that do not contribute to modeling.

**Encode Categorical Variables**

Two encoding techniques were used:

**Ordinal Encoding**

Used for ordered features:

- clarity
- colour
- cut
- polish
- symmetry
- fluorescence

**One-Hot Encoding**

Used for nominal features:

```
Shape_CUSHION
Shape_ROUND
Shape_OVAL
...
```

### Feature Engineering

Final dataset contains:

```
19 features
```

### Correlation Analysis

A **correlation heatmap** was used to study relationships between numerical variables.

### Output

```
transform_diamonds.csv
transform_diamonds.br
```

---

# 📈 Hypothesis Testing (`04_hypothesis_testing`)

Statistical tests were used to determine whether diamond characteristics significantly affect price.

### Methods Used

- **One-Way ANOVA**
- **Tukey HSD (Post-hoc test)**
- **Chi-Square Test**

### Key Findings

**Polish**
```
EX > VG > GD > FR
```
Excellent polish diamonds are most expensive.

**Symmetry**
Also significantly affects price.

```
EX > VG > GD > FR
```

**Fluorescence**
Shows statistically significant impact on price.

**Categorical Relationships**

Most categorical variables are correlated, except:

```
Cut ↔ Fluorescence
```

---

# 🤖 Model Selection (`06_model_selection`)

Multiple machine learning models were evaluated to predict diamond prices.

### Models Tested

- Gradient Boosting
- Random Forest
- CatBoost
- XGBoost
- Ridge Regression
- Lasso Regression
- ElasticNet

### Evaluation Method

- **K-Fold Cross Validation**
- Metrics:
  - MAE (Mean Absolute Error)
  - R² Score

### Feature Importance

Top predictors:

| Feature | Importance |
| ------- | ---------- |
| Weight  | 32.93%     |
| Colour  | 18.78%     |
| Depth   | 12.67%     |

---

# ⚡ ML Pipeline Construction (`07_pipeline`)

The final notebook builds an **automated ML pipeline**.

### Pipeline Components

**Imputer**

Handles missing values:

- Categorical → Most Frequent
- Numerical → KNN Imputer

**Encoder**

- OneHotEncoder → Shape
- Custom Ordinal Converter → Other categorical features

**Feature Selector**

Custom transformer using **CatBoostRegressor**  
Selects **top 14 most important features**.

**Regressor**

Final prediction model:

```
CatBoostRegressor
```

### Hyperparameter Optimization

Used:

```
GridSearchCV
```

to find the best parameters.

---

# 📊 Final Output

The final result is a **trained machine learning pipeline capable of predicting diamond prices based on physical and quality attributes.**

Outputs include:

```
clean dataset
transformed dataset
trained ML pipeline
feature importance analysis
```

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- CatBoost
- XGBoost
- Kaggle API

---

# 🚀 Future Improvements

- Deploy model as an API
- Add real-time diamond pricing interface
- Improve feature engineering
- Add model explainability (SHAP)

---

# 📌 Key Takeaways

- **Weight is the most important factor affecting diamond price**
- Quality characteristics like **colour, polish, and symmetry** significantly impact price
- **CatBoost-based pipeline achieved the best performance**

---

# 📜 License

MIT License
