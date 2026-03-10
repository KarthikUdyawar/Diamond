# 💎 Diamond — UX Design Document

**Version:** 1.0  
**Author:** Karthik Udyawar  
**Type:** UX-First Design Document (not pixel-perfect)  
**Companion Doc:** [Diamond PRD v1.0](./Diamond_PRD.md)  
**Status:** Draft  

---

## 1. Design Philosophy

> "A data scientist should never have to open a notebook to answer a business question."

Diamond's UI is built around **three types of users in a single person** — the analyst who wants to explore model behaviour, the engineer who wants to verify API correctness, and the end-user who just wants a price for a specific diamond. The dashboard must serve all three without making any of them feel like second-class citizens.

### Core UX Principles

| Principle                           | What it means in practice                                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Inputs lead, outputs follow**     | Controls are always on the left or top. Results appear immediately to the right or below — never on a separate page |
| **Explain without being asked**     | Every prediction comes with a one-line plain-English reason by default. SHAP is not hidden behind a button          |
| **Numbers need context**            | A predicted price of $3,842 means nothing without a range and a comparison. Always show both                        |
| **Failure is informative**          | If a field is out of range, tell the user why and what the valid range is — not just a red border                   |
| **No loading spinners for < 200ms** | Fast operations should feel instant. Use optimistic UI patterns                                                     |

---

## 2. Users & Goals

### Primary User: Karthik (Developer / Portfolio Viewer)

**Context:** Running the stack locally after cloning the repo. Wants to demonstrate the end-to-end system to a recruiter or use it to evaluate a model upgrade.

**Goals:**
- Quickly verify that predictions look reasonable
- Show a recruiter how SHAP explainability works in practice
- Compare model runs without opening the MLflow UI directly

**Frustrations to avoid:**
- Having to restart a service to swap models
- Waiting more than 2 seconds for any interaction
- A dashboard that looks like a default Streamlit template

### Secondary User: A Curious Visitor (GitHub / Portfolio)

**Context:** Landed on the README, clicked a demo link or screenshot. Has no ML background.

**Goals:**
- Understand what the tool does within 10 seconds
- Try a prediction with default values immediately
- Read the explanation in plain English

**Frustrations to avoid:**
- Walls of technical jargon
- No default values pre-filled
- Explanation charts with no labels

---

## 3. Information Architecture

```
Diamond Dashboard
│
├── Tab 1 — Price Predictor          ← default landing tab
│   ├── Input Panel (left)
│   │   ├── Carat slider
│   │   ├── Cut / Color / Clarity dropdowns
│   │   ├── Depth, Table sliders
│   │   ├── Dimensions (x, y, z) sliders
│   │   └── [Predict] button
│   │
│   └── Output Panel (right)
│       ├── Predicted Price (hero number)
│       ├── Confidence Range bar
│       ├── Plain-English summary sentence
│       └── Top 3 SHAP contributors (inline, small)
│
├── Tab 2 — Explain This Prediction
│   ├── Input Panel (same as Tab 1, synced)
│   ├── SHAP Waterfall Chart (full width)
│   ├── Feature Contribution Table
│   └── Per-feature plain-English sentences
│
└── Tab 3 — Model Dashboard
    ├── Active Model Card (name, version, run ID, R²)
    ├── Model Comparison Table (all MLflow runs)
    ├── Global SHAP Summary Plot
    └── Training Metrics Chart (RMSE across runs)
```

---

## 4. Screen-by-Screen Design

---

### 4.1 Tab 1 — Price Predictor

**Purpose:** The primary workhorse. Give inputs, get a price.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  💎 Diamond Price Predictor          [Tab 1] [Tab 2] [Tab 3]    │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                      │
│   INPUT PANEL            │   OUTPUT PANEL                       │
│                          │                                      │
│   Carat        [━━●━━━]  │   ┌──────────────────────────────┐  │
│   0.20      ▲  1.50  5.0 │   │   Estimated Price            │  │
│                          │   │                              │  │
│   Cut          [Premium▼]│   │   $ 3,842                    │  │
│   Color        [H      ▼]│   │                              │  │
│   Clarity      [SI2    ▼]│   │   Range: $3,540 — $4,145     │  │
│                          │   │   ████████████░░░░           │  │
│   Depth        [━━●━━━]  │   └──────────────────────────────┘  │
│   Table        [━━━●━━]  │                                      │
│                          │   This diamond's high carat weight   │
│   Dimensions             │   and Premium cut place it in the   │
│   x  [━━●━━]             │   upper-mid price range for H-color  │
│   y  [━━●━━]             │   stones with SI2 clarity.           │
│   z  [━━●━━]             │                                      │
│                          │   Top contributors:                  │
│   [  Predict Price  ]    │   ↑ carat     +$1,243               │
│                          │   ↑ clarity   +$310                  │
│                          │   ↓ color     -$122                  │
│                          │                                      │
│                          │   Model: XGBoost v1 · R² 0.984      │
└──────────────────────────┴──────────────────────────────────────┘
```

#### UX Decisions

**Pre-filled defaults:** All inputs load with a realistic "average diamond" on startup (carat: 0.89, cut: Premium, color: H, clarity: SI2). The output panel shows a result immediately — the user never sees an empty state.

**Sliders over number inputs:** Carat, depth, table, x, y, z are sliders. They prevent out-of-range errors by construction and give users a spatial sense of how inputs affect price. The exact value is displayed next to each slider and is editable by clicking it.

**Real-time vs button-triggered:** Dropdowns (cut, color, clarity) update the prediction instantly on change. Sliders update on mouse release, not on drag — prevents spamming the API while the user is mid-slide.

**Confidence range bar:** Displayed as a filled horizontal bar with low and high labels. The fill colour shifts from green (narrow range, high confidence) to amber (wide range, lower confidence) based on the range width relative to the predicted price.

**Plain-English summary:** One sentence, always present, written in plain language. Generated from a template using the top SHAP feature, not from the LLM. Example: *"This diamond's high carat weight and Premium cut place it in the upper-mid price range for H-color stones with SI2 clarity."*

**Top 3 contributors (inline):** Three lines, each with an arrow (↑ positive, ↓ negative), feature name, and dollar contribution. These are SHAP values, always shown without requiring a click to Tab 2. This makes explainability ambient, not opt-in.

**Model metadata footer:** Small grey text at bottom of output panel — model name, version, and R². Gives technical users confidence without distracting non-technical ones.

#### Empty / Error States

| Situation                   | UI Behaviour                                                                |
| --------------------------- | --------------------------------------------------------------------------- |
| API unreachable             | Orange banner: "Prediction service is offline. Run `make up` to start."     |
| Field out of range          | Inline red helper text below the input: "Carat must be between 0.2 and 5.0" |
| Prediction takes > 300ms    | Subtle pulsing animation on the output card, no spinner                     |
| Prediction returns an error | Error card replaces output: "Something went wrong. Check the API logs."     |

---

### 4.2 Tab 2 — Explain This Prediction

**Purpose:** Deep-dive into *why* a diamond got its price. Designed for the ML-curious user and for portfolio demonstrations.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  💎 Diamond Price Predictor          [Tab 1] [Tab 2] [Tab 3]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Explaining prediction for:  0.89ct · Premium · H · SI2         │
│  Predicted Price: $3,842   Base Value: $3,812                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  SHAP Waterfall Chart                      │ │
│  │                                                            │ │
│  │  E[f(x)] = $3,812 ──────────────────────────────┐         │ │
│  │  carat           ████████████████████  +$1,243   │         │ │
│  │  clarity         ████████             +$310       │         │ │
│  │  volume          ██████               +$210       │         │ │
│  │  cut             ████                 +$84        │         │ │
│  │  depth           ▓                   -$18         │         │ │
│  │  color           ▓▓▓▓                -$122        │         │ │
│  │  f(x) = $3,842 ──────────────────────────────────┘         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Feature Breakdown                                               │
│  ┌──────────────┬──────────┬──────────┬──────────────────────┐  │
│  │ Feature      │ Value    │ Impact   │ In Plain English      │  │
│  ├──────────────┼──────────┼──────────┼──────────────────────┤  │
│  │ carat        │ 0.89     │ +$1,243  │ Above average weight  │  │
│  │ clarity      │ SI2      │ +$310    │ Minor inclusions      │  │
│  │ volume       │ 143.8mm³ │ +$210    │ Large physical size   │  │
│  │ cut          │ Premium  │ +$84     │ Near-ideal cut        │  │
│  │ color        │ H        │ -$122    │ Slight yellow tint    │  │
│  │ depth        │ 62.4%    │ -$18     │ Slightly deep         │  │
│  └──────────────┴──────────┴──────────┴──────────────────────┘  │
│                                                                  │
│  [ ← Back to Predictor ]            [ Export Explanation PDF ]  │
└─────────────────────────────────────────────────────────────────┘
```

#### UX Decisions

**Input sync:** The inputs from Tab 1 are shared state. Switching to Tab 2 uses the same values — no need to re-enter anything. A summary line at the top ("Explaining: 0.89ct · Premium · H · SI2") confirms what is being explained.

**Waterfall chart:** Rendered with Plotly for interactivity. Hovering on any bar shows the exact SHAP value, feature value, and a tooltip with a one-line explanation. Positive contributions are shown in a warm teal, negative in a muted coral — not red/green (avoids colour-blind accessibility issues).

**Feature table with plain English:** The fourth column translates the feature value into a human-readable interpretation. These are template-based (e.g., for `color = H` → "Slight yellow tint"), not generated by an LLM. This column is the most important differentiator from a standard SHAP output.

**Base value explanation:** A small info icon (ℹ) next to "Base Value: $3,812" opens a tooltip: *"This is the average predicted price across all diamonds in the training set. Every feature's SHAP value shows how much it pushed the price above or below this baseline."*

**Export button:** "Export Explanation PDF" generates a simple one-page PDF with the waterfall chart, feature table, and prediction details. Useful for portfolio demos and sharing.

---

### 4.3 Tab 3 — Model Dashboard

**Purpose:** Give the developer a view of the ML experiment landscape — which models were trained, how they compare, and which one is currently serving.

#### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  💎 Diamond Price Predictor          [Tab 1] [Tab 2] [Tab 3]    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  🏆 Active Model                                         │   │
│  │  XGBoost Regressor · Version 1 · Run: abc123def          │   │
│  │  R² 0.984   RMSE $498   MAE $297   Trained: 2026-03-09   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  All Experiment Runs                                             │
│  ┌───────────┬──────────┬───────┬──────┬───────┬───────────┐   │
│  │ Model     │ Run ID   │ R²    │ RMSE │ MAE   │ Status    │   │
│  ├───────────┼──────────┼───────┼──────┼───────┼───────────┤   │
│  │ XGBoost   │ abc123   │ 0.984 │ $498 │ $297  │ ✅ Active │   │
│  │ LightGBM  │ def456   │ 0.981 │ $521 │ $318  │ Archived  │   │
│  │ CatBoost  │ ghi789   │ 0.979 │ $548 │ $334  │ Archived  │   │
│  │ GBM Base  │ jkl012   │ 0.962 │ $714 │ $445  │ Archived  │   │
│  └───────────┴──────────┴───────┴──────┴───────┴───────────┘   │
│                                                                  │
│  RMSE Across Runs                  Global Feature Importance    │
│  ┌──────────────────────────┐      ┌─────────────────────────┐  │
│  │  $800 ┤                  │      │ carat      ████████████ │  │
│  │  $700 ┤ ■                │      │ volume     ██████       │  │
│  │  $600 ┤   ■              │      │ clarity    █████        │  │
│  │  $500 ┤     ■  ■         │      │ cut        ███          │  │
│  │       └──────────────    │      │ color      ██           │  │
│  │       GBM Cat  LGB XGB   │      │ table      █            │  │
│  └──────────────────────────┘      └─────────────────────────┘  │
│                                                                  │
│  [ 🔗 Open MLflow UI ]                    Last refreshed: now   │
└─────────────────────────────────────────────────────────────────┘
```

#### UX Decisions

**Active model card:** Always at the top, prominent. Answers the most critical question first: *what is currently serving?* Uses a trophy emoji and a highlighted border to distinguish it clearly from archived runs.

**Run table from MLflow API:** Data is pulled live from `http://localhost:5000/api/2.0/mlflow/runs/search`. A "Last refreshed: now" timestamp with a subtle refresh icon keeps trust high. Auto-refreshes every 60 seconds.

**RMSE bar chart:** Simple bar chart, sorted by RMSE ascending. The active model bar is highlighted in the accent colour. Hovering shows the full run metadata as a tooltip.

**Global SHAP summary:** A horizontal bar chart showing mean absolute SHAP values per feature across the training set — pulled from the MLflow artifact store of the active run. This gives an instant answer to "what drives diamond prices?" without needing to open a notebook.

**MLflow deep link:** A clearly labelled button at the bottom — "Open MLflow UI" — opens `http://localhost:5000` in a new tab. This is the escape hatch for power users who want the full experiment detail.

---

## 5. Component Inventory

| Component             | Used In  | Description                                       |
| --------------------- | -------- | ------------------------------------------------- |
| `PriceBadge`          | Tab 1    | Large price display with currency formatting      |
| `ConfidenceBar`       | Tab 1    | Horizontal range bar with colour-coded confidence |
| `SHAPContributors`    | Tab 1    | 3-line inline SHAP summary (↑ / ↓ arrows)         |
| `DiamondInputPanel`   | Tab 1, 2 | All input controls, shared state                  |
| `PlainEnglishSummary` | Tab 1, 2 | Template-based explanation sentence               |
| `SHAPWaterfall`       | Tab 2    | Plotly waterfall chart                            |
| `FeatureTable`        | Tab 2    | Feature value + SHAP + plain-English table        |
| `ActiveModelCard`     | Tab 3    | Highlighted card for the serving model            |
| `RunComparisonTable`  | Tab 3    | MLflow runs table with metrics                    |
| `RMSEBarChart`        | Tab 3    | Plotly bar chart of RMSE per run                  |
| `GlobalSHAPBar`       | Tab 3    | Mean absolute SHAP horizontal bar chart           |
| `ErrorBanner`         | All      | Inline error messages with recovery hint          |

---

## 6. State Management

The dashboard has three pieces of shared state:

```
AppState
├── inputs: DiamondInputs          # Shared across Tab 1 and Tab 2
│   ├── carat: float
│   ├── cut: str
│   ├── color: str
│   ├── clarity: str
│   ├── depth: float
│   ├── table: float
│   ├── x / y / z: float
│
├── prediction: PredictionResult   # Updated on [Predict] click
│   ├── price: float
│   ├── confidence_low: float
│   ├── confidence_high: float
│   ├── shap_values: dict
│   └── model_version: str
│
└── mlflow_runs: List[RunRecord]   # Fetched on Tab 3 open, refreshed every 60s
```

In Streamlit, this maps to `st.session_state`. The `inputs` dict is initialized with default values at app startup so the user always sees a result on first load.

---

## 7. Accessibility

| Requirement          | Implementation                                                                     |
| -------------------- | ---------------------------------------------------------------------------------- |
| Colour contrast      | All text meets WCAG AA (4.5:1 minimum). SHAP colours use teal/coral, not red/green |
| Keyboard navigation  | All inputs are tab-navigable. Dropdowns support arrow key selection                |
| Screen reader labels | Every slider and dropdown has an `aria-label`. Charts have text alt descriptions   |
| Error messages       | Never rely on colour alone — use icons (⚠️) and text                                |
| Font size            | Minimum 14px for body text. Price hero number is 40px+                             |

---

## 8. Tone & Copy Guidelines

### Voice
Direct, calm, and numerically grounded. Never say "amazing" or "incredible". Treat the user as a data-literate adult.

### Number formatting
- Prices: `$3,842` (not `$3842.00`, not `3842`)
- Percentages: `98.4%` (not `0.984`)
- SHAP values: always show sign (`+$1,243` / `-$122`)
- RMSE/MAE: always show dollar sign

### Explanation sentence templates

```
# High carat
"This diamond's high carat weight ({carat}ct) is the primary driver of its price,
contributing an estimated +${shap_carat} above the baseline."

# Low color grade
"The {color}-color grade introduces a slight yellow tint, reducing the estimated
price by ${abs(shap_color)} compared to a colorless stone."

# Ideal cut
"An Ideal cut maximizes light return, adding +${shap_cut} to the estimated price."

# Generic fallback
"This diamond is estimated at ${price} based on its {carat}ct weight
and {cut} cut quality."
```

---

## 9. Visual Direction (Non-Prescriptive)

This section guides the aesthetic *intent*, not exact pixel values. The implementer has creative freedom within these guardrails.

### Mood
**Refined industrial** — like a gemologist's workbench. Clean, precise, slightly dark. Not a consumer app, not a toy. Data is the hero.

### Colour Palette Guidance

| Role           | Intent                   | Example Direction                               |
| -------------- | ------------------------ | ----------------------------------------------- |
| Background     | Dark neutral             | Deep slate or warm charcoal — not pure black    |
| Surface        | Slightly lighter neutral | Cards and panels lift off the background subtly |
| Accent         | Single cool tone         | A clear teal or electric blue — used sparingly  |
| Positive SHAP  | Warm teal                | Confident, not aggressive                       |
| Negative SHAP  | Muted coral              | Informative, not alarming                       |
| Text primary   | Near-white               | High contrast against dark background           |
| Text secondary | Medium grey              | Metadata, labels, helper text                   |

### Typography Guidance

- **Hero number** (predicted price): Large, monospaced or tabular-figures font — numbers must not shift width as they update
- **Labels and UI chrome**: Clean sans-serif, slightly condensed
- **Plain-English summaries**: Slightly warmer, readable serif or humanist sans — feels like a written explanation, not a UI label

### Spacing Principle
Generous. Let each panel breathe. The input panel and output panel should feel like two distinct work surfaces, not a cramped form.

### What to Avoid
- Default Streamlit theme (grey sidebar, white background, blue buttons)
- Generic purple-gradient-on-white data science aesthetic
- Drop shadows on everything
- Animated loading bars that take longer than the actual operation

---

## 10. Responsive Behaviour

The primary context is a local desktop browser (1280px+). Mobile is not a priority for v1.0 but should not actively break.

| Breakpoint     | Behaviour                                                       |
| -------------- | --------------------------------------------------------------- |
| > 1280px       | Two-column layout: Input left, Output right                     |
| 900px – 1280px | Two-column layout, narrower panels                              |
| < 900px        | Single-column stacked layout. Input above output                |
| < 600px        | Sliders replaced with number inputs. Charts scroll horizontally |

---

## 11. Out of Scope for v1.0 UI

- Dark / light theme toggle
- Multi-diamond batch comparison
- User accounts or saved predictions
- Animated diamond 3D model
- Real-time price streaming

These are noted here so they don't creep into the v1.0 implementation.

---

## 12. Open UX Questions

| #   | Question                                                                               | Impact             | Suggested Default                            |
| --- | -------------------------------------------------------------------------------------- | ------------------ | -------------------------------------------- |
| U1  | Should Tab 2 auto-explain on load or wait for a button click?                          | Tab 2 feel         | Auto-explain using Tab 1's current inputs    |
| U2  | Should the confidence range be from bootstrap sampling or a fixed ±% heuristic for v1? | Output credibility | Fixed ±8% heuristic for v1, swap later       |
| U3  | Should the plain-English sentence live in Tab 1's output or below the input panel?     | Scannability       | Output panel — user reads inputs then result |
| U4  | Should MLflow runs table on Tab 3 be paginated or scrollable?                          | Tab 3 usability    | Scrollable (unlikely to have > 20 runs)      |
| U5  | Should the "Export Explanation PDF" button be in v1 or deferred?                       | Scope              | Defer to v1.1 unless easy with WeasyPrint    |

---

*Design Document v1.0 — Diamond Project — March 2026*  
*UX-first, implementation-agnostic. Pixel-perfect specs to follow in v1.1 if needed.*
