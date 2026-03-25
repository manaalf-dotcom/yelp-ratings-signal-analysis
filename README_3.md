# What Does Yelp Actually Reward?
### Decoding the Signals Behind Restaurant Ratings

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Yelp%20Open%20Dataset-150K%20records-red?style=flat-square)

---

## The Question

Yelp ratings are treated as a quality signal — by consumers choosing where to eat, by platforms deciding what to surface, and by businesses anxious about their reputation. But what if ratings don't primarily reflect quality?

This project tests a specific hypothesis:

> **Platform-level engagement signals (review volume, operating hours, delivery availability) predict high Yelp ratings as well as — or better than — physical amenities (parking, WiFi, outdoor seating). If true, Yelp ratings may reflect visibility and engagement more than quality.**

The answer has real implications: for how platforms design recommendation systems, for whether ratings are a reliable quality proxy, and for where a business should actually invest to improve its standing.

---

## Key Findings

**1. Review count is the single strongest non-geographic predictor of high ratings.**
Across every model — logistic regression, random forest, and XGBoost — `review_count` consistently ranks as the top or second-highest importance feature. More-reviewed restaurants are more likely to be classified as highly rated. This points to a platform reinforcement loop: high ratings drive visibility, visibility drives reviews, reviews reinforce predicted ratings.

**2. Platform signals retain 98.7% of predictive power without any location data.**
When latitude and longitude are removed, XGBoost — the strongest discriminator — loses only **0.01 AUC points** (0.7896 → 0.7790). Platform-level signals alone explain the vast majority of the rating signal, meaning Yelp ratings are not simply a proxy for "good neighborhood." They reflect genuine platform-level engagement patterns that hold across geographies.

**3. Physical amenities are weak predictors.**
WiFi, parking, outdoor seating, and TV rank near the bottom of feature importance in every model. L1 regularization shrinks many of these coefficients to zero entirely. The data does not support investing in amenities to drive Yelp ratings — engagement-focused operational changes are far better supported.

---

## Results Summary

| Model | Accuracy | ROC-AUC | Stratified Acc (5-CV) |
|---|---|---|---|
| Benchmark (Majority Class) | 0.6872 | 0.50 | 0.6872 |
| Logistic Regression | 0.6463 | 0.7218 | 0.6528 |
| Logistic + L1 Regularization | 0.6464 | 0.7218 | 0.6526 |
| Random Forest | 0.7462 | 0.7797 | 0.7503 |
| XGBoost | 0.7175 | 0.7896 | 0.7211 |
| **Random Forest (No Geo)** | **0.7319** | **0.7476** | **0.7324** |
| **XGBoost (No Geo)** | **0.7048** | **0.7790** | **0.7091** |

> **Primary metric is Stratified Accuracy (5-fold CV)** — preferred over raw accuracy given the ~2:1 class imbalance between high and low-rated businesses.

---

## Analytical Approach

### Problem Framing
Binary classification: predict whether a restaurant achieves **≥ 3.5 stars** (high rating). This threshold is meaningful because it marks the point at which Yelp's algorithm begins prominently surfacing businesses in search results — a platform-defined boundary, not an arbitrary statistical split.

### Data
- **Source:** [Yelp Open Dataset](https://www.yelp.com/dataset) — publicly available for academic use
- **Scale:** 150,246 business records across 11 metropolitan areas, flattened from nested JSON to tabular format
- **Working dataset:** 27,622 records after filtering for completeness on key attributes
- **Class distribution:** ~69% high-rated, ~31% low-rated (2.20:1 ratio — addressed via `class_weight='balanced'` and `scale_pos_weight`)

### Feature Engineering
Raw Yelp data required non-trivial transformation before modeling:

- **Operating hours** — parsed `"HH:MM-HH:MM"` strings into numeric hours-per-day; aggregated into `hours_per_week`
- **Parking** — flattened nested dictionary (`{"garage": True, "street": False, ...}`) to binary `has_parking`
- **City binning** — 1,400+ unique city values reduced to top 10 + "Other" to manage cardinality
- **Boolean conversion** — Yelp's string `"True"`/`"False"` fields converted to numeric binary

### Model Progression

Each model escalation was deliberate, not exhaustive:

| Stage | Model | Reason for escalation |
|---|---|---|
| Baseline | Majority class classifier | True floor — ~67% accuracy |
| Benchmark | Logistic Regression (raw) | Interpretable linear baseline |
| Main | Logistic Regression (engineered features) | Tests linear signal after cleaning |
| Main | + L1 Regularization | Feature selection; isolates linear predictors |
| Main | Random Forest | Captures non-linear interactions; handles mixed types |
| Main | XGBoost | Sequential error correction; strongest on tabular data |
| **New** | **Sensitivity: RF + XGB without lat/long** | **Tests geographic vs. platform signal contribution** |

### Class Imbalance Strategy
Three approaches tested systematically:
- `class_weight='balanced'` — adjusts loss function weights inversely proportional to class frequency
- SMOTE (Synthetic Minority Over-sampling) — applied on training data only to avoid leakage; yielded minimal improvement over `class_weight` and was not carried forward
- `scale_pos_weight` in XGBoost — ratio of negative to positive class samples

---

## Sensitivity Analysis

The centerpiece of this project's analytical contribution. Both Random Forest and XGBoost assign high feature importance to geographic coordinates — but raw importance scores don't tell us whether geography is a confounder or a genuine signal.

**Method:** Rerun both best-performing models with `latitude` and `longitude` removed. Compare Accuracy, ROC-AUC, and Stratified Accuracy across both conditions.

**Result:**

| Model | AUC With Geo | AUC Without Geo | Drop |
|---|---|---|---|
| Random Forest | 0.7797 | 0.7476 | −0.0320 |
| **XGBoost** | **0.7896** | **0.7790** | **−0.0105** |

XGBoost — the model with the highest discriminative power overall — loses only **0.01 AUC points** after geography is removed entirely. That means platform-level signals (review count, operating hours, delivery availability) account for 98.7% of XGBoost's ability to distinguish high from low-rated restaurants. Random Forest drops more (−0.03), but this is partly a modeling artifact — Random Forest as a tree ensemble tends to over-rely on continuous variables like coordinates that it can split on repeatedly across hundreds of trees.

**Why this matters for platform design:** If ratings were primarily a location story, a platform using them for ranking would be amplifying neighborhood effects more than quality. The XGBoost result shows this isn't the case — engagement signals dominate independently of geography, suggesting ratings carry a genuine platform signal, but one driven by visibility mechanics more than objective quality.

---

## Platform Implications

This analysis raises three questions worth taking seriously in product and platform contexts:

**Are ratings measuring quality or engagement?**
The reinforcement loop between review volume, predicted ratings, and search visibility means the platform may be surfacing popular restaurants rather than good ones. These overlap but are not the same thing.

**What happens to new, high-quality restaurants?**
A restaurant with genuinely great food but few reviews will be systematically underranked relative to an established mediocre one. This is a cold-start problem with fairness implications for independent restaurants competing against established chains.

**Can ratings be influenced through engagement tactics alone?**
The data suggests yes. Operational changes that increase review accumulation — reminder campaigns, QR codes at checkout, loyalty prompts — may improve a restaurant's predicted rating class independent of service quality. Platforms should consider whether their rating systems are robust to this.

---

## Limitations

- **Correlation, not causation.** Review count predicting ratings does not prove reviews *cause* higher ratings — both may be driven by a third factor such as restaurant age or chain vs. independent status.
- **No text signals.** Review content is absent from this model. NLP on review text could separate sentiment quality from review volume — a cleaner test of the quality-vs-engagement question.
- **Snapshot data.** The dataset is a point-in-time record. A longitudinal model tracking how ratings evolve as review counts grow would better validate the engagement hypothesis.
- **Geographic proxies.** Lat/long are weak substitutes for meaningful geo features. Replacing them with neighborhood-level signals (foot traffic, competitor density, median income) would give a richer and more interpretable geographic layer.

---

## Repository Structure

```
yelp-rating-signals/
│
├── yelp_rating_signals.ipynb   # Full analysis notebook
├── README.md                    # This file
└── data/
    └── business.csv             # Not included — download from Yelp Open Dataset
                                 # https://www.yelp.com/dataset
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/[your-username]/yelp-rating-signals.git
cd yelp-rating-signals

# 2. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn

# 3. Download the Yelp Open Dataset
# https://www.yelp.com/dataset
# Flatten yelp_academic_dataset_business.json to business.csv
# Place business.csv in the data/ folder

# 4. Run the notebook
jupyter notebook yelp_rating_signals.ipynb
```

---

## About

Built as part of an MSBA machine learning curriculum, extended and reframed for portfolio purposes. The academic version focused on modeling accuracy; this version focuses on what the model reveals about the platform itself.

Dataset: [Yelp Open Dataset](https://www.yelp.com/dataset) — publicly released by Yelp for academic and research use.
