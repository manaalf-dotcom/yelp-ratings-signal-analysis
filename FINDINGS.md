# Key Findings
## What Does Yelp Actually Reward?

> A machine learning analysis of 27,622 Yelp business records across 11 U.S. metropolitan areas.

---

### Finding 1 — Review volume predicts ratings more than any physical amenity

`review_count` is the single strongest non-geographic predictor of whether a restaurant achieves a high Yelp rating (≥ 3.5 stars), ranking first or second in feature importance across every model tested. Restaurants with more reviews are systematically more likely to be classified as highly rated — independent of their actual amenities.

This points to a **platform reinforcement loop**: high ratings drive search visibility → visibility drives foot traffic → foot traffic drives reviews → reviews reinforce predicted ratings. A restaurant does not need to be better to score higher; it may simply need to be more visible.

---

### Finding 2 — Geographic location explains only a small fraction of the rating signal

**XGBoost, the strongest model, loses only 0.01 AUC points (0.7896 → 0.7790) when latitude and longitude are removed entirely.**

This means platform-level signals — review volume, operating hours, delivery availability — account for 98.7% of the model's ability to distinguish high from low-rated restaurants. Yelp ratings are not simply a proxy for "restaurants in good neighborhoods get good ratings." The engagement signal is real and holds across geographies.

Random Forest shows a larger drop (−0.03 AUC) when geography is removed, but this is partly a modeling artifact: tree ensembles tend to over-rely on continuous variables they can split on repeatedly. XGBoost's near-zero drop is the more reliable signal.

---

### Finding 3 — Physical amenities are largely irrelevant to predicted ratings

WiFi, outdoor seating, TV, and parking consistently rank at the bottom of feature importance across all models. When L1 regularization was applied to logistic regression, many of these amenity coefficients were shrunk to zero entirely — meaning a linear model finds them statistically uninformative once engagement and hours signals are accounted for.

**Investing in amenities to improve Yelp ratings is not supported by this data.** Investing in review accumulation and operating hour coverage is.

---

### Finding 4 — Linear models cannot capture what drives ratings; tree models can

Logistic regression — even with feature engineering and regularization — failed to outperform the majority-class benchmark on accuracy. Random Forest and XGBoost both beat the benchmark meaningfully:

| Model | Stratified Accuracy (5-CV) | ROC-AUC |
|---|---|---|
| Benchmark (majority class) | 0.6872 | 0.50 |
| Logistic Regression | 0.6528 | 0.7218 |
| Random Forest | **0.7503** | 0.7797 |
| XGBoost | 0.7211 | **0.7896** |

The gap between logistic regression and tree models suggests that what drives Yelp ratings is **not a simple additive combination of features** — it is the interaction between them. A restaurant that is open long hours *and* has high review volume *and* offers delivery behaves differently than the sum of those parts would predict.

---

### Platform Design Implications

Three questions these findings raise for any platform using ratings as a quality signal:

**1. Are ratings measuring quality or popularity?**
If review volume drives predicted ratings and ratings drive search visibility, the platform may be amplifying existing popularity rather than surfacing genuinely high-quality restaurants. Popular and high-quality overlap — but they are not the same thing.

**2. Is there a cold-start problem?**
A new restaurant with genuinely excellent food but few reviews will be systematically underranked relative to an older, mediocre restaurant with hundreds of reviews. This creates a structural disadvantage for new and independent restaurants competing against established chains.

**3. Are ratings gameable through engagement tactics alone?**
The data suggests yes. Operational changes that increase review accumulation — follow-up reminders, QR codes, loyalty programs — may improve a restaurant's predicted rating class independent of actual service quality. Platforms should consider whether their rating systems are robust to this kind of optimization.

---

### Limitations

- This analysis is **correlational** — review count predicting ratings does not prove reviews *cause* higher ratings
- **Review text was not analyzed** — NLP on review content could separate sentiment quality from review volume, providing a cleaner test of the quality-vs-engagement question
- The dataset is a **point-in-time snapshot** — a longitudinal study tracking rating evolution as review counts grow would better validate the reinforcement loop hypothesis
- Lat/long are **weak geographic proxies** — richer features like neighborhood foot traffic, competitor density, or median income would give a more interpretable location signal

---

*Full methodology, code, and model evaluation in [`yelp_rating_signals.ipynb`](yelp_rating_signals.ipynb)*
