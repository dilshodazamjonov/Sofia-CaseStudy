# Submission summary

**What I built.** An offline system that takes a proposed branch location (latitude, longitude) and
returns an expected sales index on the supplied 1–100 scale. It also returns the prediction's percentile
among predictions for the existing branches. It includes a Python function, a command-line tool and a
minimal Streamlit page.

**What data I used.** The supplied sales data for 127 branches (1–10 September 2026), Safia's official
store list, stat.uz district statistics and a local OpenStreetMap extract. Eleven wrong branch coordinates
were corrected through a documented override file. Only information available before opening is used.
Orders and average check are excluded.

**How I validated it.** Branch-level cross-validation in two forms: random 5-fold, and geography-aware
(whole areas held out: four Tashkent sectors and all branches outside Tashkent). The geography-aware form
is primary because the task is prediction at new locations. Feature families were compared by ablation,
and Ridge, ExtraTrees and CatBoost were compared on the same folds.

**Final result.** ExtraTrees on 22 features: 9 POI counts within 1 km, 7 Safia-network features and
6 demographic features.
- **Geography-aware CV:** MAE 7.59 against 9.07 for a median baseline (improvement 1.40, 95% CI
  0.40–2.33); Spearman 0.34.
- **Random CV:** MAE 8.31–8.70 against 8.84–8.94 for the baseline.

The signal is real but modest. Predictions are compressed (13–37 against actual values of 1.6–86.5), so
the model is better for screening sites than for spotting exceptional top sellers.

**Main finding.** These are predictive associations, not causal effects:
- Branches with other Safia stores within 1 km tend to sell less.
- Branches in denser commercial and food-retail surroundings (bakeries, supermarkets, shops) tend to sell
  more.
- Network and POI context carry the usable signal only in combination. Demographics and road features add
  little or nothing.

**How to run it.**

```bash
uv sync
uv run pytest -q
uv run python scripts/predict_location.py --lat 41.3111 --lon 69.2797
uv run streamlit run app.py
```

Details are in `README.md` and `reports/final_report.md`.
