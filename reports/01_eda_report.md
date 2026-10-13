# Data understanding report — Safia branch sales, 1–10 September 2026

Source notebook: `notebooks/01_eda.ipynb`. Data: `data/processed/branches.csv`, `daily_performance.csv`, `dataset_metadata.csv`.

> **Early-stage document (1 October 2026).** This report was written before the coordinate QA and the
> modelling, and uses the supplied coordinates and city labels. Two points were settled later, in
> [final_report.md](final_report.md):
> - **Tashkent count.** The coordinate QA moved branch W074 to Fergana, so 101 branches (not 102) are in
>   Tashkent.
> - **Store format.** Format was not modelled, because two Safia Mini branches are too few to estimate a
>   format effect.

**Purpose.** Decide whether `avg_sales_index` (the 10-day mean of daily `sales_index`) is a sound target for a location-quality model, and document what it represents.

**Decision.** Use `avg_sales_index` as provided. Daily performance is stable, mean and median agree, and branch rankings persist from day to day. Caveats: only 10 days, a skewed 1.6–86.5 scale, and a handful of noisy branches (details below).

---

## 1. Data

| item | value |
|---|---|
| branches | 127 (one row each), all with sales on all 10 days |
| branch-days | 1 270 (127 × 10), 1–10 Sep 2026, Tue → Thu, one weekend, 1 Sep is a public holiday |
| cities | 20; Tashkent holds 102 branches, the other 19 cities hold 25 |
| formats | 125 `Safia`, 2 `Safia mini` |
| excluded at source | 14 branches with partial days, 17 with no sales, 68 without location, 14 with a wrong location, 38 non-stores, 1 not in iiko |

Index definitions (from the metadata): `sales_index` = daily sales after discounts ÷ highest single-day sales of any branch × 100. `orders_index` on the same principle. `avg_check_index` = branch sales ÷ orders over the 10 days, best branch = 100. Real amounts are not included.

**Integrity.** All checks pass: no missing values, no duplicates, exactly 10 days per branch, same 127 branches in both files, indices within [0, 100]. The provided `avg_sales_index` and `avg_orders_index` equal the recomputed means of the daily file to machine precision, and `avg_check_index` equals (sales ÷ orders) rescaled to max 100 within rounding.

## 2. Target distribution

| min | p10 | p25 | median | mean | p75 | p90 | max | std | skewness |
|---|---|---|---|---|---|---|---|---|---|
| 1.61 | 8.96 | 12.16 | 17.84 | 20.36 | 25.98 | 35.52 | 86.51 | 12.52 | 1.86 |

| band | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–80 | 80–90 |
|---|---|---|---|---|---|---|---|---|
| branches | 18 | 58 | 29 | 14 | 5 | 2 | 0 | 1 |

- Strongly right-skewed. Half of all branches sit between 12 and 26.
- Only 8 branches exceed 40. **Мега (W035) at 86.5 is a lone outlier**, 27 points above Зенит (59.2) and МедГор (58.0).
- Seven branches are below 5 (lowest: АНД Экопарк 1.6, Заркайнар 1.7).
- Consequence: judge model error in relative as well as absolute terms, and keep a log or rank view of the target.

**Is 100 attainable?** No. The scale is anchored on the best *single day* of any branch. 100 occurs exactly once in the daily file (Мега, Saturday 5 Sep). The best branch average is 86.5; the best single day of any other branch is 68.9. Only 2.9 % of branch-days exceed 50 and 18 % exceed 30. Training targets span **1.6–86.5**, not 1–100.

## 3. Target stability

Per-branch statistics over the 10 days (`std`, `range = max − min`, `CV = std ÷ mean`):

| statistic | median branch | 75th pct | max |
|---|---|---|---|
| std of daily sales_index | 1.96 | 3.01 | 7.15 (ТТЗ) |
| range (max − min) | 6.0 | 9.5 | 23.4 (Матонат) |
| CV | 11.3 % | 15.4 % | 33.9 % (Бочка) |

- Day-to-day ranges are narrow compared with the gaps between branches (inter-quartile spread of the means is 14 points).
- Volatility grows with level in absolute terms, but relative noise is modest.

**Mean vs median.** r = 0.998. The typical gap is +1 % (means sit slightly above medians because the unusual days are mostly *high* days). 21 branches differ by more than 5 %, only 2 by more than 10 %: Бочка (+16 %) and Жомий (−11 %). Mean and median differ by less than 1 point for 108 of 127 branches, so the provided mean is a defensible target; the median is kept as a robustness check.

**Ten most volatile branches (by CV)** and the pattern seen in their daily series:

| branch | city | mean | median | std | CV | pattern |
|---|---|---|---|---|---|---|
| Бочка | near Gazalkent (16 km) | 15.7 | 13.4 | 5.3 | 34 % | leisure: peaks on 1 Sep (holiday), Sat 5, Sun 6 |
| Жомий | Tashkent | 11.5 | 12.9 | 3.8 | 33 % | collapse on 9–10 Sep (7.6 → 2.1): probable closure or partial day |
| Матонат | Tashkent | 21.6 | 20.0 | 6.8 | 31 % | one-off spike 39.1 on Thu 10 Sep |
| Эко (Chimgan) | Tashkent | 14.1 | 13.2 | 3.3 | 24 % | leisure: holiday / weekend peaks |
| Заркайнар | Tashkent | 1.7 | 1.6 | 0.4 | 23 % | tiny mean inflates CV; barely moves |
| Келес | Keles | 10.2 | 9.4 | 2.3 | 23 % | weekend peak |
| Алпомиш | Tashkent | 3.6 | 3.8 | 0.8 | 22 % | tiny mean; one low Sunday |
| Кукча | Tashkent | 17.4 | 16.5 | 3.7 | 21 % | Saturday peak |
| Цум | Tashkent | 27.7 | 29.4 | 5.8 | 21 % | one low day (15.8 on Sun 6) against a median of 29 |
| Уч Кахрамон | Tashkent | 13.9 | 13.2 | 2.9 | 21 % | Sunday peak |

None of these changes the target choice, but they are flagged as **noisier labels** and are candidates for a median-target or down-weighting robustness check.

## 4. Temporal behaviour

Mean `sales_index` across all branches by date, and each branch indexed to its own 10-day mean:

| date | Tue 1 | Wed 2 | Thu 3 | Fri 4 | Sat 5 | Sun 6 | Mon 7 | Tue 8 | Wed 9 | Thu 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean sales_index | 21.3 | 20.1 | 19.3 | 20.9 | **22.6** | 21.1 | 19.8 | 19.2 | 19.6 | 20.0 |
| vs own branch mean | 1.04 | 0.99 | 0.95 | 1.03 | **1.11** | 1.02 | 0.98 | 0.95 | 0.97 | 0.98 |
| branches with best day | 22 | 12 | 5 | 12 | **41** | 19 | 5 | 0 | 6 | 5 |

- Mild common calendar effect: Saturday +10 %, Thursday 3 and Tuesday 8 −5 %. Weekend mean 21.8 vs weekday 20.0.
- Tuesday 1 Sep behaves like a weekend day (+4 %, 22 best days). It is Independence Day in Uzbekistan.
- A shared date factor explains only **12.5 %** of within-branch day-to-day variation. The 10-day mean contains a slightly leisure-heavy mix (2 weekend days + 1 holiday out of 10). No correction is applied; it is documented.

**Ranking stability (Spearman ρ between daily branch rankings).**

| measure | value |
|---|---|
| ρ between any two days | 0.90 – 0.97 (mean 0.95) |
| ρ of each single day vs the 10-day mean | 0.95 – 0.98 |
| ρ among the leisure days (Tue 1, Sat 5, Sun 6) | 0.96 – 0.97 |
| ρ leisure days vs working days | 0.90 – 0.95 |
| median rank movement per branch (of 127) | 20 positions |
| branches that never leave their daily quartile | 53 of 127 |

- Location-specific performance clearly exists and is visible even in a single day of data.
- Мега, Зенит and МедГор never leave the top 3. Movement is largest in the middle of the distribution, where branches sit 1–2 points apart, and small at both ends.
- The only structured reshuffle is the leisure-day vs working-day mix, not random noise.
- Stability across *seasons* cannot be checked with 10 days. This is the main caveat on the target.

## 5. What the target measures: sales vs orders vs average check

| relation | Pearson r |
|---|---|
| sales vs orders | 0.93 |
| sales vs average check | 0.49 |
| orders vs average check | 0.17 |

Variance of log(sales) decomposes into orders **60 %**, average check **21 %**, covariance **19 %**.

| branch group (split at medians) | n | mean target |
|---|---|---|
| orders above median, check above median | 37 | 33.2 |
| orders above median, check below median | 26 | 22.0 |
| orders below median, check above median | 26 | 15.4 |
| orders below median, check below median | 38 | 10.1 |

- The sales index is primarily a **volume** index, with ticket size as a secondary component.
- Two routes to the same sales level: volume with ordinary tickets (Мега, Зенит, Ганга, СамПИ: central Tashkent footfall locations) vs fewer but larger orders (МедГор, Бухара, Чирчик, Карши, Коканд).
- Average check is systematically higher outside Tashkent (median check index 72 vs 57 inside). Orders is the component most tied to catchment and footfall; check carries a regional / format component. Both are worth modelling or at least inspecting separately.

## 6. Geography

| region | branches | median target | mean | min | max |
|---|---|---|---|---|---|
| Tashkent | 102 | 15.7 | 19.9 | 1.7 | 86.5 |
| other cities (19) | 25 | 20.7 | 22.3 | 1.6 | 43.4 |

- Only four non-Tashkent cities have more than one branch (Andijan 4, Bukhara 2, Namangan 2, Chirchiq 2). City can only enter a model as a coarse Tashkent-vs-other flag.
- Regional branches are not weaker; they have a higher median and a narrower spread. Tashkent contains both the best branches and most of the weakest, so the variation a location model must explain lives mainly inside Tashkent.
- Raw latitude and longitude correlate with the target at |r| < 0.07. The strongest Tashkent branches cluster in the centre and north-east, but there is no smooth gradient; the signal must come from local context features, not coordinates.
- The two `Safia mini` branches average 4.4 vs 20.6 for `Safia`; format should be a feature or the two treated separately.

## 7. Conclusions

- **Target:** the 10-day mean of daily `sales_index` (`avg_sales_index`), used as provided.
- **Why the mean is acceptable:** median branch std 2.0 points, CV ~11 %; mean and median agree at r = 0.998 and differ by less than 1 point for 108 of 127 branches; daily rankings correlate at ρ = 0.90–0.97 between days and ρ ≥ 0.95 with the 10-day mean.
- **Scale:** 100 is the best single day of any branch (Мега, Sat 5 Sep). Branch averages span 1.6–86.5; one branch above 60, eight above 40. The distribution is strongly right-skewed (median 17.8, mean 20.4, skewness 1.9), so relative errors and a log/rank view matter as much as absolute errors.
- **Caution, only 10 days:** one weekend and one public holiday. The calendar effect is small (~12 % of within-branch variation) but present. Seasonality, trend and cross-month ranking stability cannot be assessed; present the target as "early-September 2026 performance".
- **Potential noise:** leisure locations with a holiday/weekend regime (Бочка, Эко Chimgan), a probable closure at the end of the period (Жомий), one-off spikes (Матонат, Цум). Flagged as noisier labels; candidates for a median-target or down-weighting check.
- **Meaning:** mostly a volume index (r = 0.93 with orders, ~60 % of log-variance) with a ticket-size component that is higher outside Tashkent.
- **Geography:** 102 of 127 branches are in Tashkent, where the full range of outcomes occurs; city is at best a coarse flag.
