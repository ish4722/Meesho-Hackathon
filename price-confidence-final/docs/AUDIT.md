# Phase 1 audit: the pipeline as it was (commit `ea8422b`)

> How each problem below was fixed, and the v2 results, are in [`METHODOLOGY.md`](METHODOLOGY.md) (section 1 maps every item to its fix).

This records what the code actually did before the v2 changes, from reading every file and running the pipeline in a clean copy (Python 3.13, numpy 2.5.3, pandas 3.0.5, statsmodels 0.15.0, Node 22).

## Pipeline as implemented

```
generate_dataset.py ──► products.csv, weekly_panel.csv, reviews.csv, market_weekly.csv
        │                 (300 listings: 150 soap, 150 t-shirt; one market series per category)
        ▼
train_models.py ──────► model.json
   prep(): lagged reviews/rating, EMA of past conversion, price vs CURRENT-week market median,
           CURRENT-week category demand index, rating prior from final ratings of ALL products
   A1 conversion: Poisson GLM, product fixed effects, log(impressions) offset; 2nd stage on listing quality
   A2 visibility: Poisson GLM, product fixed effects; 2nd stage on listing quality
   B  keep share: pooled RTO rate, return logit with FE, resale share, review rate
   backtest(): 20% of products held out, 8-week orders, median absolute % error
        ▼
build_app.py ─────────► dist/index.html  (model.json + engine.js inlined into app_template.html)
        ▼
engine.js recommend():
   unitEconomics → floorPrice (first whole rupee with expected contribution ≥ 0 per dispatched order)
   candidates: prices ending in 9 from 0.6×p25 to p75 (step ₹10)
   simulate(): ONE deterministic expected path per candidate, 12 weeks, recursive on its own
               predicted reviews and conversion (no future truth used)
   pick: argmax of that single point forecast above the floor; then the lowest price within 5%
   launchOption(): same deterministic path for good / typical / weak hand-set scenarios, 26 weeks
```

## Problems found

| # | Area | What the code did | Why it matters |
|---|---|---|---|
| 1 | Generator | One price-response curve per category: every product shares the same `a, b, c` sensitivities; products differ only through a latent quality that shifts the "fair" price. Demand always falls with price. | A model can only rediscover "higher price → fewer orders". No product for which a higher price is better, no "too cheap to trust", no reputation penalty. |
| 2 | Generator | Two categories, one market median series per category, no cancellations, no stock-outs, no competitor-price dynamics beyond a drifting median. | Narrow test bed; a funnel stage the case brief cares about (cancellations) is missing. |
| 3 | Generator | Price changes come from seller strategies (several react to their own sales) but the panel does not record why a price changed. | Endogenous and exogenous variation are indistinguishable, so the price effect cannot be audited. |
| 4 | Reproducibility | Re-running `generate_dataset.py` with the committed seed reproduces soap rows exactly but **not** t-shirt rows (7,656 vs 7,627 listing-weeks; 297 differences in `model.json`). numpy's random streams changed between versions and nothing records which version produced the committed files. | "Fixed seed → same data" was not true across environments. |
| 5 | Leakage (hold-out) | `train_models.py` sets the rating prior and the good/typical/weak rating quantiles from `final_rating` of **all** products, including the backtest hold-out. | Information from held-out products' futures enters the backtest. |
| 6 | Leakage (temporal) | Price is measured against the **same week's** market median and visibility uses the **same week's** realised category demand index. | Neither is known when forecasting week t from week t−1. |
| 7 | Schema | `products.csv` mixes launch-time facts, future-derived summaries (`final_rating`, `review_count`, `total_*`, `status`, `current_listing_price`) and `hidden_*` truth. Only a naming convention separates them; nothing enforces it. | A future-derived or hidden column could become a feature without anyone noticing. |
| 8 | Evaluation | The "8-week backtest" feeds each held-out listing its **actual** past reviews, rating and conversion week by week (teacher forcing), reports a median absolute % error, drops listings with fewer than 5 orders, and has no time split and no untouched final hold-out. | The reported 31% / 41% error is not the error of the forecast the app actually makes (a 12-week recursive forecast from launch). |
| 9 | Uncertainty | One deterministic path per price. Uncertainty appears only as three hand-picked scenarios (±0.674 SD). No parameter uncertainty, no week-to-week noise, no probability of loss. | The app shows a single number as if it were known. |
| 10 | Decision | The best price is often the top of the search range (Seller 6: best = ₹229 = band top). The code flags `cappedAtBand` but never explores beyond it. | The optimum may lie outside the searched range. |
| 11 | Economics | No cancellations, no RTO handling fee option, no advertising, no launch or inventory cash. The generator's hidden contribution omitted packaging while the app's floor included it. The floor is called the "lowest safe price". | Accounting differed between the truth and the app; "safe" overstated what the floor guarantees. |
| 12 | Causality | UI copy reads the within-listing price coefficient as a causal effect ("+10% price → −20% orders"). Fixed effects remove fixed product quality, not reactive price cuts or campaign timing. | Overclaims what the data identifies. |
| 13 | Tests | `test_engine.js` prints values and asserts nothing. No Python tests. `truth_test.py` re-implements the generator's mechanics by hand, so it can drift from the generator. | No automated check that anything still works. |

## Baseline numbers (re-run of the committed code)

| Item | Value |
|---|---|
| Listing-weeks / listings | 7,656 / 300 (committed files: 7,627) |
| Products with more than one price | 67% |
| Median within-product SD of log price | 0.041 |
| Weekly orders: mean / median / variance-to-mean | 28.4 / 11 / 75 |
| Seller 6 (soap, cost 64/12/15): floor / start / range | ₹145 / ₹209 / ₹189–229, best at band top (`cappedAtBand`) |
| T-shirt example: floor / start | ₹275 / ₹339 |
| `recommend()` time | 1.5 ms |
