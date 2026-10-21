# Price Confidence v2: methodology, results and limits

Team Edge Case, IIT Roorkee. Meesho DICE Challenge S3.

> **Everything in this repository runs on synthetic data.** We do not have access to Meesho's listing-week data, nothing here was scraped, and no synthetic number should be read as a Meesho observation. Metrics measured on synthetic data show that the pipeline recovers the assumptions built into the generator. They do not show real-world accuracy.

Exact figures quoted below come from `code/reports/evaluation.md` and `code/reports/data_quality.md`, produced by the commands in section 9 (seed 20261021, generator 2.0.0, numpy 2.5.3, pandas 3.0.5, statsmodels 0.15.0, Node 22).

---

## 1. Problems found

The full audit of the previous pipeline (commit `ea8422b`) is in [`AUDIT.md`](AUDIT.md). In short, and with where each is fixed:

| # | Problem in v1 | Fixed in v2 by |
|---|---|---|
| 1 | One price-response curve per category; demand always fell with price | Six price-response archetypes per product (`synth_market.py`), section 3 |
| 2 | Two categories, no cancellations, one market series per category | Four categories × three subcategories, cancellations, stock-outs, COD/prepaid RTO, competitor price per subcategory |
| 3 | Price changes had no recorded reason | `price_change_reason` on every week; exogenous vs endogenous share in the data-quality report |
| 4 | Same seed did not reproduce the data across numpy versions; nothing recorded the versions | `dataset_manifest.json` with seed, versions, config hash and file hashes; `run_all.sh` checks the hashes |
| 5 | Rating prior built from final ratings of all listings, including the holdout | `features.fit_constants()` uses training rows only; test `test_constants_come_from_training_rows_only` |
| 6 | Same-week market median and demand used as features | Lag-1 market features; temporal rule enforced in `schema.py`; tests fail on same-week market or outcomes |
| 7 | Hidden and future columns separated only by naming | Every column has a declared role; `design()` refuses anything not whitelisted |
| 8 | Teacher-forced backtest, median % error, no time split or final holdout | Product × time split with an untouched holdout; tasks A, B, C; MAE/WAPE/bias; generator-judged decisions |
| 9 | One deterministic path per price | Monte Carlo with parameter, listing-level and weekly uncertainty |
| 10 | Optimum often at the edge of the search range, unreported | Explore range above the cap and a boundary report |
| 11 | Economics gaps (no cancellations, RTO fee, ads, launch cash); floor called "safe" | `economics.py` / `engine.js` share one set of formulas; "break-even price", never "safe" |
| 12 | Within-listing price effect presented as causal | Copy says "association learnt within listings"; the learning policy proposes a randomised test |
| 13 | No tests | 47 Python tests + 16 JavaScript engine tests (section 8) |

## 2. Files changed and why

All paths under `price-confidence-final/`.

| File | Status | Why |
|---|---|---|
| `code/synth_market.py` | new | The synthetic world: categories, archetypes, market, funnel, economics of a week. Separate from the script that runs it so evaluation can reuse it as ground truth. |
| `code/generate_dataset.py` | rewritten | Runs seller strategies through the world, records price-change reasons, writes the manifest. Scenarios `default`, `homogeneous`, `low_price_variation`. |
| `code/schema.py` | new | Role of every column, the engineered features with their source columns and lags, the per-model whitelist, `LeakageError`. |
| `code/data_quality.py` | new | 40 checks (schema, missing values, duplicates, funnel identities, rates, regular weeks, coverage, concentration, price variation). Writes `reports/data_quality.{md,json}`; exits 1 on a hard failure. |
| `code/economics.py`, `code/assumptions.json` | new | One definition of contribution, break-even and cash exposure; every fee and policy limit in one editable file. |
| `code/features.py` | new | History-only features and whitelisted design matrices. |
| `code/train_models.py` | rewritten | Models on whitelisted features, parameter covariances, listing-effect covariance, price-slope spread, state ranges; `--train-ids` and `--cutoff` for evaluation. |
| `code/model_schema.py` | new | Contract for `model.json`; checked before writing and before building the app. `engine.js` has the same check. |
| `code/engine.js` | rewritten | Seeded Monte Carlo, candidate metrics, three policies, infeasibility reasons, boundary report, strategy comparison, forecasts. Old entry points (`recommend`, `launchOption`, `cohortEvidence`) kept with new fields. |
| `code/evaluate.py`, `code/eval_runner.js` | new | Splits, tasks A/B/C, baselines, parameter recovery, generator-judged decisions. Task C and decisions run the same `engine.js` the app runs. |
| `code/app_template.html`, `code/build_app.py` | updated | Any category in the model; type, variant and positioning inputs; risk setting; "How sure are we?"; infeasible and price-test views; corrected copy. `build_app.py` validates the model and embeds a short evaluation summary. |
| `code/build_workbook.py` | rewritten | Excel copy with a data dictionary generated from `schema.py`, data-quality results and formula checks (99 formulas, 0 errors after LibreOffice recalculation). |
| `code/tests/` | new | pytest suite and `engine.test.js`. |
| `code/run_all.sh`, `code/requirements.txt` | new / pinned | One command to rebuild and check everything; exact versions. |
| `code/test_engine.js` | removed | Printed values, asserted nothing; replaced by `tests/engine.test.js`. |
| `legacy/v1/` | new (frozen copies) | v1 code, model, data and demo, so the numbers in the Round 2 deck and video (Seller 6: floor ₹145, start ₹209) stay reproducible. |
| `docs/AUDIT.md`, `docs/METHODOLOGY.md` | new | This documentation. |
| CSVs, `model.json`, `dist/index.html`, `OPEN_THIS_Price_Confidence_demo.html`, `Price_Confidence_Synthetic_Dataset.xlsx` | regenerated | Outputs of the v2 pipeline. |

The Round 2 deck (`Price_Confidence_DICE_Round2.pptx`) and the demo video were not changed. Their numbers are v1 numbers.

## 3. Generator design (`synth_market.py`, `generate_dataset.py`)

**World.** 4 categories (soap, t-shirt, bedsheet, earrings) × 3 subcategories, 150 listings per category, launches from 1 Sep 2025 to 3 Aug 2026, weekly data to 21 Sep 2026. Each subcategory has its own competitor median (trend + AR(1) noise + season), p25 ≈ 0.80 × median and p75 ≈ 1.22 × median. Category demand has a monthly season and festival weeks. Calendar: illustrative festival and campaign weeks, not Meesho's sale dates.

**Listings.** Observable: subcategory, positioning tier (basic / standard / premium), a variant attribute (pack size, fabric, thread count, material), a listing-quality score correlated with latent quality, a delivery promise. Seller-private: unit cost, packaging, gift, loss budget. Latent (generator only, `hidden_*`): quality, archetype and its parameters, fair-price multiplier, demand and conversion multipliers, stock-out risk, COD share, RTO propensity, seller strategy.

**Price response.** Relative price is measured against the buyer's fair price (`rel = price / (subcategory median × fair_mult) − 1`, clipped to [−0.6, 1.0]):

```
f(rel) = −β·rel − κ_hi·max(rel, 0)² − κ_lo·max(−rel − θ, 0)² + sig·[bump(rel) − bump(0)],   bump(r) = exp(−((r − c)/w)²)
```

| Archetype | β | κ_hi | κ_lo, θ | sig | Meaning |
|---|---|---|---|---|---|
| commodity | 3.4–4.8 | 1–3 | – | – | buyers shop on price |
| moderate | 1.6–2.6 | 0.5–1.5 | – | – | typical |
| insensitive | 0.3–0.9 | 0–0.6 | – | – | price barely matters in the band |
| premium_signal | 0.9–1.5 | 0.3–1 | – | 0.30–0.50 | a higher price signals quality up to a point |
| too_cheap_distrust | 1.2–2.0 | 0.5–1.5 | 5–9, 0.08–0.18 | – | prices far below the market look fake |
| reputation_penalty | 1.6–2.6 | 6–12 | – | – | above-fair prices collapse |

β is scaled by a subcategory multiplier. Archetype probabilities depend on tier (budget listings are mostly commodity; premium listings signal quality only if quality is genuinely high). `f` is split across the funnel: 0.35 on click-through, 0.35 on add-to-cart, 0.30 on ordering. So for some products a higher price is better and for others a lower one, which `tests/test_generator.py::test_truth_optimal_price_differs_by_archetype` checks in the generator itself.

**Funnel and after-order events, per week.** Impressions (Poisson-lognormal; demand × rank × launch boost × campaign × stock), clicks, add-to-cart, orders (binomial chain), cancellations (COD 6%, prepaid 3%, more when overpriced), dispatch, RTO (COD rate by category, worse with slower delivery and overpricing, listing propensity), delivery, returns (quality, attribute, overpricing, listing quality) with six reasons, resale of returns, reviews (14% of kept, 18% of returned orders), stars (quality, overpricing; returned buyers rate lower). Rank depends on reviews, rating, delivery promise and recent conversion, so early sales feed later visibility. Overdispersion comes from lognormal weekly shocks on impressions and orders and from listing-level multipliers (weekly orders: variance ≫ mean).

**Seller behaviour.** Six strategies (investment, market, premium, cost-plus-stale, reactive, experimenter) set prices; campaign discounts are joined at random. Each week records `price_change_reason`: exogenous (`experiment`, `campaign_discount`, `campaign_ended`, `market_follow`) or endogenous (`reactive_cut`, `reactive_raise`, `raise_after_reviews`). Sellers may delist when orders dry up or losses pass their budget, so failures stay in the data.

**Leakage-safe latent variables.** Latent drivers live only in `hidden_*` columns (role `latent`); summaries of the future (`final_rating`, `total_*`, `status`, …) have role `future_derived`; seller costs are `seller_private`. None can enter a model (section 4).

**Provenance and reproducibility.** `dataset_manifest.json` records provenance ("SYNTHETIC … Not Meesho data"), generator and schema versions, seed, scenario, the full configuration and its SHA-256, row counts, SHA-256 of every file, Python / numpy / pandas versions and every column's role. Same seed, scenario and versions give byte-identical files (tested).

**Data-quality report** (`reports/data_quality.md`, default data): 40 checks, 0 failures, 1 warning (the smallest category × subcategory × tier cell has 2 listings). 600 listings, 15,250 listing-weeks, 44,545 reviews. 79% of listings have more than one selling price; median within-listing SD of log price 0.046; 86% of price changes are exogenous. The `low_price_variation` scenario triggers the price-variation warning (tested).

## 4. Feature whitelist and leakage safeguards (`schema.py`, `features.py`)

**Roles.** Every column of every table has a role: `id`, `key`, `meta`, `observed_static`, `decision`, `calendar`, `condition`, `market`, `outcome`, `state_after`, `future_derived`, `seller_private`, `accounting`, `latent`.

**Temporal rule.** A feature may use a column from the *same week* only if its role is `key`, `observed_static`, `decision` (the seller's own price, discount, delivery promise) or `calendar`. Outcomes, end-of-week state, market values and conditions may be used only from *earlier* weeks. `future_derived`, `seller_private`, `accounting`, `latent`, `meta` and `id` are never features. Exposures (impressions for conversion, in-stock share for visibility, binomial denominators) enter only as offsets of conditional models; at forecast time they come from another model or the plan.

**Whitelist.** Each engineered feature declares its source columns and lags (`schema.FEATURES`); each model has a fixed list (`schema.MODEL_FEATURES`):

| Model | Features |
|---|---|
| conversion | rel, rel², rel×premium, rel×basic, discount, festival, campaign, delivery promise, rating shown at start of week, log(1 + reviews before the week) |
| visibility | launch weeks 1–2, 3–4, log reviews, rating, EMA of past conversion, log of last week's demand, campaign, delivery promise |
| returns, RTO | max(rel, 0), delivery promise |
| cancellations, stars | max(rel, 0) |
| second stage (listing level) | listing quality, tier, subcategory, attribute |

`rel` = selling price / **last week's** subcategory median − 1.

**Enforcement.** `schema.assert_features_allowed()` raises `LeakageError` for a column outside the whitelist, an undeclared feature, a never-feature role, or a same-week outcome / market value. `features.design()` calls it and also refuses NaN. Data-summarising constants (rating prior, conversion norm, listing-quality scaling, rating quantiles) come from `fit_constants()` on training rows only.

**Tests** (`tests/test_leakage.py`): every whitelisted feature passes the rule; seven hidden, future, private and same-week columns are rejected; a same-week outcome feature and a same-week market feature are rejected; a hidden column injected into a design matrix is refused; tripling a listing's outcomes from week 6 onward leaves every feature in weeks 1–6 unchanged (and does change later weeks, so the test is not vacuous); doubling this week's market leaves this week's features unchanged; changing held-out listings and post-cutoff weeks does not change training constants or coefficients; the v1 leak (`final_rating`) is rejected.

## 5. Evaluation methodology and results (`evaluate.py`)

**Splits.** Listings are split 60 / 20 / 20 (train / validation / final holdout), stratified by category, seed 11. Evaluation models are trained on **training listings, weeks before 2 Mar 2026** only. Validation and holdout listings never enter training. Nothing was tuned on validation or holdout results; the one modelling change made during evaluation (the price-slope spread estimator, section 6) was motivated by the training-set parameter-recovery check.

**Tasks.**

- **A, one week ahead.** Validation / holdout listing-weeks on or after the cutoff, week 2 onward. Inputs: the listing's own past, last week's market, this week's price and calendar. The listing's level is updated from its own past weeks by empirical Bayes: multiplier = (a + Σ past orders) / (a + Σ past model means), with a from the fitted listing-effect variance.
- **B, launch time.** Listings launched on or after the cutoff: week-1 and weeks 1–4 orders from launch-time facts, the planned prices and calendar, and the market as of launch.
- **C, recursive 12 weeks.** The same listings simulated from launch by `engine.js` without any realised outcome; market frozen at launch. Listings that stopped early are compared over the weeks they were live.
- **Decisions.** Every holdout listing is relaunched in the generator on 29 Jun 2026 with its hidden response and its seller's private costs; 40 replications per price on a ₹20 grid plus every recommended price. Regret = generator's mean 12-week contribution at the best price at or below the cap minus that at the recommended price.

**Metrics.** MAE, WAPE (Σ|error| / Σ actual), bias (Σ error / Σ actual), 80% interval coverage, Brier score and binned calibration for the probability of loss, slices by category, archetype, launch-price segment and seller strategy (archetype and strategy are generator-only and used for slicing only), regret, true probability of loss, share of picks with negative true expected contribution, below the true per-order break-even, or breaching the seller's budget more than 10% of the time.

**Baselines.** Category × week-since-launch average; pooled price response (Poisson GLM with category, launch phase, rel, rel², no listing effects); last week's orders (Task A); the v1 specification refitted on the same training rows (same-week market, full-sample rating prior, no tier / subcategory / attribute terms; teacher-forced in A, its deterministic path in B and C); v2 with no price change (last week's price in A, launch price held in C); the v1 decision rule on the v2 model (best expected path at the typical listing, between break-even and the cap, no uncertainty or risk limits); the market median held.

### Results (holdout unless stated)

**Task A: one week ahead** (2,259 holdout listing-weeks)

| Method | MAE (orders) | WAPE | Bias |
|---|---|---|---|
| Category × week average | 27.0 | 87.7% | −33.3% |
| Pooled price response | 26.3 | 85.3% | −34.4% |
| v1 spec (same-week market, teacher-forced) | 14.7 | 47.6% | −23.4% |
| Last week's orders | 11.2 | 36.3% | −3.2% |
| v2, launch-time listing level only | 18.4 | 59.8% | −30.4% |
| v2, last week's price instead of this week's | 9.2 | 29.8% | +1.8% |
| **v2, online listing level** | **9.1** | **29.4%** | +3.9% |

80% interval coverage: 73.6% (validation 75.3%): intervals are somewhat too narrow. Validation WAPE for the main model: 29.9%. Knowing this week's price improves the one-week forecast only slightly over last week's price (29.4% vs 29.8%), which is expected when most weeks have no price change.

**Task B: launch time** (50 holdout listings; validation 57)

| Method | Week 1 WAPE | Weeks 1–4 WAPE | Weeks 1–4 bias |
|---|---|---|---|
| Category × week average | 69.1% | 100.5% | +5.4% |
| Pooled price response | 89.6% | 101.0% | +7.0% |
| v1 spec, deterministic path | 68.6% | 85.7% | −37.7% |
| **v2 engine, median of simulations** | 73.8% | **71.5%** | +3.8% |

Week 1 for a brand-new listing is barely predictable by any method (the v2 engine is slightly worse than the category average in the holdout and better in validation: 59.2% vs 69.6%). 80% coverage for weeks 1–4: 82.0% (validation 75.4%).

**Task C: recursive 12 weeks** (50 holdout listings, 36 with all 12 weeks; validation 57 / 44)

| Method | WAPE | Bias |
|---|---|---|
| Category × week average | 99.9% | +4.1% |
| Pooled price response | 99.2% | +5.8% |
| v1 spec, deterministic path | 92.8% | −41.7% |
| v2, launch price held (no price change) | 77.2% | +2.3% |
| **v2 engine, recursive, median** | **73.8%** | −1.9% |
| v2 engine, recursive, mean | 142.2% | +90.9% |
| v2 teacher-forced (uses realised history; diagnostic only) | 54.3% | +1.5% |

Validation: recursive median 73.3% (bias −26.1%), teacher-forced 48.9%. The gap between teacher-forced and recursive (54% → 74%) is the cost of not knowing how a new listing will be received. 80% coverage of 12-week orders: 86.1% holdout, 77.3% validation; 12-week contribution: 86.1% / 75.0%. Predicted vs observed share of listings losing money: 18.1% vs 22.2% (Brier 0.07) holdout, 17.7% vs 11.4% (Brier 0.09) validation. Contribution over the observed weeks (seller's private costs, evaluation only): median-forecast MAE ₹8,911 against a mean actual of ₹8,269, so contribution for a single new listing is only roughly predictable.

**Important finding: the simulated right tail is too heavy.** The mean over simulations overshoots actual orders by 64–91% while the median is close and the quantiles are roughly calibrated (validation PIT histogram near uniform). A few simulated futures in which a well-received listing gets more reviews, better rating and more visibility compound too strongly. The app therefore leads with the typical (median) future and warns when the average is far above it. Because the Balanced policy maximises the *mean*, this tail matters for decisions (below).

**Price-sensitivity recovery** (training listings; slope of log conversion per 100% price change, at each listing's own typical price): standard-tier slopes are recovered well (true vs fitted: bedsheet −2.16 / −2.11, soap −2.50 / −3.07, t-shirt −2.62 / −2.49, earrings −2.78 / −1.92). Budget-tier listings are more price-sensitive than fitted (e.g. earrings −5.02 / −3.29) and premium soap and bedsheet less (−1.50 / −2.94, −1.41 / −2.28). The fitted spread of listing slopes (0.7–1.2) is below the true within-tier spread (0.9–2.0). The naive between-listing slope is badly biased (soap +0.60, t-shirt +1.05: "dearer listings sell more"), which is why the models are fitted within listings.

**Decisions judged by the generator** (120 holdout listings)

| Policy | n priced | Mean regret | Median relative regret | Within 10% of best | True P(loss) | True mean < 0 | Budget breach > 10% |
|---|---|---|---|---|---|---|---|
| Balanced (v2) | 107 | ₹2,249 | 13.0% | 45.8% | 8.3% | 6.5% | 0.0% |
| Conservative (v2) | 107 | ₹2,177 | 7.0% | 57.0% | 8.1% | 5.6% | 0.0% |
| v1 rule on the v2 model | 117 | ₹1,996 | 12.7% | 49.6% | 9.2% | 6.0% | 0.0% |
| Market median, held | 120 | ₹3,525 | 27.5% | 28.3% | 16.5% | 13.3% | 1.7% |

- v2 returned *No feasible candidate under current constraints* for 13 listings. The v1 rule still priced 10 of them; there the generator gives a true probability of loss of 22% and a negative expected contribution for 20% of them.
- The engine's probability of loss is well calibrated against the generator: predicted 1.7% / 8.7% / 21.6% / 38.8% / 60.7% / 97.5% in six bins vs observed 3.9% / 11.0% / 26.9% / 35.7% / 52.3% / 97.9%.
- The learning policy flagged 52.5% of listings as uncertain (best price spread wider than 20% of the median) and proposed a test for 59 of them; 2 abstained. Test plans cost ₹2,721 mean regret over 12 weeks with 9.7% true P(loss); what the test teaches is not credited.
- No policy can tell archetypes apart at launch: the Balanced price is 1.05–1.07 × median for every archetype, while the best price ranges from 0.96 (commodity) to 1.18–1.19 (insensitive, premium signal). This is exactly where a price test is meant to help.

**Reading these numbers honestly.** v2 forecasts are clearly better than every baseline in tasks A and C and for weeks 1–4 in task B. For decisions, risk-aware pricing did **not** beat the v1 decision rule on mean profit in this world: Balanced has slightly higher mean regret than the v1 rule (₹2,249 vs ₹1,996), Conservative has lower median regret and the most picks within 10% of the best. What v2 adds is a slightly lower loss rate, calibrated loss probabilities, refusal to price where losses are likely, and an explicit "we don't know, test it" for half the listings. We did not change the policies or their defaults after seeing these results.

## 6. Risk formulas and policy rules (`engine.js`, `assumptions.json`)

**Draws.** For simulation s = 1…S (S = 400), with a seeded generator (mulberry32 hashed by seed and s):

- model coefficients θₛ = θ̂ + L·z, block by block, L = Cholesky factor of the cluster-robust (by listing) covariance; return price effect clamped ≥ 0 and star price effect ≤ 0;
- listing effects (conversion, visibility, returns, stars) ηₛ = L_η·z with the cross-listing covariance of second-stage residuals (sampling noise removed from the variances);
- a listing price-slope deviation δₛ ~ N(0, τ²) and an RTO effect;
- weekly noise: impressions ~ Poisson(λ·G), G ~ Gamma(mean 1, variance d_impr); orders ~ Poisson(impressions·conv·G'), G' with variance d_orders; cancellations, RTO, returns, resale and reviews binomial; stars normal per review.

Every sampler uses exactly one uniform number, and every price uses the same S streams (common random numbers), so differences between prices are not simulation noise. The baseline price never beats itself in any simulation (tested).

**Price term.** log-conversion gains (β_rel + β_tier + δ)·rel + β_rel²·rel² inside the observed price range of the tier (1st–99th percentile); outside it the curve continues with the boundary slope only in the direction that does **not** reward going further out. Simulated reviews, rating and recent conversion are kept inside their training range.

**τ (spread of listing price slopes).** Per-listing slopes from listings with ≥ 8 weeks and real price variation, combined with the DerSimonian–Laird random-effects estimator, then pooled with a prior (τ₀ = 1.0, worth 20 listings; `assumptions.json: modeling`) so a small training set cannot claim every listing responds identically.

**Per-candidate metrics** for a flat price over H = 12 weeks, from the S totals Tₛ and running minima mₛ of cumulative contribution (including any one-off launch cost):

```
mean = (1/S) Σ Tₛ      median, P10, P90 = quantiles of Tₛ
P(loss) = share of Tₛ < 0
P(budget breach) = share of mₛ < −budget            (only if the seller gives a budget)
P(beat baseline) = share of Tₛ > Tₛ(baseline price)  (same s; baseline = market median, or the offline price)
```

**Contribution** (identical in `economics.py` and `engine.js`, tested for parity including all optional fees):

```
revenue  = kept × price / (1 + GST) × (1 − commission − payment fee)
cost     = dispatched × (shipping + packaging + gift) + returns × shipping × pickup multiplier + RTO × RTO fee
         + cancelled × cancellation cost + (kept + returns − resold [+ RTO if not restockable]) × unit cost
         + resold × refurbishment + ad spend
contribution per placed order = expected contribution of one order at the moment it is placed
contribution per retained sale = contribution per placed order / share of orders kept
cumulative contribution = running sum of weekly contribution; cash exposure = its lowest point after launch cost and up-front stock
break-even price = lowest whole-rupee price with expected contribution per placed order ≥ 0
```

The break-even price is never called safe: above it, an average order makes money, but bad weeks, launch costs or unsold stock can still lose money. The app shows its likely range (P10–P90 of break-even across coefficient and listing draws).

**Candidates.** Prices ending in 9, step ₹10, from 0.6 × p25 to 1.3 × p75. Each candidate is flagged `inSupport` (inside the observed price range for the tier), `withinCap` (≤ p75, since a new listing has no reviews) and `aboveFloor`.

**Eligibility.** inSupport ∧ withinCap ∧ price ≥ break-even ∧ P(loss) ≤ 0.25 ∧ (no budget given ∨ P(budget breach) ≤ 0.10).

**Policies.**

- **Balanced:** the eligible price with the highest mean (tie-break parameter set to 0, i.e. pure maximisation). Good range: eligible neighbours with mean ≥ 90% of the best.
- **Conservative ("Careful" in the app):** the eligible price with the highest P10.
- **Learning ("Test first"):** for J = 60 draws, the best price of the expected (noise-free) path is found; if P90 − P10 of those best prices ≤ 20% of the market median, the recommendation is confident and equals Balanced. Otherwise a test is designed: three distinct prices (around P10, P50, P90 of the best prices, at least two grid steps apart, all eligible on break-even, cap and observed range), 2 weeks each over 8 weeks (middle, low, high, middle), then the Balanced price. It is simulated under the same draws; if its P(loss) and P(budget breach) pass the limits the app recommends it, with its cost against holding the Balanced price and the expected narrowing of the listing's price-slope uncertainty (posterior SD = 1 / √(1/τ² + I), I = Σ_w μ_w/(1 + d·μ_w)·(x_w − x̄)²). Otherwise it **abstains**: no recommended price, the reason, and the Conservative price as the steadiest option.
- **Nothing eligible:** status `no_feasible_candidate`, message "No feasible candidate under current constraints", reasons (how many in-range prices fail each limit, or break-even above the cap) and trade-offs (the probability of loss you would have to accept, the budget that would make a price workable, the unit cost at which the median breaks even). Limits are never relaxed silently; the returned constraints equal the configured ones (tested).

**Boundary report.** Eligible range, whether the chosen price is at its top or bottom and what sets each end (`cap`, `observed_prices`, `risk_limits`, `break_even`), and whether a price above the cap but inside the observed range has a higher mean (`optimumBeyondCap`, with that price).

**Strategy comparison** (26 weeks, same S draws, costs, market and limits): the recommended price held; the market median held; 4 weeks at 0.9 × break-even then the recommended price, with the seller's stop rule inside each simulation (move to the recommended price once losses reach half the budget) and budget breaches still counted; the offline price held. Reported: mean, P10, P(loss), P(budget breach), P(beating the recommended plan).

## 7. Comparison with the previous implementation

| | v1 | v2 |
|---|---|---|
| Data | 300 listings, 2 categories, one price curve per category | 600 listings, 4 categories × 3 subcategories, 6 price-response archetypes, cancellations, stock-outs, price-change reasons, manifest |
| Features | same-week market and demand; rating prior from all final ratings | lag-1 market; training-only constants; enforced whitelist |
| Models | conversion / visibility / returns with listing effects, second stage on listing quality | adds tier interactions, subcategory and attribute baselines, cancellations, RTO by price and delivery, stars, parameter covariances, listing-effect covariance, slope spread |
| Forecast | one deterministic path | 400 simulated futures with three kinds of uncertainty |
| Decision | best point estimate in the band, lower price within 5% | Balanced / Conservative / Learning, risk limits, infeasibility, boundary report |
| Economics | no cancellations, RTO fee, ads or launch cash; "lowest safe price" | all optional; "break-even price" with a likely range |
| Evaluation | teacher-forced 8-week median % error on a random 20% | product × time split, untouched holdout, tasks A/B/C, generator-judged decisions |
| Tests | none | 47 Python + 16 JavaScript |
| Seller 6 (soap, ₹64 + ₹12 + ₹15 gift) | floor ₹145, start ₹209, range ₹189–229 | break-even ₹145 (likely ₹139–153), Balanced ₹219 at the cap, range ₹189–219, Careful ₹209; best price uncertain (₹178–229), so Test first proposes ₹179 / ₹199 / ₹219 |
| `recommend()` time | 1.5 ms | 0.4–0.7 s (Node 22, 400 simulations) |

Measured against v1's own specification refitted on the same data: one-week-ahead WAPE 29.4% vs 47.6% (and v1's figure needs same-week market data it would not have); 12-week recursive WAPE 74% vs 93%. On decisions, see the honest reading in section 5.

## 8. Test results

```
$ python -m pytest tests -q
................................................                         [100%]
48 passed in 15s
```

48 pytest items = 47 Python tests + one item that runs `tests/engine.test.js` under node (16 JavaScript tests: determinism, common random numbers, metric ordering, constraints honoured, Balanced = max mean and Conservative = max P10, break-even sign change and no "safe" field, infeasible status with reasons, a strict loss limit reported not relaxed, **a higher price optimal for one product and a lower for another**, **no confident recommendation when price response is very uncertain**, boundary flags above the cap, common conditions in the strategy comparison, forecast quantiles, category detection, fields the app reads).

| File | Tests | What they cover |
|---|---|---|
| `test_generator.py` | 10 | byte-identical regeneration, different seed differs, manifest (provenance, versions, hashes, config hash), no hard data-quality failure, funnel identities, archetype and tier coverage, every price change has a reason, low-variation scenario warns, homogeneous scenario, truth-optimal price higher for an insensitive than a commodity product |
| `test_leakage.py` | 15 | section 4 |
| `test_economics.py` | 9 | hand-computed order, realised = expected in aggregate, zero costs and full resale, nothing kept, break-even definition, fees raise break-even, cash exposure, Python / JavaScript parity with and without all fees |
| `test_model_schema.py` | 13 | shipped and small models valid, ten kinds of broken model refused, engine refuses a wrong schema version |
| `test_engine_node.py` | 1 (16 node tests) | above |

The app was also driven headlessly (Chromium, Playwright) at 1280 px and 390 px: all four examples, all three risk settings, the infeasible view and the launch slider render without script errors and without horizontal scrolling.

## 9. Exact commands

From `price-confidence-final/code/`:

```bash
pip install -r requirements.txt          # Python 3.13; Node.js 22 must be on PATH
./run_all.sh                             # everything below, about 10 minutes
./run_all.sh --quick                     # same with a small evaluation, about 1 minute

# or step by step
python generate_dataset.py --out .                       # synthetic data + dataset_manifest.json (seed 20261021)
python generate_dataset.py --scenario low_price_variation --out /tmp/lpv   # comparison worlds
python data_quality.py --data . --out reports            # reports/data_quality.{md,json}; exit 1 on a hard failure
python train_models.py --data . --out model.json         # models on all data
python -m pytest tests -q                                # 47 Python tests + the 16 node engine tests
node --test tests/engine.test.js                         # engine tests alone
python evaluate.py --out reports                         # reports/evaluation.{md,json}, evaluation_*.csv
python build_app.py                                      # dist/index.html (offline, single file)
cp dist/index.html ../OPEN_THIS_Price_Confidence_demo.html
python build_workbook.py --out ../Price_Confidence_Synthetic_Dataset.xlsx
```

## 10. Limitations and roadmap for real data

**Limitations.**

1. Synthetic data only. The models can only rediscover what `synth_market.py` assumes; good scores here are not evidence about Meesho buyers.
2. Price effects are within-listing associations. Fixed effects remove fixed product differences, not reactive price cuts (12% of changes here) or campaign timing. Only randomised tests identify causal effects.
3. The simulated right tail is too heavy (Task C mean bias +64–91%), so mean-based figures, including the Balanced objective, are optimistic; the app leads with the median.
4. One-week intervals are a little narrow (74–75% coverage for 80%); listing-level price sensitivity is under-dispersed (fitted spread 0.7–1.2 vs true 0.9–2.0) and tier slopes are biased for budget and premium listings.
5. Risk-aware policies did not beat the v1 point rule on mean regret in this world (section 5).
6. Market is frozen at launch for 12 weeks; competitors do not react to the seller; stock-outs are an average, not simulated; ads are modelled only as a fixed weekly budget.
7. Fees, shipping, GST treatment and policies are team assumptions (`assumptions.json`), not Meesho's.
8. Small evaluation samples for launch-time tasks (50–57 listings per split); slices by archetype have single-digit counts.
9. `recommend()` takes about half a second in a browser; fine for a demo, not for batch pricing of a catalogue without moving the simulation server-side.

**Roadmap for real data** (only with authorised access):

1. Map Meesho's listing-week tables onto `schema.py` roles before any modelling; run `data_quality.py` and treat its price-variation and exogenous-change checks as a gate.
2. Replace `assumptions.json` values with authorised fee, shipping, return and RTO policies, versioned with the model.
3. Refit with the same whitelist and splits; add a rolling-origin backtest over several cutoffs; recalibrate interval width and the listing-effect tails on a validation period.
4. Run seller-consented randomised price tests (the Learning policy's design) to estimate causal price effects by category and tier, and use them to correct the observational slopes.
5. Choose the default policy and the risk limits (`max_loss_probability`, budget breach) with sellers and on a pilot with a control group, not on synthetic results.
6. Monitor live: predicted vs realised orders and loss rates per week, drift in return and RTO rates, and the share of sellers the tool declines to price.
