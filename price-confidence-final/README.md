# Price Confidence

Team Edge Case, IIT Roorkee. Meesho DICE Challenge S3 (pricing case).

A launch-price tool for new sellers: it works out a seller's break-even price, simulates many possible 12-week futures at every candidate price, and recommends a price (or a short price test, or nothing) according to the seller's appetite for risk.

> **All data is synthetic.** It is generated from assumed buyer behaviour (`code/synth_market.py`) to demonstrate the method. It is not Meesho data, nothing was scraped, and no number here is a forecast for real sellers. Good scores on synthetic data show that the pipeline recovers the generator's assumptions, not that it would be accurate on Meesho.

## What's in this folder

| Path | What it is |
|---|---|
| `OPEN_THIS_Price_Confidence_demo.html` | The working demo (v2). Double-click it or drag it into Chrome. Works offline. |
| `docs/METHODOLOGY.md` | **Start here.** Generator design, leakage safeguards, evaluation method and results, risk formulas and policies, comparison with v1, tests, commands, limitations and the roadmap for real data. |
| `docs/AUDIT.md` | What the previous pipeline did and the 13 problems found in it. |
| `code/` | Everything needed to regenerate the data, retrain, evaluate, test and rebuild the demo. |
| `code/reports/` | `data_quality.md`, `evaluation.md` (+ JSON and CSV detail). |
| `Price_Confidence_Synthetic_Dataset.xlsx` | Excel copy of the dataset with a data dictionary, data-quality results and formula checks. |
| `Price_Confidence_DICE_Round2.pptx` | The Round 2 deck. **Its numbers are v1 numbers** (e.g. Seller 6: floor ₹145, start ₹209), as is the demo video: https://www.loom.com/share/c300fa0f2e26450d9860fd87da4de9cf |
| `legacy/v1/` | Frozen v1 code, model, data and demo, so the deck and video stay reproducible. `ROUND2_NOTES.md` has the v1 model notes and the response to the Round 2 review. |

## Run it

From `code/` (Python 3.13 and Node.js 22; exact library versions in `requirements.txt`):

```bash
pip install -r requirements.txt
./run_all.sh            # data -> data-quality report -> models -> tests -> evaluation -> app -> workbook (~10 min)
./run_all.sh --quick    # same with a small evaluation (~1 min)
```

Step by step and all options: `docs/METHODOLOGY.md`, section 9.

## In one paragraph

600 synthetic listings in 4 categories × 3 subcategories, where each product reacts to price in one of six ways (commodity, moderate, insensitive, price-signals-quality, too-cheap-to-trust, reputation penalty). Models of conversion, visibility, cancellations, RTO, returns and stars are fitted within listings on features known at the start of each week only (enforced by `schema.py` and tests). The engine draws model coefficients, the listing's own level and price sensitivity, and weekly luck, and reports for every price the mean, median, P10, P90, chance of a loss, chance of passing the seller's loss budget and chance of beating the market median. **Balanced** maximises expected profit under those limits, **Careful** maximises the bad case, **Test first** proposes an affordable three-price test when the best price is too uncertain, or abstains. If no price passes, the app says *No feasible candidate under current constraints* and shows what would change that.

On the synthetic holdout: one-week-ahead orders WAPE 29% (last week's orders 36%, v1 specification 48%); 12-week orders simulated from launch WAPE 74% (category average 100%, v1 93%); the engine's probability of loss is well calibrated against the generator. Risk-aware policies did **not** beat the v1 decision rule on mean profit in this world; they lose money slightly less often, decline to price 13 of 120 listings where the v1 rule's price loses money 22% of the time, and flag half the listings as needing a price test. Details and caveats: `docs/METHODOLOGY.md`, section 5.
