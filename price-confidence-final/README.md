# Price Confidence: final submission pack

Team Edge Case, IIT Roorkee. Meesho DICE Challenge S3 (pricing case).

## What's in this folder

| File | What it is |
|---|---|
| `Price_Confidence_DICE_Round2.pptx` | The 10-slide Round 2 deck (DICE template, speaker notes on every slide). Slide 8 links to the team's demo video on Loom through a clickable storyboard, a button and a QR code, so it works in PowerPoint and in a PDF export. |
| `Price_Confidence_Models_Explained.pdf` | Two-page explainer of the three models, shareable with the team |
| `OPEN_THIS_Price_Confidence_demo.html` | The working demo. Double-click it, or drag it into Chrome. Works offline. |
| `Price_Confidence_Synthetic_Dataset.xlsx` | The dataset with formula-driven checks |
| `code/` | Everything needed to regenerate the data, retrain the models and rebuild the demo |

**All data is synthetic.** It is generated from assumed buyer behaviour to demonstrate the method. Nothing here is a forecast for real Meesho sellers.

Demo video (recorded by the team): https://www.loom.com/share/c300fa0f2e26450d9860fd87da4de9cf

## Pipeline (run inside `code/`)

```
python generate_dataset.py     # 150 soap + 150 t-shirt listings, weekly from launch to 21 Sep 2026
python train_models.py         # fits A1, A2, B; hold-out backtest; writes model.json
python build_app.py            # model.json + engine.js + app_template.html -> dist/index.html
node test_engine.js            # pricing engine on five sample sellers
python truth_compare.py        # runs Seller 6's soap through the simulator's own buyer rules -> truth.json
node model_compare.js model.json   # compares the models' answers with that ground truth
python build_workbook.py       # optional: Excel workbook of the dataset
```

## The models

All three give every listing its own baseline (fixed effects), so effects are learnt from a listing changing over time, never from comparing good products with bad ones. A second step puts those baselines on listing quality, so a brand-new listing gets a starting level.

- **A1 Conversion:** orders per impression (Poisson, log-impressions offset, so exposure is handled and more views are never mistaken for better conversion). Inputs: price relative to the market median (quadratic), rating shown, reviews so far, festival weeks.
- **A2 Visibility:** weekly impressions (Poisson). Inputs: launch phase, recent sales vs the category norm, reviews, rating, category demand, sale weeks.
- **B Keep share:** RTO rate; return rate (logistic, rises with price above market, lower for better listings); spread of return rates between products; resale share; review rate.
- **Product scenarios:** a good product (top quarter) gets a better rating, a higher baseline in A1 and A2, and fewer returns; a weak one the opposite.

## Key results (synthetic data)

- +10% price: about -20% orders per impression for soap, -26% for t-shirts. Comparing different listings suggests only -7% and -10%.
- Selling at twice the category norm lifts next week's impressions by about 35%; new listings are shown about 1.8x at first.
- Backtest, 8-week orders on held-out listings: median error 31% (soap) and 41% (t-shirt), against 153% and 87% for a simple average.
- Checked against the simulator (Seller 6, 26 weeks, change vs starting at Rs 209):

| Plan | Good: model / simulator | Typical: model / simulator | Weak: model / simulator |
|---|---|---|---|
| Rs 129 for 4 weeks (below the Rs 145 floor) | -2,626 / -779 | -2,249 / -1,308 | -1,688 / -1,698 |
| Rs 169 for 4 weeks (above the floor) | +72 / +1,690 | -241 / +531 | -277 / -15 |

Both agree that launching below the floor loses money for every kind of product. The models are more cautious than the simulator about discounts above the floor: sellers in the data cut prices when sales are slow, which blurs the price effect. Only randomised price tests remove that, so the pilot includes them.

## Response to the external review

| # | Review point | Valid? | What we changed |
|---|---|---|---|
| 1 | Deck has 11 slides; brief allows 6-10 | Yes | Merged the lifecycle and funnel-signals slides. The deck is now 10 slides including the cover. |
| 2 | "Rs 380 per extra review" did not match the chart | Yes | The figure came from the typical-rating run while the chart showed good/weak runs. All launch-window figures now come from the good-product lines on the chart: week 4 is -Rs 2,175 vs +Rs 2,624, a Rs 4,799 gap for about 20 extra reviews, so about Rs 235 each. The slide shows the division. |
| 3 | 30-day rollout said the floor needs only data Meesho holds | Yes | Slide 10 now says the seller types making cost, packaging and gifts; Meesho fills fees, return and RTO rates and GST. Slide 3 adds a "Comes from" column; slide 8 lists who provides what. |
| 4 | Slide 4's 12-week profit needed a clear denominator | Yes | Now "about 69 orders placed, 58 of them paid; 58 x Rs 61 = Rs 3,537". The app says the same. |
| 5 | Slide 10 (now 9) did not state exposure treatment | Yes | A1 is orders per impression with a log-impressions offset; stated on the slide, in the app and in the PDF. |
| 6 | Competitor claims need traceable sources; "no tool" overclaims | Yes | Each tool row has a reference number; a numbered list with links is on the slide and full URLs are in the notes. The claim is narrowed to "none we found models full seller cost", and the gap box says three tools use cost partly. |
| 7 | The 15% willingness-to-pay premium should be framed as a demo finding | Yes | Labelled "in our demo data" on the slide, in the app and in the notes. After retraining, the figure is about 8%. |

While checking, we also fixed:
- The visibility model had the same quality trap as conversion. Fitted within listings, the effect of recent sales went from 0.007 to 0.30.
- "Good" and "weak" products now differ in conversion, visibility and returns, not only in rating.
- The launch-lower verdict was tied to the loss budget, so an empty budget always meant "not worth it". Worth-it and affordability are now separate.
- The app now shows a gentler option (a discount that stays just above the floor), which buys reviews for about Rs 147 each and never goes below zero.
