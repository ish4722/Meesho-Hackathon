"""
Price Confidence - model training (Team Edge Case, Meesho DICE S3)

Reads the synthetic dataset and fits, per category:

  Model A1  Conversion   orders per impression as a function of price vs market,
                         rating, review count, discount and festival weeks.
                         Fitted WITHIN each listing (listing fixed effects), so the
                         price effect comes from price changes on the same listing,
                         not from comparing good products with bad ones.
  Model A2  Visibility   impressions per week as a function of launch phase,
                         reviews, rating, last week's conversion and category demand.
                         This is the "early sales -> reviews -> visibility" loop.
  Model B   Keep share   RTO rate, customer return rate (by price position and
                         listing quality), share of returns resold, review rate.

It also summarises launch cohorts, the current market, what reviews complain
about at each price level, and a hold-out backtest. Everything is written to
model.json, which the web app reads. The app does the seller-specific maths
(costs, floor, 12-week simulation) in the browser.

All data is synthetic: these models can only rediscover the generator's
assumptions. They demonstrate the method.

Run:  python train_models.py
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

RNG = np.random.default_rng(7)
GST = 0.05
SHIP = {"soap": 30, "tshirt": 70}
SHRINK_M = 20            # rating shrinkage strength (reviews)
HOLDOUT_SHARE = 0.2

products = pd.read_csv("products.csv", parse_dates=["launch_week"])
weekly = pd.read_csv("weekly_panel.csv", parse_dates=["week_start"])
market = pd.read_csv("market_weekly.csv", parse_dates=["week_start"])
reviews = pd.read_csv("reviews.csv", parse_dates=["week_start"])
LAST = weekly.week_start.max()


def prep(cat):
    w = weekly[weekly.category == cat].merge(
        market[market.category == cat][["week_start", "platform_median_price", "category_demand_index"]],
        on="week_start").merge(products[["product_id", "listing_quality_score"]], on="product_id")
    w = w.sort_values(["product_id", "week_num"]).reset_index(drop=True)
    p = products[products.category == cat]
    # prior rating = review-weighted category mean
    prior = float(p[p.review_count >= 5].final_rating.median())    # typical rating (5+ reviews: less survivor bias)
    lq_mean, lq_sd = float(p.listing_quality_score.mean()), float(p.listing_quality_score.std())
    g = w.groupby("product_id")
    w["rev_prev"] = g.cumulative_reviews.shift(1).fillna(0)
    w["rat_prev"] = g.cumulative_rating.shift(1)
    w["rating_shrunk"] = ((w.rat_prev.fillna(prior) * w.rev_prev + prior * SHRINK_M) / (w.rev_prev + SHRINK_M))
    w["rating_c"] = w.rating_shrunk - 4.0
    w["logrev"] = np.log1p(w.rev_prev)
    w["lp"] = np.log(w.selling_price / w.platform_median_price)
    w["lp2"] = w.lp ** 2
    w["rel"] = w.selling_price / w.platform_median_price - 1      # price vs market, linear scale
    w["rel2"] = w.rel ** 2
    w["lp_pos"] = w.lp.clip(lower=0)
    w["lq"] = (w.listing_quality_score - lq_mean) / lq_sd
    w["log_di"] = np.log(w.category_demand_index)
    w["ph12"] = (w.week_num <= 2).astype(float)
    w["ph34"] = w.week_num.between(3, 4).astype(float)
    conv_ref = float(w.orders_placed.sum() / w.impressions.sum())
    w["conv"] = w.orders_placed / w.impressions.replace(0, np.nan)
    # recent-conversion signal: exponential average of past weeks' conversion vs category norm
    ema = []
    for _, t in w.groupby("product_id", sort=False):
        e = None
        for cv in t.conv.fillna(0).values:
            ema.append(np.nan if e is None else e)
            e = cv if e is None else 0.6 * e + 0.4 * cv
    w["convratio"] = (pd.Series(ema, index=w.index).fillna(conv_ref) / conv_ref - 1).clip(-1, 3)
    return w, prior, lq_mean, lq_sd, conv_ref


PRICE_FORM = "rel"          # "rel" (price/median - 1, quadratic) or "lp" (log price ratio, quadratic); chosen by backtest
PX = ["rel", "rel2"] if PRICE_FORM == "rel" else ["lp", "lp2"]
import os
USE_DISC = os.environ.get("USE_DISC", "0") == "1"   # off: a separate badge term soaked up part of the price effect
CONV_X = PX + ["rating_c", "logrev"] + (["discount_pct"] if USE_DISC else []) + ["festival_week"]
IMPR_X = ["ph12", "ph34", "logrev", "rating_c", "convratio", "log_di", "campaign_week"]


def fit_conversion(w):
    """Poisson GLM with listing fixed effects and log(impressions) offset."""
    d = w[w.impressions > 0]
    fe = pd.get_dummies(d.product_id, prefix="fe", drop_first=False, dtype=float)
    X = pd.concat([d[CONV_X].astype(float), fe], axis=1)
    m = sm.GLM(d.orders_placed, X, family=sm.families.Poisson(), offset=np.log(d.impressions)).fit()
    coefs = {k: float(m.params[k]) for k in CONV_X}
    fes = m.params[[c for c in X.columns if c.startswith("fe_")]]
    fes.index = [c[3:] for c in fes.index]
    # second stage: listing effect explained by listing quality (what we can see before launch)
    lq = d.groupby("product_id").lq.first().reindex(fes.index)
    wts = d.groupby("product_id").orders_placed.sum().reindex(fes.index).clip(lower=1)
    # level: every listing counts equally (a "typical" listing, not the best sellers);
    # spread: precision-weighted, so noisy small listings don't exaggerate how different products are
    s2 = sm.OLS(fes.values, sm.add_constant(lq.values)).fit()
    s2w = sm.WLS(fes.values, sm.add_constant(lq.values), weights=np.sqrt(wts.values)).fit()
    coefs["a"] = float(s2.params[0])
    coefs["b_lq"] = float(s2.params[1])
    coefs["fe_sd"] = float(np.sqrt(np.average((fes.values - s2w.fittedvalues) ** 2, weights=np.sqrt(wts.values))))
    return coefs


def naive_price_slope(w):
    """What you'd conclude comparing listings with each other (no fixed effects)."""
    d = w[w.impressions > 0]
    X = sm.add_constant(d[["lp", "discount_pct", "festival_week"]].astype(float))
    m = sm.GLM(d.orders_placed, X, family=sm.families.Poisson(), offset=np.log(d.impressions)).fit()
    return float(m.params["lp"])


def fit_visibility(w):
    """Poisson GLM with listing fixed effects: dynamics (reviews, rating, recent sales) learnt within each listing,
    then a second stage puts the listing baselines on listing quality, so a new listing gets a starting level."""
    d = w
    fe = pd.get_dummies(d.product_id, prefix="fe", dtype=float)
    m = sm.GLM(d.impressions, pd.concat([d[IMPR_X].astype(float), fe], axis=1), family=sm.families.Poisson()).fit()
    c = {k: float(m.params[k]) for k in IMPR_X}
    fes = m.params[[x for x in m.params.index if x.startswith("fe_")]]
    fes.index = [x[3:] for x in fes.index]
    lq = d.groupby("product_id").lq.first().reindex(fes.index)
    wts = d.groupby("product_id").impressions.sum().reindex(fes.index).clip(lower=1)
    s2 = sm.OLS(fes.values, sm.add_constant(lq.values)).fit()
    s2w = sm.WLS(fes.values, sm.add_constant(lq.values), weights=np.sqrt(wts.values)).fit()
    c["intercept"] = float(s2.params[0])
    c["lq"] = float(s2.params[1])
    c["fe_sd"] = float(np.sqrt(np.average((fes.values - s2w.fittedvalues) ** 2, weights=np.sqrt(wts.values))))
    return c


def fit_keep(w):
    """Return rate: within-listing effect of price (fixed effects), baseline from listing quality."""
    d = w[w.delivered_orders > 0]
    fe = pd.get_dummies(d.product_id, prefix="fe", dtype=float)
    X = pd.concat([d[["lp_pos"]].astype(float), fe], axis=1)
    y = np.column_stack([d.customer_returns, d.delivered_orders - d.customer_returns])
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    lp_pos = max(0.0, float(m.params["lp_pos"]))          # never let price reduce returns
    # spread of return rates between products: listings with enough deliveries, binomial noise removed
    agg = d.groupby("product_id")[["customer_returns", "delivered_orders"]].sum()
    agg = agg[agg.delivered_orders >= 100]
    rr = ((agg.customer_returns + 0.5) / (agg.delivered_orders + 1)).values
    lg = np.log(rr / (1 - rr))
    noise = np.mean(1 / (agg.delivered_orders.values * rr * (1 - rr)))
    ret_fe_sd = float(np.sqrt(max(np.var(lg) - noise, 0.0)))
    # baseline: category return rate at market price, by listing quality
    base = sm.GLM(y, sm.add_constant(d[["lq"]].astype(float)), family=sm.families.Binomial(),
                  offset=lp_pos * d.lp_pos).fit()
    tot = w.sum(numeric_only=True)
    return dict(
        rto=float(tot.rto_orders / tot.orders_placed),
        ret_intercept=float(base.params["const"]), ret_lp_pos=lp_pos, ret_lq=float(base.params["lq"]),
        resale=float(tot.returns_resold / tot.customer_returns) if tot.customer_returns else 0.0,
        review_rate=float(tot.new_reviews / (tot.kept_orders + tot.customer_returns)),
        ret_fe_sd=ret_fe_sd,
    )


def predict_week(cc, ic, row_like):
    """One-step prediction of orders for a week (used by the backtest)."""
    eta_c = (cc["a"] + cc["b_lq"] * row_like["lq"] + sum(cc[k] * row_like[k] for k in CONV_X))
    eta_i = ic["intercept"] + ic["lq"] * row_like["lq"] + sum(ic[k] * row_like[k] for k in IMPR_X)
    return float(np.exp(eta_i) * np.exp(eta_c))


def backtest(w):
    ids = w.product_id.unique()
    hold = set(RNG.choice(ids, size=int(len(ids) * HOLDOUT_SHARE), replace=False))
    train, test = w[~w.product_id.isin(hold)], w[w.product_id.isin(hold)]
    cc, ic = fit_conversion(train), fit_visibility(train)
    naive_by_week = train[train.week_num <= 8].groupby("week_num").orders_placed.mean()
    errs_m, errs_n, n = [], [], 0
    for pid, t in test.groupby("product_id"):
        t = t[t.week_num <= 8]
        if len(t) < 8:
            continue
        pred = sum(predict_week(cc, ic, r) for _, r in t.iterrows())
        naive = float(naive_by_week.reindex(t.week_num).sum())
        actual = float(t.orders_placed.sum())
        if actual < 5:
            continue
        errs_m.append(abs(pred - actual) / actual)
        errs_n.append(abs(naive - actual) / actual)
        n += 1
    return dict(n_listings=n, model_median_abs_pct_error=float(np.median(errs_m)),
                naive_median_abs_pct_error=float(np.median(errs_n)))


def cohorts(w, cat):
    p = products[products.category == cat].copy()
    first12 = w[w.week_num <= 12].copy()
    first12["kept_x_price"] = first12.kept_orders * first12.selling_price
    agg = first12.groupby("product_id").agg(
        kept_x_price=("kept_x_price", "sum"), orders=("orders_placed", "sum"), returns=("customer_returns", "sum"),
        kept=("kept_orders", "sum"), resold=("returns_resold", "sum"), weeks=("week_num", "max"),
        reviews12=("new_reviews", "sum"))
    p = p.merge(agg, left_on="product_id", right_index=True)
    p["eligible"] = p.launch_week <= (LAST - pd.Timedelta(weeks=11))
    out = []
    for r in p.itertuples():
        out.append(dict(id=r.product_id, pos=r.launch_position, lvm=round(r.launch_vs_median, 3),
                        eligible=bool(r.eligible), survived12=bool(r.eligible and r.weeks >= 12),
                        rating=None if pd.isna(r.final_rating) else round(r.final_rating, 2),
                        reviews=int(r.review_count), kxp=round(r.kept_x_price), orders=int(r.orders),
                        returns=int(r.returns), kept=int(r.kept), resold=int(r.resold)))
    return out


def complaints(cat):
    rv = reviews[(reviews.category == cat) & (reviews.sentiment == "negative")].merge(
        weekly[["product_id", "week_start", "selling_price"]], on=["product_id", "week_start"]).merge(
        market[market.category == cat][["week_start", "platform_median_price"]], on="week_start")
    rel = rv.selling_price / rv.platform_median_price
    bins = [0, 0.95, 1.05, 1.15, 10]
    labels = ["more than 5% below median", "within 5% of median", "5-15% above median", "more than 15% above median"]
    rv["bucket"] = pd.cut(rel, bins=bins, labels=labels)
    s = rv.groupby("bucket", observed=False).theme.apply(lambda t: float((t == "price").mean()) if len(t) else None)
    n = rv.groupby("bucket", observed=False).size()
    return [dict(bucket=b, price_share=None if pd.isna(s[b]) else round(s[b], 3), n=int(n[b])) for b in labels]


def market_now(cat):
    m = market[market.category == cat].sort_values("week_start")
    last8 = m.tail(8).copy()
    age = (LAST - last8.week_start).dt.days / 7
    wts = 0.5 ** (age / 2.0)                     # 2-week half-life
    wavg = lambda col: float(np.average(last8[col], weights=wts))
    last4, prev4 = m.tail(4), m.iloc[-8:-4]
    return dict(median=round(wavg("platform_median_price")), p25=round(wavg("platform_p25_price")),
                p75=round(wavg("platform_p75_price")),
                median_prev4=round(float(prev4.platform_median_price.mean())),
                median_last4=round(float(last4.platform_median_price.mean())),
                demand_prev4=round(float(prev4.category_demand_index.mean()), 3),
                demand_last4=round(float(last4.category_demand_index.mean()), 3),
                as_of=LAST.date().isoformat())


out = dict(meta=dict(
    synthetic=True, seed_note="generate_dataset.py seed 20260930",
    listings=int(len(products)), listing_weeks=int(len(weekly)), reviews=int(len(reviews)),
    first_launch=products.launch_week.min().date().isoformat(), data_to=LAST.date().isoformat(),
    gst=GST, shrink_m=SHRINK_M, horizon_weeks=12), categories={})

for cat in ["soap", "tshirt"]:
    w, prior, lq_mean, lq_sd, conv_ref = prep(cat)
    cc = fit_conversion(w)
    ic = fit_visibility(w)
    kp = fit_keep(w)
    p = products[(products.category == cat) & (products.review_count >= 5)]
    q = p.final_rating.quantile([0.25, 0.5, 0.75])
    bt = backtest(w)
    out["categories"][cat] = dict(
        ship=SHIP[cat], conv=cc, impr=ic, conv_ref=conv_ref, keep=kp,
        rating=dict(prior=round(prior, 3), weak=round(float(q[0.25]), 2), expected=round(float(q[0.5]), 2),
                    good=round(float(q[0.75]), 2)),
        lq=dict(mean=lq_mean, sd=lq_sd),
        market=market_now(cat),
        price_form=PRICE_FORM,
        stats=dict(naive_price_coef=naive_price_slope(w), backtest=bt,
                   listings=int((products.category == cat).sum()), weeks=int(len(w))),
        complaints=complaints(cat),
        cohorts=cohorts(w, cat),
    )
    print(cat, "| price terms:", {k: round(cc[k], 2) for k in PX}, "naive lp:", round(out["categories"][cat]["stats"]["naive_price_coef"], 2),
          "| backtest", {k: round(v, 3) if isinstance(v, float) else v for k, v in bt.items()})

with open("model.json", "w") as f:
    json.dump(out, f, indent=1)
print("model.json written")
