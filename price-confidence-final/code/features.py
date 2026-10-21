"""
Price Confidence - model features with the temporal rule enforced.

build_frame() turns the weekly panel into one row per listing-week with every engineered feature in
schema.FEATURES. State features (reviews, rating, recent conversion, market median, demand) come only
from EARLIER weeks; same-week inputs are limited to seller decisions, the calendar and launch-time
listing facts. design() returns a model's design matrix and refuses any column outside the whitelist.

Constants that summarise the data (rating prior, conversion norm, listing-quality scaling) are computed
by fit_constants() from TRAINING rows only and passed in, so evaluation splits never borrow from held-out
listings or from the future.
"""
import numpy as np
import pandas as pd

import schema

SHRINK_M = 20            # reviews of prior weight when showing a rating
EMA_KEEP = 0.6           # weight on the previous EMA of conversion


def fit_constants(weekly, products):
    """Training-only constants. `weekly`/`products` must already be restricted to training data."""
    last = weekly.sort_values("week_num").groupby("product_id").tail(1)
    settled = last[last.cumulative_reviews >= 5]
    prior = float(settled.cumulative_rating.median()) if len(settled) else 3.8
    conv_ref = float(weekly.orders_placed.sum() / max(1, weekly.impressions.sum()))
    lq = products.listing_quality_score
    q = settled.cumulative_rating.quantile([0.25, 0.5, 0.75]) if len(settled) >= 8 else pd.Series({0.25: prior - 0.3, 0.5: prior, 0.75: prior + 0.3})
    return dict(prior_rating=prior, conv_ref=conv_ref, lq_mean=float(lq.mean()), lq_sd=float(lq.std(ddof=1) or 1.0),
                rating_weak=float(q[0.25]), rating_expected=float(q[0.5]), rating_good=float(q[0.75]))


def lagged_market(market):
    """Market values shifted by one week within each (category, subcategory): what was known at the start of a week."""
    m = market.sort_values(["category", "subcategory", "week_start"]).copy()
    g = m.groupby(["category", "subcategory"])
    out = m[["category", "subcategory", "week_start"]].copy()
    for col in ["platform_median_price", "platform_p25_price", "platform_p75_price", "category_demand_index"]:
        out[col + "_lag1"] = g[col].shift(1)
    return out


def build_frame(weekly, products, market, k):
    """One row per listing-week with engineered features. `k` = fit_constants(...) from training data."""
    w = weekly.merge(products[["product_id", "positioning_tier", "attribute_value", "listing_quality_score"]],
                     on="product_id", how="left", validate="many_to_one")
    w = w.merge(lagged_market(market), on=["category", "subcategory", "week_start"], how="left", validate="many_to_one")
    w = w.sort_values(["product_id", "week_num"]).reset_index(drop=True)
    g = w.groupby("product_id", sort=False)

    rev_prev = g.cumulative_reviews.shift(1).fillna(0)
    rat_prev = g.cumulative_rating.shift(1)
    shown = (rat_prev.fillna(k["prior_rating"]) * rev_prev + k["prior_rating"] * SHRINK_M) / (rev_prev + SHRINK_M)
    w["rating_c"] = shown - 4.0
    w["logrev"] = np.log1p(rev_prev)

    conv = (w.orders_placed / w.impressions.where(w.impressions > 0)).fillna(0.0)
    ema_prev = np.full(len(w), np.nan)
    pid = w.product_id.values
    e = None
    for i in range(len(w)):
        if i == 0 or pid[i] != pid[i - 1]:
            e = None
        ema_prev[i] = np.nan if e is None else e
        e = conv.iat[i] if e is None else EMA_KEEP * e + (1 - EMA_KEEP) * conv.iat[i]
    w["convratio"] = (pd.Series(ema_prev, index=w.index).fillna(k["conv_ref"]) / k["conv_ref"] - 1).clip(-1, 3)

    w["rel"] = w.selling_price / w.platform_median_price_lag1 - 1
    w["rel2"] = w.rel ** 2
    w["rel_pos"] = w.rel.clip(lower=0)
    w["rel_x_premium"] = np.where(w.positioning_tier == "premium", w.rel, 0.0)
    w["rel_x_basic"] = np.where(w.positioning_tier == "basic", w.rel, 0.0)
    w["delivery_days_c"] = w.delivery_promise_days - 4
    w["log_di_lag"] = np.log(w.category_demand_index_lag1)
    w["ph12"] = (w.week_num <= 2).astype(float)
    w["ph34"] = w.week_num.between(3, 4).astype(float)
    w["lq"] = (w.listing_quality_score - k["lq_mean"]) / k["lq_sd"]
    for c in ["discount_pct", "festival_week", "campaign_week"]:
        w[c] = w[c].astype(float)
    return w


def design(frame, model):
    """Design matrix for `model`, whitelisted. Raises schema.LeakageError on anything else."""
    cols = schema.MODEL_FEATURES[model]
    schema.assert_features_allowed(model, cols)
    bad = schema.forbidden_columns() & set(cols)
    if bad:
        raise schema.LeakageError(f"forbidden columns in design matrix: {sorted(bad)}")
    X = frame[cols].astype(float)
    if X.isna().any().any():
        raise ValueError(f"{model}: missing feature values in {X.columns[X.isna().any()].tolist()}")
    return X


def second_stage_matrix(prod, levels):
    """Launch-time product baseline design: listing quality + tier + subcategory + attribute dummies.

    `levels` fixes the dummy coding (reference = first level) so training and inference agree.
    """
    X = pd.DataFrame(index=prod.index)
    X["const"] = 1.0
    X["lq"] = prod["lq"].astype(float)
    for name, col in (("tier", "positioning_tier"), ("sub", "subcategory"), ("attr", "attribute_value")):
        for lv in levels[name][1:]:
            X[f"{name}:{lv}"] = (prod[col].astype(str) == str(lv)).astype(float)
    return X


def levels_for(products):
    tiers = [t for t in ["standard", "basic", "premium"] if t in set(products.positioning_tier)]
    subs = sorted(products.subcategory.unique().tolist())
    attrs = sorted(products.attribute_value.astype(str).unique().tolist())
    return dict(tier=tiers, sub=subs, attr=attrs)
