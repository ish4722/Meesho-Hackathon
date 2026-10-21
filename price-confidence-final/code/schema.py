"""
Price Confidence - dataset schema, column roles and the model feature whitelist.

Every column of every table has exactly one ROLE. The role decides whether, and with what lag,
a column may feed a model:

  id, key            identifiers and time keys
  meta               descriptive text, never a feature
  observed_static    listing facts known at launch (category, tier, listing quality, ...)
  decision           set by the seller before the week starts (price, discount, delivery promise)
  calendar           published in advance (festival / campaign weeks; synthetic calendar)
  condition          how the week turned out for the listing (stock availability); known only afterwards
  market             platform-wide prices and demand for the week; known only afterwards
  outcome            what happened in the week (impressions ... reviews)
  state_after        running totals at the END of the week (cumulative reviews / rating)
  future_derived     product summaries computed from the listing's whole history
  seller_private     the seller's own costs and loss budget: an INPUT the seller types, never a model feature
  accounting         money computed from outcomes and seller-private costs
  latent             generator-only truth (hidden_*): never available outside the generator and its diagnostics

Temporal rule: a feature for week t may use decision / calendar / observed_static columns of week t,
and outcome / state_after / market / condition columns only from weeks < t.
"""

SCHEMA_VERSION = "2.0"

ALL_ROLES = {"id", "key", "meta", "observed_static", "decision", "calendar", "condition", "market", "outcome",
             "state_after", "future_derived", "seller_private", "accounting", "latent"}
SAME_WEEK_OK = {"key", "observed_static", "decision", "calendar"}           # usable at lag 0
LAGGED_OK = SAME_WEEK_OK | {"condition", "market", "outcome", "state_after"}  # usable at lag >= 1
NEVER_FEATURE = {"meta", "future_derived", "seller_private", "accounting", "latent", "id"}

PRODUCTS = {
    "product_id": ("id", "Listing identifier"),
    "category": ("observed_static", "Category key (soap, tshirt, bedsheet, earrings)"),
    "subcategory": ("observed_static", "Subcategory key; sets which competitor median the price is compared with"),
    "positioning_tier": ("observed_static", "Seller-declared positioning: basic, standard or premium"),
    "attribute_name": ("meta", "Name of the variant attribute for this category"),
    "attribute_value": ("observed_static", "Variant attribute value (pack size, fabric, thread count, material)"),
    "title": ("meta", "Listing title"),
    "listing_quality_score": ("observed_static", "0-100 score for photos, description and size chart (observable)"),
    "delivery_promise_days_base": ("observed_static", "Usual delivery promise in days"),
    "launch_week": ("observed_static", "First week the listing was live"),
    "launch_price": ("observed_static", "Listing price in launch week"),
    "platform_median_at_launch": ("observed_static", "Subcategory competitor median in the week before launch"),
    "launch_vs_median": ("observed_static", "launch_price / platform_median_at_launch"),
    "launch_position": ("observed_static", "below / at / above the median at launch (6% bands)"),
    "current_listing_price": ("future_derived", "Listing price in the last observed week"),
    "status": ("future_derived", "active or delisted at the end of the data"),
    "delist_week": ("future_derived", "Week number in which the seller delisted, if any"),
    "weeks_live": ("future_derived", "Number of observed weeks"),
    "total_orders": ("future_derived", "Orders over the whole observed life"),
    "total_kept": ("future_derived", "Kept orders over the whole observed life"),
    "total_returns": ("future_derived", "Customer returns over the whole observed life"),
    "total_rto": ("future_derived", "RTO parcels over the whole observed life"),
    "review_count": ("future_derived", "Reviews at the end of the data"),
    "final_rating": ("future_derived", "Average stars at the end of the data"),
    "seller_unit_cost_inr": ("seller_private", "Making or buying cost per unit"),
    "seller_packaging_inr": ("seller_private", "Packaging cost per parcel"),
    "seller_free_gift_inr": ("seller_private", "Free gift per parcel (0 if none)"),
    "seller_loss_budget_inr": ("seller_private", "Most the seller is willing to lose while building reviews"),
    "hidden_true_quality": ("latent", "Latent product quality (generator only)"),
    "hidden_archetype": ("latent", "Price-response archetype (generator only)"),
    "hidden_resp_beta": ("latent", "Linear price sensitivity"),
    "hidden_resp_kappa_hi": ("latent", "Curvature above the fair price"),
    "hidden_resp_kappa_lo": ("latent", "'Too cheap' penalty strength"),
    "hidden_resp_theta": ("latent", "Discount depth at which the 'too cheap' penalty starts"),
    "hidden_resp_sig": ("latent", "Price-as-quality-signal amplitude"),
    "hidden_resp_center": ("latent", "Relative price where the signal peaks"),
    "hidden_resp_width": ("latent", "Width of the signal bump"),
    "hidden_fair_mult": ("latent", "Fair price / subcategory median"),
    "hidden_ret_mult": ("latent", "Attribute effect on returns"),
    "hidden_demand_mult": ("latent", "Product-specific baseline demand multiplier"),
    "hidden_conv_mult": ("latent", "Product-specific baseline conversion multiplier"),
    "hidden_stock_risk": ("latent", "Weekly probability of a partial stock-out"),
    "hidden_cod_share": ("latent", "Share of orders paid cash on delivery"),
    "hidden_rto_mult": ("latent", "Product-specific RTO propensity multiplier"),
    "hidden_seller_strategy": ("latent", "Seller pricing strategy"),
    "hidden_cumulative_contribution_inr": ("latent", "Contribution over the observed life (uses seller-private costs)"),
}

WEEKLY_BASE = {
    "product_id": ("id", "Listing identifier"),
    "category": ("observed_static", "Category key"),
    "subcategory": ("observed_static", "Subcategory key"),
    "week_start": ("key", "Monday of the week"),
    "week_num": ("key", "Weeks since launch, starting at 1"),
    "listing_price": ("decision", "Listing price before any campaign discount"),
    "discount_pct": ("decision", "Campaign discount taken this week"),
    "selling_price": ("decision", "Price buyers paid this week"),
    "price_change_reason": ("decision", "Why the price differs from last week (none if unchanged)"),
    "festival_week": ("calendar", "Synthetic festival week flag"),
    "campaign_week": ("calendar", "Synthetic sale-campaign week flag"),
    "delivery_promise_days": ("decision", "Delivery promise shown this week"),
    "in_stock_share": ("condition", "Share of the week the listing was in stock"),
    "impressions": ("outcome", "Times shown in feeds and search"),
    "clicks": ("outcome", "Product-page views"),
    "add_to_cart": ("outcome", "Add-to-cart events"),
    "orders_placed": ("outcome", "Orders placed"),
    "cancelled_orders": ("outcome", "Orders cancelled before dispatch"),
    "dispatched_orders": ("outcome", "Orders shipped"),
    "cod_orders": ("outcome", "Dispatched orders paid cash on delivery"),
    "rto_orders": ("outcome", "Dispatched parcels returned undelivered"),
    "delivered_orders": ("outcome", "Parcels delivered"),
    "customer_returns": ("outcome", "Delivered parcels returned by the buyer"),
    "returns_resold": ("outcome", "Returned units put back in stock"),
    "kept_orders": ("outcome", "Delivered and not returned"),
    "new_reviews": ("outcome", "Reviews received this week"),
    "new_reviews_avg_stars": ("outcome", "Average stars of this week's reviews"),
    "positive_reviews": ("outcome", "4-5 star reviews"),
    "neutral_reviews": ("outcome", "3 star reviews"),
    "negative_reviews": ("outcome", "1-2 star reviews"),
    "neg_price_mentions": ("outcome", "Negative reviews about price"),
    "neg_quality_mentions": ("outcome", "Negative reviews about quality"),
    "cumulative_reviews": ("state_after", "Reviews to date, at the END of the week"),
    "cumulative_rating": ("state_after", "Average stars to date, at the END of the week"),
    "seller_contribution_inr": ("accounting", "This week's contribution with the seller's own costs"),
}
RETURN_REASONS = ["size_issue", "quality_not_as_expected", "not_as_pictured", "damaged_in_transit", "wrong_item",
                  "changed_mind"]
WEEKLY = dict(WEEKLY_BASE)
for _r in RETURN_REASONS:
    WEEKLY[f"return_{_r}"] = ("outcome", f"Returns giving the reason '{_r}'")

MARKET = {
    "category": ("key", "Category key"),
    "subcategory": ("key", "Subcategory key"),
    "week_start": ("key", "Monday of the week"),
    "platform_median_price": ("market", "Median price of comparable live listings (synthetic stand-in for public listing prices)"),
    "platform_p25_price": ("market", "25th percentile price of comparable listings"),
    "platform_p75_price": ("market", "75th percentile price of comparable listings"),
    "category_demand_index": ("market", "Category demand relative to a normal week (realised)"),
    "festival_week": ("calendar", "Synthetic festival week flag"),
    "campaign_week": ("calendar", "Synthetic sale-campaign week flag"),
}

REVIEWS = {
    "review_id": ("id", "Review identifier"),
    "product_id": ("id", "Listing identifier"),
    "category": ("observed_static", "Category key"),
    "week_start": ("key", "Week the review arrived"),
    "stars": ("outcome", "1-5 stars"),
    "sentiment": ("outcome", "positive / neutral / negative"),
    "theme": ("outcome", "praise / mixed / price / quality"),
    "comment": ("outcome", "Review text"),
    "from_returned_order": ("outcome", "1 if the reviewer returned the order"),
}

TABLES = {"products": PRODUCTS, "weekly_panel": WEEKLY, "market_weekly": MARKET, "reviews": REVIEWS}


def role(table, column):
    spec = TABLES[table]
    if column not in spec:
        raise KeyError(f"{table}.{column} has no declared role in schema.py")
    return spec[column][0]


def check_columns(table, columns):
    """Every column must be declared, and every declared column present."""
    spec = TABLES[table]
    unknown = [c for c in columns if c not in spec]
    missing = [c for c in spec if c not in columns]
    return unknown, missing


# ---------------------------------------------------------------- engineered model features
# Each model feature is built in features.py from these source columns at the stated lag.
# lag 0 = the same week; lag 1 = values from earlier weeks only.
FEATURES = {
    "rel":            dict(sources=[("weekly_panel", "selling_price", 0), ("market_weekly", "platform_median_price", 1)],
                           doc="selling price / last week's subcategory median - 1"),
    "rel2":           dict(sources=[("weekly_panel", "selling_price", 0), ("market_weekly", "platform_median_price", 1)],
                           doc="rel squared"),
    "rel_pos":        dict(sources=[("weekly_panel", "selling_price", 0), ("market_weekly", "platform_median_price", 1)],
                           doc="max(rel, 0)"),
    "rel_x_premium":  dict(sources=[("weekly_panel", "selling_price", 0), ("market_weekly", "platform_median_price", 1),
                                    ("products", "positioning_tier", 0)], doc="rel for premium-tier listings, else 0"),
    "rel_x_basic":    dict(sources=[("weekly_panel", "selling_price", 0), ("market_weekly", "platform_median_price", 1),
                                    ("products", "positioning_tier", 0)], doc="rel for basic-tier listings, else 0"),
    "discount_pct":   dict(sources=[("weekly_panel", "discount_pct", 0)], doc="campaign discount this week"),
    "festival_week":  dict(sources=[("weekly_panel", "festival_week", 0)], doc="calendar flag"),
    "campaign_week":  dict(sources=[("weekly_panel", "campaign_week", 0)], doc="calendar flag"),
    "delivery_days_c": dict(sources=[("weekly_panel", "delivery_promise_days", 0)], doc="delivery promise - 4 days"),
    "rating_c":       dict(sources=[("weekly_panel", "cumulative_rating", 1), ("weekly_panel", "cumulative_reviews", 1)],
                           doc="rating shown at the start of the week (shrunk towards the training prior) - 4"),
    "logrev":         dict(sources=[("weekly_panel", "cumulative_reviews", 1)], doc="log(1 + reviews before the week)"),
    "convratio":      dict(sources=[("weekly_panel", "orders_placed", 1), ("weekly_panel", "impressions", 1)],
                           doc="EMA of past weeks' orders per impression / training norm - 1"),
    "log_di_lag":     dict(sources=[("market_weekly", "category_demand_index", 1)], doc="log of last week's demand index"),
    "ph12":           dict(sources=[("weekly_panel", "week_num", 0)], doc="launch weeks 1-2"),
    "ph34":           dict(sources=[("weekly_panel", "week_num", 0)], doc="launch weeks 3-4"),
    "lq":             dict(sources=[("products", "listing_quality_score", 0)], doc="standardised listing quality"),
}

# Exposures enter a model as offsets, not as predictors. They are realised quantities of the same week, used
# only to fit a CONDITIONAL model; at forecast time they come from another model or a plan.
EXPOSURES = {
    "conversion": dict(column=("weekly_panel", "impressions", 0),
                       why="orders per impression; at forecast time impressions come from the visibility model"),
    "visibility": dict(column=("weekly_panel", "in_stock_share", 0),
                       why="impressions are only served while in stock; forecasts assume the seller's planned availability"),
    "returns": dict(column=("weekly_panel", "delivered_orders", 0), why="binomial denominator"),
    "rto": dict(column=("weekly_panel", "dispatched_orders", 0), why="binomial denominator"),
    "cancel": dict(column=("weekly_panel", "orders_placed", 0), why="binomial denominator"),
}

MODEL_FEATURES = {
    "conversion": ["rel", "rel2", "rel_x_premium", "rel_x_basic", "discount_pct", "festival_week", "campaign_week",
                   "delivery_days_c", "rating_c", "logrev"],
    "visibility": ["ph12", "ph34", "logrev", "rating_c", "convratio", "log_di_lag", "campaign_week", "delivery_days_c"],
    "returns": ["rel_pos", "delivery_days_c"],
    "rto": ["rel_pos", "delivery_days_c"],
    "cancel": ["rel_pos"],
    "stars": ["rel_pos"],
}
SECOND_STAGE = ["lq", "positioning_tier", "subcategory", "attribute_value"]    # launch-time product baseline


class LeakageError(AssertionError):
    pass


def assert_features_allowed(model, columns):
    """Raise LeakageError if any column is not whitelisted for `model` or breaks the temporal rule."""
    allowed = set(MODEL_FEATURES[model])
    bad = [c for c in columns if c not in allowed]
    if bad:
        raise LeakageError(f"{model}: columns not in the feature whitelist: {bad}")
    undeclared = [c for c in columns if c not in FEATURES]
    if undeclared:
        raise LeakageError(f"{model}: features with no declared source columns and lags: {undeclared}")
    for c in columns:
        for table, col, lag in FEATURES[c]["sources"]:
            r = role(table, col)
            if r in NEVER_FEATURE:
                raise LeakageError(f"{model}.{c} draws on {table}.{col} with role '{r}', which may never be a feature")
            if lag == 0 and r not in SAME_WEEK_OK:
                raise LeakageError(f"{model}.{c} uses {table}.{col} ({r}) from the same week")
            if lag >= 1 and r not in LAGGED_OK:
                raise LeakageError(f"{model}.{c} uses {table}.{col} with role '{r}'")
    return True


def forbidden_columns():
    """Columns that must never appear in a design matrix."""
    out = set()
    for spec in TABLES.values():
        out |= {c for c, (r, _) in spec.items() if r in NEVER_FEATURE - {"id"}}
    return out
