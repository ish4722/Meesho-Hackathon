"""
Price Confidence - synthetic marketplace "world" (generator v2).

EVERY relationship in this file is an ASSUMPTION chosen by the team to build a realistic-looking,
heterogeneous test bed. None of it is estimated from Meesho data, and nothing a model learns from
this data is evidence about real Meesho buyers.

What the world contains
  * 4 categories x 3 subcategories, each listing with a positioning tier (basic / standard / premium),
    one variant attribute, listing quality, delivery promise and stock availability (all observable).
  * Latent, generator-only variables (columns prefixed hidden_ in products.csv): true quality,
    a price-response ARCHETYPE with product-specific coefficients, baseline demand and conversion
    multipliers, stock-out risk, COD share, RTO propensity and the seller's pricing strategy.
  * A funnel with a separate random draw at every stage:
        impressions ~ Poisson(lambda x lognormal week shock)       (Poisson-lognormal, overdispersed)
        clicks      ~ Binomial(impressions, CTR)
        carts       ~ Binomial(clicks, add-to-cart rate)
        orders      ~ Binomial(carts, order rate x lognormal week shock)
        cancelled   ~ Binomial(orders, cancel rate)            dispatched = orders - cancelled
        rto         ~ Binomial(COD / prepaid dispatched, RTO rate)
        returns     ~ Binomial(delivered, return rate)          resold ~ Binomial(returns, resale share)
        reviews     ~ Binomial(kept, 0.14) + Binomial(returns, 0.18); stars ~ rounded Normal
  * Sellers who change prices for recorded reasons: launch, campaign discounts (random participation),
    scheduled review-based raises, market following, reactive cuts after slow weeks, and randomised
    price experiments. The reason is written to the panel so price variation can be audited.

Price response (per product i, rel = selling price / buyer's fair price - 1):
    f_i(rel) = -beta_i * rel
               - kappa_hi_i * max(rel, 0)^2                       (penalty above fair: competition, reputation)
               - kappa_lo_i * max(-rel - theta_i, 0)^2            ("too cheap to trust")
               + sig_i * [bump(rel) - bump(0)],  bump(r) = exp(-((r - c_i) / w_i)^2)   (price as quality signal)
    log CTR += 0.35 f,  log add-to-cart += 0.35 f,  log order rate += 0.30 f
The fair price is the subcategory's current competitor median x tier, attribute and quality multipliers.
"""
import math
from datetime import date, timedelta

import numpy as np

import economics as econ

GENERATOR_VERSION = "2.0.0"

MARKET_START = date(2025, 6, 2)
LAUNCH_FIRST = date(2025, 9, 1)
LAUNCH_LAST = date(2026, 8, 3)          # every listing has >= 8 possible weeks of data
LAST_WEEK = date(2026, 9, 21)           # last complete Monday-start week

# Synthetic calendar (illustrative, not Meesho's actual sale dates)
FEST_WEEKS = {date(2025, 9, 22), date(2025, 9, 29), date(2025, 10, 6), date(2025, 10, 13),
              date(2025, 10, 20), date(2026, 3, 2), date(2026, 8, 24)}
CAMPAIGN_WEEKS = {date(2025, 9, 22), date(2025, 9, 29), date(2025, 10, 6),
                  date(2026, 1, 12), date(2026, 3, 2), date(2026, 6, 22), date(2026, 9, 21)}

REASONS = ["size_issue", "quality_not_as_expected", "not_as_pictured",
           "damaged_in_transit", "wrong_item", "changed_mind"]
TIERS = ["basic", "standard", "premium"]
TIER_FAIR = {"basic": 0.85, "standard": 1.00, "premium": 1.20}       # fair-price multiplier
TIER_Q_MEAN = {"basic": -0.25, "standard": 0.0, "premium": 0.30}       # mean latent quality

# ---------------------------------------------------------------- price-response archetypes
# Ranges are sampled uniformly per product. beta is then multiplied by the subcategory's beta_mult.
ARCHETYPES = {
    "commodity":          dict(beta=(3.4, 4.8), kappa_hi=(1.0, 3.0), kappa_lo=(0.0, 0.0), theta=(0.0, 0.0),
                               sig=(0.0, 0.0), doc="Near-identical substitutes; buyers shop on price."),
    "moderate":           dict(beta=(1.6, 2.6), kappa_hi=(0.5, 1.5), kappa_lo=(0.0, 0.0), theta=(0.0, 0.0),
                               sig=(0.0, 0.0), doc="Typical listing; demand falls with price at a moderate rate."),
    "insensitive":        dict(beta=(0.3, 0.9), kappa_hi=(0.0, 0.6), kappa_lo=(0.0, 0.0), theta=(0.0, 0.0),
                               sig=(0.0, 0.0), doc="Niche or loyal demand; price matters little within the band."),
    "premium_signal":     dict(beta=(0.9, 1.5), kappa_hi=(0.3, 1.0), kappa_lo=(0.0, 0.0), theta=(0.0, 0.0),
                               sig=(0.30, 0.50), doc="Price signals quality up to a point above fair, then hurts."),
    "too_cheap_distrust": dict(beta=(1.2, 2.0), kappa_hi=(0.5, 1.5), kappa_lo=(5.0, 9.0), theta=(0.08, 0.18),
                               sig=(0.0, 0.0), doc="Prices far below the market look fake; conversion peaks in the middle."),
    "reputation_penalty": dict(beta=(1.6, 2.6), kappa_hi=(6.0, 12.0), kappa_lo=(0.0, 0.0), theta=(0.0, 0.0),
                               sig=(0.0, 0.0), doc="Weak reputation or strong competition: above-fair prices collapse."),
}
SIGNAL_CENTER, SIGNAL_WIDTH = (0.12, 0.22), (0.15, 0.22)

# Archetype probabilities by tier. Premium listings only signal quality when quality is genuinely high.
ARCHETYPE_P = {
    "basic":            dict(commodity=0.45, moderate=0.30, insensitive=0.05, too_cheap_distrust=0.05,
                             reputation_penalty=0.15, premium_signal=0.00),
    "standard":         dict(commodity=0.20, moderate=0.45, insensitive=0.10, too_cheap_distrust=0.10,
                             reputation_penalty=0.10, premium_signal=0.05),
    "premium_highq":    dict(commodity=0.00, moderate=0.20, insensitive=0.25, too_cheap_distrust=0.00,
                             reputation_penalty=0.10, premium_signal=0.45),
    "premium_lowq":     dict(commodity=0.00, moderate=0.35, insensitive=0.10, too_cheap_distrust=0.00,
                             reputation_penalty=0.45, premium_signal=0.10),
}

# ---------------------------------------------------------------- categories
# Funnel bases (ctr0, atc0, ord0) give ~0.15% orders per impression at the fair price for a typical listing.
CATEGORIES = {
    "soap": dict(
        label="Soap", plural="soaps", group="personal_care", n=150,
        keywords=["soap", "sabun", "साबुन", "bath bar", "body bar"],
        subcats={
            "herbal":      dict(label="Herbal soap", share=0.45, median=(172, 186), beta_mult=1.00, cost_mu=82),
            "glycerin":    dict(label="Glycerin soap", share=0.30, median=(139, 149), beta_mult=1.25, cost_mu=58),
            "luxury_gift": dict(label="Luxury / gift soap", share=0.25, median=(259, 279), beta_mult=0.70, cost_mu=118),
        },
        attribute=dict(name="pack_size_g", values=[75, 100, 125, 150, 200],
                       fair_mult=[0.92, 0.96, 1.00, 1.04, 1.10], ret_mult=[1, 1, 1, 1, 1]),
        packaging=(8, 15), gift_share=0.2, gift=(10, 15), cost_sd=0.22,
        base_impr=9000, ctr0=0.045, atc0=0.09, ord0=0.36,
        ret0=0.08, ret_q=-0.5, ret_gap=0.8, rto_cod=0.14, rto_prepaid=0.02, cod_share=0.66, resale=0.0,
        season=[1.10, 1.00, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.10, 1.30, 1.20, 1.10], fest_boost=1.25,
        reasons=dict(size_issue=0.0, quality_not_as_expected=0.30, not_as_pictured=0.25,
                     damaged_in_transit=0.30, wrong_item=0.08, changed_mind=0.07),
    ),
    "tshirt": dict(
        label="T-shirt", plural="t-shirts", group="apparel", n=150,
        keywords=["t-shirt", "t shirt", "tshirt", "tee", "tees", "polo"],
        subcats={
            "cotton_basic":    dict(label="Cotton basic tee", share=0.45, median=(329, 309), beta_mult=1.00, cost_mu=120),
            "printed_graphic": dict(label="Printed / graphic tee", share=0.35, median=(299, 289), beta_mult=1.20, cost_mu=105),
            "polo":            dict(label="Polo t-shirt", share=0.20, median=(449, 469), beta_mult=0.80, cost_mu=175),
        },
        attribute=dict(name="fabric", values=["polycotton", "cotton_160gsm", "cotton_180gsm"],
                       fair_mult=[0.90, 1.00, 1.08], ret_mult=[1.20, 1.00, 0.90]),
        packaging=(8, 12), gift_share=0.0, gift=(0, 0), cost_sd=0.20,
        base_impr=12000, ctr0=0.055, atc0=0.08, ord0=0.33,
        ret0=0.26, ret_q=-0.35, ret_gap=0.6, rto_cod=0.17, rto_prepaid=0.02, cod_share=0.70, resale=0.70,
        season=[0.80, 0.90, 1.05, 1.20, 1.25, 1.15, 1.00, 0.95, 0.90, 1.05, 0.95, 0.80], fest_boost=1.15,
        reasons=dict(size_issue=0.42, quality_not_as_expected=0.22, not_as_pictured=0.16,
                     damaged_in_transit=0.06, wrong_item=0.06, changed_mind=0.08),
    ),
    "bedsheet": dict(
        label="Bedsheet", plural="bedsheets", group="home_essentials", n=150,
        keywords=["bedsheet", "bed sheet", "chadar", "चादर", "bedspread"],
        subcats={
            "single_cotton":  dict(label="Single cotton bedsheet", share=0.40, median=(349, 369), beta_mult=1.10, cost_mu=150),
            "double_cotton":  dict(label="Double cotton bedsheet", share=0.40, median=(549, 579), beta_mult=1.00, cost_mu=235),
            "double_premium": dict(label="Premium double bedsheet", share=0.20, median=(899, 949), beta_mult=0.70, cost_mu=390),
        },
        attribute=dict(name="thread_count", values=[144, 180, 210, 300],
                       fair_mult=[0.90, 0.97, 1.03, 1.12], ret_mult=[1.15, 1.0, 0.95, 0.90]),
        packaging=(15, 25), gift_share=0.0, gift=(0, 0), cost_sd=0.20,
        base_impr=7000, ctr0=0.040, atc0=0.075, ord0=0.34,
        ret0=0.12, ret_q=-0.45, ret_gap=0.7, rto_cod=0.15, rto_prepaid=0.02, cod_share=0.62, resale=0.65,
        season=[1.15, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.00, 1.05, 1.25, 1.20, 1.10], fest_boost=1.30,
        reasons=dict(size_issue=0.15, quality_not_as_expected=0.35, not_as_pictured=0.25,
                     damaged_in_transit=0.08, wrong_item=0.07, changed_mind=0.10),
    ),
    "earrings": dict(
        label="Earrings", plural="earrings", group="accessories", n=150,
        keywords=["earring", "earrings", "jhumka", "jhumki", "studs", "बाली"],
        subcats={
            "oxidised": dict(label="Oxidised jhumka", share=0.45, median=(149, 159), beta_mult=1.30, cost_mu=48),
            "kundan":   dict(label="Kundan / ethnic", share=0.30, median=(249, 269), beta_mult=1.00, cost_mu=85),
            "studs":    dict(label="Everyday studs", share=0.25, median=(119, 129), beta_mult=1.40, cost_mu=36),
        },
        attribute=dict(name="material", values=["alloy", "oxidised_brass", "silver_plated"],
                       fair_mult=[0.92, 1.00, 1.15], ret_mult=[1.15, 1.0, 0.90]),
        packaging=(5, 10), gift_share=0.0, gift=(0, 0), cost_sd=0.25,
        base_impr=11000, ctr0=0.060, atc0=0.085, ord0=0.38,
        ret0=0.10, ret_q=-0.45, ret_gap=0.7, rto_cod=0.16, rto_prepaid=0.02, cod_share=0.72, resale=0.50,
        season=[0.95, 1.05, 1.00, 1.05, 1.10, 0.95, 0.95, 1.00, 1.05, 1.30, 1.25, 1.10], fest_boost=1.30,
        reasons=dict(size_issue=0.0, quality_not_as_expected=0.30, not_as_pictured=0.35,
                     damaged_in_transit=0.15, wrong_item=0.10, changed_mind=0.10),
    ),
}

# ---------------------------------------------------------------- seller strategies
STRATEGIES = ["investment", "market", "premium", "cost_plus_stale", "reactive", "experimenter"]
STRAT_P = [0.20, 0.30, 0.15, 0.15, 0.12, 0.08]

REVIEW_TEXT = {
    "soap": dict(pos=["Lovely fragrance, skin feels soft", "Good soap, lasts long", "Nice herbal smell, will buy again",
                      "Value for money", "Gentle and nice lather"],
                 neu=["Okay soap, nothing special", "Average, smell fades quickly", "Fine for the price"],
                 neg_price=["Too costly for such a small soap", "Not worth the price", "Same soap is cheaper elsewhere"],
                 neg_qual=["Soap melted in a few days", "Fragrance not as described", "Arrived broken", "Very small, poor quality"]),
    "tshirt": dict(pos=["Fabric is soft and fits well", "Good quality for the price", "Colour same as photo",
                        "Comfortable, nice stitching", "Fits perfectly, will order again"],
                   neu=["Okay t-shirt, average fabric", "Fits fine, colour slightly dull", "Average quality"],
                   neg_price=["Overpriced for this fabric", "Not worth the price", "Cheaper tees are better than this"],
                   neg_qual=["Size runs small", "Fabric is thin and rough", "Colour faded after one wash", "Stitching came off"]),
    "bedsheet": dict(pos=["Soft cotton, colours are bright", "Fits my bed well", "Good quality for the price",
                          "Pillow covers match nicely", "Washed well, no fading"],
                     neu=["Okay bedsheet, a bit thin", "Colour lighter than photo", "Average for the price"],
                     neg_price=["Too expensive for this quality", "Not worth the price", "Same sheet is cheaper elsewhere"],
                     neg_qual=["Size smaller than mentioned", "Colour bled in first wash", "Fabric feels rough", "Stitching loose"]),
    "earrings": dict(pos=["Looks exactly like the photo", "Light and comfortable", "Beautiful design, got compliments",
                          "Good finish for the price", "Perfect for festive wear"],
                     neu=["Okay earrings, a bit small", "Finish is average", "Fine for daily use"],
                     neg_price=["Too costly for alloy earrings", "Not worth the price", "Cheaper ones look better"],
                     neg_qual=["Colour turned black in a week", "One earring was broken", "Hook is weak", "Looks cheap in person"]),
}


def mondays(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=7)


ALL_WEEKS = list(mondays(MARKET_START, LAST_WEEK))
WEEK_INDEX = {wk: i for i, wk in enumerate(ALL_WEEKS)}


def ends_in_9(x):
    return int(max(49, round(x / 10.0) * 10 - 1))


def price_response(rel, p):
    """Log-multiplier on conversion at relative price rel (vs the buyer's fair price). f(0) = 0."""
    rel = min(max(rel, -0.6), 1.0)
    bump = lambda r: math.exp(-((r - p["center"]) / p["width"]) ** 2)
    return (-p["beta"] * rel
            - p["kappa_hi"] * max(rel, 0.0) ** 2
            - p["kappa_lo"] * max(-rel - p["theta"], 0.0) ** 2
            + p["sig"] * (bump(rel) - bump(0.0)))


# Mean and variance of clip(round(Normal(mu, sd)), 1, 5): the star a single review gives.
def _star_moments(mu, sd=0.8):
    edges = [-math.inf, 1.5, 2.5, 3.5, 4.5, math.inf]
    cdf = [0.5 * (1 + math.erf((e - mu) / (sd * math.sqrt(2)))) if math.isfinite(e) else (0.0 if e < 0 else 1.0) for e in edges]
    probs = [cdf[i + 1] - cdf[i] for i in range(5)]
    m = sum((i + 1) * p for i, p in enumerate(probs))
    v = sum((i + 1) ** 2 * p for i, p in enumerate(probs)) - m * m
    return m, v


class Market:
    """Weekly category demand and subcategory competitor prices (true values)."""

    def __init__(self, rng):
        self.median, self.demand = {}, {}
        for cat, c in CATEGORIES.items():
            n = len(ALL_WEEKS)
            dem = np.empty(n)
            for i, wk in enumerate(ALL_WEEKS):
                s = c["season"][wk.month - 1] * (c["fest_boost"] if wk in FEST_WEEKS else 1.0)
                dem[i] = s * (1 + rng.normal(0, 0.03))
            self.demand[cat] = dem
            for sub, sc in c["subcats"].items():
                ar, vals = 0.0, np.empty(n)
                for i, wk in enumerate(ALL_WEEKS):
                    frac = i / (n - 1)
                    base = sc["median"][0] + (sc["median"][1] - sc["median"][0]) * frac
                    ar = 0.85 * ar + rng.normal(0, 0.012)          # competitor price noise, AR(1)
                    vals[i] = base * (1 + 0.03 * (c["season"][wk.month - 1] - 1)) * math.exp(ar)
                self.median[(cat, sub)] = vals

    def comp_median(self, cat, sub, wk):
        return float(self.median[(cat, sub)][WEEK_INDEX[wk]])

    def demand_index(self, cat, wk):
        return float(self.demand[cat][WEEK_INDEX[wk]])

    def rows(self, rng):
        out = []
        for (cat, sub), vals in self.median.items():
            for i, wk in enumerate(ALL_WEEKS):
                med = vals[i]
                out.append(dict(category=cat, subcategory=sub, week_start=wk.isoformat(),
                                platform_median_price=round(med),
                                platform_p25_price=round(med * 0.80 * (1 + rng.normal(0, 0.01))),
                                platform_p75_price=round(med * 1.22 * (1 + rng.normal(0, 0.01))),
                                category_demand_index=round(float(self.demand[cat][i]), 3),
                                festival_week=int(wk in FEST_WEEKS), campaign_week=int(wk in CAMPAIGN_WEEKS)))
        return out


def draw_product(cat, k, rng, launch):
    """Draw one listing: observable attributes, seller-private costs and generator-only latent variables."""
    c = CATEGORIES[cat]
    subs = list(c["subcats"])
    sub = subs[rng.choice(len(subs), p=[c["subcats"][s]["share"] for s in subs])]
    sc = c["subcats"][sub]
    tier = TIERS[rng.choice(3, p=[0.30, 0.50, 0.20])]
    q = float(np.clip(rng.normal(TIER_Q_MEAN[tier], 1.0), -2.5, 2.5))
    lq = float(np.clip(rng.normal(62 + 5 * q + (4 if tier == "premium" else 0), 12), 20, 98))
    ai = int(rng.integers(len(c["attribute"]["values"])))
    key = tier if tier != "premium" else ("premium_highq" if q > 0.2 else "premium_lowq")
    arch_p = ARCHETYPE_P[key]
    arch = list(arch_p)[rng.choice(len(arch_p), p=list(arch_p.values()))]
    a = ARCHETYPES[arch]
    u = lambda lo_hi: float(rng.uniform(*lo_hi)) if lo_hi[1] > lo_hi[0] else float(lo_hi[0])
    resp = dict(beta=u(a["beta"]) * sc["beta_mult"], kappa_hi=u(a["kappa_hi"]), kappa_lo=u(a["kappa_lo"]),
                theta=u(a["theta"]), sig=u(a["sig"]), center=u(SIGNAL_CENTER), width=u(SIGNAL_WIDTH))
    unit = float(sc["cost_mu"] * math.exp(c["cost_sd"] * rng.normal() + 0.06 * q)
                 * (1.15 if tier == "premium" else 0.92 if tier == "basic" else 1.0))
    gift = float(rng.uniform(*c["gift"])) if rng.random() < c["gift_share"] else 0.0
    obs = dict(
        product_id=f"{cat[:2].upper()}{k + 1:03d}", category=cat, subcategory=sub, positioning_tier=tier,
        attribute_name=c["attribute"]["name"], attribute_value=str(c["attribute"]["values"][ai]),
        listing_quality_score=round(lq), delivery_promise_days_base=int(rng.choice([2, 3, 4, 5, 6, 7],
                                                                                   p=[.1, .25, .3, .2, .1, .05])),
        launch_week=launch,
    )
    private = dict(unit=round(unit, 2), packaging=round(float(rng.uniform(*c["packaging"])), 2), gift=round(gift, 2),
                   budget=round(float(rng.uniform(2000, 8000) * (sc["median"][1] / 186) ** 0.5)))
    latent = dict(
        quality=q, archetype=arch, **{f"resp_{kk}": vv for kk, vv in resp.items()},
        fair_mult=TIER_FAIR[tier] * c["attribute"]["fair_mult"][ai] * math.exp(0.06 * q),
        ret_mult=c["attribute"]["ret_mult"][ai],
        demand_mult=float(math.exp(rng.normal(0, 0.30))),
        conv_mult=float(math.exp(0.12 * q + rng.normal(0, 0.20))),
        stock_risk=float(rng.beta(1.5, 20)), cod_share=float(np.clip(rng.normal(c["cod_share"], 0.06), 0.35, 0.92)),
        rto_mult=float(math.exp(rng.normal(0, 0.20))),
        strategy=STRATEGIES[rng.choice(len(STRATEGIES), p=STRAT_P)],
    )
    return dict(obs=obs, private=private, latent=latent)


def response_params(prod):
    L = prod["latent"]
    return dict(beta=L["resp_beta"], kappa_hi=L["resp_kappa_hi"], kappa_lo=L["resp_kappa_lo"], theta=L["resp_theta"],
                sig=L["resp_sig"], center=L["resp_center"], width=L["resp_width"])


def new_state():
    return dict(w=0, cum_reviews=0, star_sum=0.0, conv_ema=None, cum_contrib=0.0)


def fair_price(prod, market, wk):
    o = prod["obs"]
    return market.comp_median(o["category"], o["subcategory"], wk) * prod["latent"]["fair_mult"]


def step_week(prod, state, wk, sell, disc, market, rng, A, delivery_days=None, record_reviews=False):
    """Advance one listing by one week at selling price `sell`. Mutates `state`; returns (row, reviews)."""
    o, L = prod["obs"], prod["latent"]
    cat = o["category"]
    c = CATEGORIES[cat]
    state["w"] += 1
    w = state["w"]
    dd = o["delivery_promise_days_base"] if delivery_days is None else delivery_days
    lz = (o["listing_quality_score"] - 62) / 12
    q = L["quality"]
    rel = sell / fair_price(prod, market, wk) - 1
    f = price_response(rel, response_params(prod))
    n_rev, rating = state["cum_reviews"], (state["star_sum"] / state["cum_reviews"] if state["cum_reviews"] else None)
    if n_rev == 0:
        star_eff, trust = 0.85, 0.0
    else:
        star_eff = math.exp(0.35 * (rating - 4.0)) * (0.9 + 0.1 * min(1, n_rev / 30))
        trust = min(1.0, math.log1p(n_rev) / math.log1p(60)) * float(np.clip((rating - 3.0) / 1.2, 0, 1.2))
    conv_ref = c["ctr0"] * c["atc0"] * c["ord0"]
    rank = 0.55 + 0.45 * trust - 0.04 * (dd - 4)
    if state["conv_ema"] is not None:
        rank += 0.3 * (state["conv_ema"] / conv_ref - 1)
    rank = float(np.clip(rank, 0.3, 1.6))
    boost = 1.8 if w <= 2 else (1.3 if w <= 4 else 1.0)
    camp = 1.3 if wk in CAMPAIGN_WEEKS else 1.0
    in_stock = 1.0
    if rng.random() < L["stock_risk"]:
        in_stock = float(rng.uniform(0.2, 0.9))

    lam = (c["base_impr"] * market.demand_index(cat, wk) * rank * boost * camp * L["demand_mult"] * in_stock
           * rng.lognormal(0, 0.15))
    impressions = int(rng.poisson(lam))
    cm = L["conv_mult"] ** (1 / 3)
    ctr = float(np.clip(c["ctr0"] * math.exp(0.35 * f) * math.exp(0.25 * lz) * star_eff * cm
                        * math.exp(-0.03 * (dd - 4)), 0.002, 0.30))
    clicks = int(rng.binomial(impressions, ctr))
    atc_r = float(np.clip(c["atc0"] * math.exp(0.35 * f) * math.exp(0.15 * q) * (0.85 + 0.15 * trust) * cm, 0.005, 0.6))
    carts = int(rng.binomial(clicks, atc_r))
    ord_r = float(np.clip(c["ord0"] * math.exp(0.30 * f) * (1 + 0.3 * disc) * (1.05 if wk in FEST_WEEKS else 1.0) * cm
                          * rng.lognormal(0, 0.10), 0.02, 0.9))
    orders = int(rng.binomial(carts, ord_r))

    gap = max(0.0, rel)
    cod = int(rng.binomial(orders, L["cod_share"]))
    canc_cod = int(rng.binomial(cod, min(0.5, 0.06 * (1 + 0.5 * gap))))
    canc_pre = int(rng.binomial(orders - cod, min(0.5, 0.03 * (1 + 0.5 * gap))))
    cancelled = canc_cod + canc_pre
    disp_cod, disp_pre = cod - canc_cod, orders - cod - canc_pre
    dispatched = disp_cod + disp_pre
    p_rto_cod = min(0.7, c["rto_cod"] * (1 + 0.08 * (dd - 4)) * (1 + 0.6 * gap) * L["rto_mult"])
    rto = int(rng.binomial(disp_cod, p_rto_cod)) + int(rng.binomial(disp_pre, c["rto_prepaid"]))
    delivered = dispatched - rto
    ret_r = float(np.clip(c["ret0"] * math.exp(c["ret_q"] * q) * (1 + c["ret_gap"] * gap) * math.exp(-0.12 * lz)
                          * L["ret_mult"], 0.005, 0.7))
    returns = int(rng.binomial(delivered, ret_r))
    kept = delivered - returns
    resold = int(rng.binomial(returns, c["resale"]))

    rp = np.array([c["reasons"][r] for r in REASONS], dtype=float)
    rp[1] *= math.exp(-0.4 * q)
    rp[2] *= math.exp(-0.2 * q)
    rp = rp / rp.sum()
    reason_counts = rng.multinomial(returns, rp) if returns else np.zeros(len(REASONS), int)

    # reviews: stars depend on quality and on paying more than the product is worth
    n_k, n_r = int(rng.binomial(kept, 0.14)), int(rng.binomial(returns, 0.18))
    mu_k = 3.95 + 0.45 * q - 0.7 * max(0.0, rel - 0.08)
    mu_r = 2.6 + 0.3 * q
    reviews, stars_sum, pos = [], 0.0, dict(pos=0, neu=0, neg=0, neg_price=0, neg_qual=0)
    if record_reviews:
        t = REVIEW_TEXT[cat]
        for is_ret in [False] * n_k + [True] * n_r:
            s = int(np.clip(round(rng.normal(mu_r if is_ret else mu_k, 0.8)), 1, 5))
            stars_sum += s
            if s >= 4:
                sent, theme, txt = "positive", "praise", t["pos"][rng.integers(len(t["pos"]))]; pos["pos"] += 1
            elif s == 3:
                sent, theme, txt = "neutral", "mixed", t["neu"][rng.integers(len(t["neu"]))]; pos["neu"] += 1
            else:
                sent = "negative"; pos["neg"] += 1
                p_price = 1 / (1 + math.exp(-12 * (rel - 0.10)))
                if rng.random() < 0.15 + 0.7 * p_price:
                    theme, txt = "price", t["neg_price"][rng.integers(len(t["neg_price"]))]; pos["neg_price"] += 1
                else:
                    theme, txt = "quality", t["neg_qual"][rng.integers(len(t["neg_qual"]))]; pos["neg_qual"] += 1
            reviews.append(dict(product_id=o["product_id"], category=cat, week_start=wk.isoformat(), stars=s,
                                sentiment=sent, theme=theme, comment=txt, from_returned_order=int(is_ret)))
    else:                                                  # fast path for truth simulations: same moments
        for n, mu in ((n_k, mu_k), (n_r, mu_r)):
            if n:
                m, v = _star_moments(mu)
                stars_sum += float(np.clip(rng.normal(n * m, math.sqrt(n * v)), n, 5 * n))
    n_new = n_k + n_r
    state["cum_reviews"] += n_new
    state["star_sum"] += stars_sum

    costs = dict(unit=prod["private"]["unit"], packaging=prod["private"]["packaging"], gift=prod["private"]["gift"])
    money = econ.realized_contribution(cat, sell, costs, A, orders=orders, cancelled=cancelled, dispatched=dispatched,
                                       rto=rto, returns=returns, resold=resold, kept=kept)
    state["cum_contrib"] += money["contribution"]
    conv_now = orders / impressions if impressions else 0.0
    state["conv_ema"] = conv_now if state["conv_ema"] is None else 0.6 * state["conv_ema"] + 0.4 * conv_now

    row = dict(
        product_id=o["product_id"], category=cat, subcategory=o["subcategory"], week_start=wk.isoformat(), week_num=w,
        delivery_promise_days=dd, in_stock_share=round(in_stock, 3),
        impressions=impressions, clicks=clicks, add_to_cart=carts, orders_placed=orders,
        cancelled_orders=cancelled, dispatched_orders=dispatched, cod_orders=disp_cod, rto_orders=rto,
        delivered_orders=delivered, customer_returns=returns,
        **{f"return_{r}": int(nr) for r, nr in zip(REASONS, reason_counts)},
        returns_resold=resold, kept_orders=kept,
        new_reviews=n_new, new_reviews_avg_stars=round(stars_sum / n_new, 2) if n_new else None,
        positive_reviews=pos["pos"], neutral_reviews=pos["neu"], negative_reviews=pos["neg"],
        neg_price_mentions=pos["neg_price"], neg_quality_mentions=pos["neg_qual"],
        cumulative_reviews=state["cum_reviews"],
        cumulative_rating=round(state["star_sum"] / state["cum_reviews"], 2) if state["cum_reviews"] else None,
        seller_contribution_inr=round(money["contribution"], 2),
    )
    return row, reviews


def simulate_plan(prod, plan, market, launch, A, seed, record_reviews=False):
    """Truth diagnostic: run one listing through a fixed weekly price plan (no strategy, no quitting).

    Returns weekly orders, kept orders and contribution. The whole plan must fall inside the market calendar.
    """
    if launch + timedelta(days=7 * (len(plan) - 1)) > LAST_WEEK or launch < MARKET_START:
        raise ValueError("plan runs outside the simulated market calendar")
    rng = np.random.default_rng(seed)
    st = new_state()
    out = dict(orders=[], kept=[], contribution=[], cum=[], reviews=[])
    for i, price in enumerate(plan):
        wk = launch + timedelta(days=7 * i)
        row, _ = step_week(prod, st, wk, price, 0.0, market, rng, A, record_reviews=record_reviews)
        out["orders"].append(row["orders_placed"])
        out["kept"].append(row["kept_orders"])
        out["contribution"].append(row["seller_contribution_inr"])
        out["cum"].append(st["cum_contrib"])
        out["reviews"].append(st["cum_reviews"])
    return out


def product_from_row(r):
    """Rebuild a generator product from a products.csv row (hidden_* and seller_* columns). Evaluation only."""
    g = lambda k: float(r[k])
    obs = dict(product_id=r["product_id"], category=r["category"], subcategory=r["subcategory"],
               positioning_tier=r["positioning_tier"], attribute_name=r["attribute_name"],
               attribute_value=str(r["attribute_value"]), listing_quality_score=int(r["listing_quality_score"]),
               delivery_promise_days_base=int(r["delivery_promise_days_base"]),
               launch_week=date.fromisoformat(str(r["launch_week"])))
    private = dict(unit=g("seller_unit_cost_inr"), packaging=g("seller_packaging_inr"), gift=g("seller_free_gift_inr"),
                   budget=g("seller_loss_budget_inr"))
    latent = dict(quality=g("hidden_true_quality"), archetype=r["hidden_archetype"],
                  **{f"resp_{k}": g(f"hidden_resp_{k}") for k in ("beta", "kappa_hi", "kappa_lo", "theta", "sig", "center", "width")},
                  fair_mult=g("hidden_fair_mult"), ret_mult=g("hidden_ret_mult"), demand_mult=g("hidden_demand_mult"),
                  conv_mult=g("hidden_conv_mult"), stock_risk=g("hidden_stock_risk"), cod_share=g("hidden_cod_share"),
                  rto_mult=g("hidden_rto_mult"), strategy=r["hidden_seller_strategy"])
    return dict(obs=obs, private=private, latent=latent)


def true_local_price_slope(prod):
    """d log(conversion) / d (price / subcategory median) at the median price, from the generator's response curve."""
    L = prod["latent"]
    p = response_params(prod)
    rel0 = 1.0 / L["fair_mult"] - 1.0
    h = 1e-4
    return (price_response(rel0 + h, p) - price_response(rel0 - h, p)) / (2 * h) / L["fair_mult"]
