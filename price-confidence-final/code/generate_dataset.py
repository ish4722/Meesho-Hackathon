"""
Price Confidence - synthetic listing dataset (Team Edge Case, Meesho DICE S3)

Generates N_PER_CATEGORY soap + N_PER_CATEGORY t-shirt listings (default 150 each), tracked week by week from launch to
the last complete week before 30 Sep 2026.

EVERYTHING HERE IS SYNTHETIC. The generator encodes assumptions about how
buyers respond to price, ratings and listing quality. A model trained on this
data can only rediscover those assumptions. Use it to demonstrate the method,
never as evidence about real Meesho buyers.

Files written:
  products.csv        one row per listing (hidden_* columns = evaluation only)
  weekly_panel.csv    one row per listing per live week
  reviews.csv         one row per review
  market_weekly.csv   platform-wide category reference per week (what Meesho
                      would see across ALL listings, not just these 100)

Run:  python generate_dataset.py      (seed fixed -> same data every time)
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 20260930
N_PER_CATEGORY = 150                    # set to 50 for the original 50/50 version
rng = np.random.default_rng(SEED)

TODAY = date(2026, 9, 30)
LAST_WEEK = date(2026, 9, 21)           # last complete Monday-start week
MARKET_START = date(2025, 6, 2)
LAUNCH_FIRST = date(2025, 9, 1)
LAUNCH_LAST = date(2026, 8, 3)          # every listing has >= 8 weeks of data
GST = 0.05                              # assumed GST rate, both categories

# ---------------------------------------------------------------- category setup
CAT = {
    "soap": dict(
        n=N_PER_CATEGORY, ship=30, resale_share=0.0,        # returned soap cannot be resold
        base_impr=9000, ctr0=0.045, atc0=0.09, wl0=0.05, ord0=0.36,
        a=0.9, b=1.1, c=1.0,                    # price sensitivity at each step
        ret0=0.08, ret_q=-0.5, ret_price=0.8,
        rto_cod=0.14, rto_prepaid=0.02, cod_share=0.66,
        cost_mu=82, cost_sd=0.22,
        median_start=172, median_end=186,       # market drifts up (gift packs)
        season=[1.10, 1.00, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.10, 1.30, 1.20, 1.10],
        fest_boost=1.25,
        reasons=dict(size_issue=0.0, quality_not_as_expected=0.30, not_as_pictured=0.25,
                     damaged_in_transit=0.30, wrong_item=0.08, changed_mind=0.07),
        free_gift_share=0.2,
    ),
    "tshirt": dict(
        n=N_PER_CATEGORY, ship=70, resale_share=0.70,       # most returned tees go back to stock
        base_impr=12000, ctr0=0.055, atc0=0.08, wl0=0.08, ord0=0.33,
        a=1.1, b=1.3, c=1.2,
        ret0=0.26, ret_q=-0.35, ret_price=0.6,
        rto_cod=0.17, rto_prepaid=0.02, cod_share=0.70,
        cost_mu=120, cost_sd=0.20,
        median_start=329, median_end=309,       # market drifts down (end of season)
        season=[0.80, 0.90, 1.05, 1.20, 1.25, 1.15, 1.00, 0.95, 0.90, 1.05, 0.95, 0.80],
        fest_boost=1.15,
        reasons=dict(size_issue=0.42, quality_not_as_expected=0.22, not_as_pictured=0.16,
                     damaged_in_transit=0.06, wrong_item=0.06, changed_mind=0.08),
        free_gift_share=0.0,
    ),
}
REASONS = ["size_issue", "quality_not_as_expected", "not_as_pictured",
           "damaged_in_transit", "wrong_item", "changed_mind"]

# Synthetic calendar (illustrative, not Meesho's actual sale dates)
FEST_WEEKS = {date(2025, 9, 22), date(2025, 9, 29), date(2025, 10, 6), date(2025, 10, 13),
              date(2025, 10, 20), date(2026, 3, 2), date(2026, 8, 24)}
CAMPAIGN_WEEKS = {date(2025, 9, 22), date(2025, 9, 29), date(2025, 10, 6),
                  date(2026, 1, 12), date(2026, 3, 2), date(2026, 6, 22), date(2026, 9, 21)}

STRATEGIES = ["investment", "market", "premium", "cost_plus_stale"]
STRAT_P = [0.25, 0.35, 0.20, 0.20]


def mondays(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=7)


ALL_WEEKS = list(mondays(MARKET_START, LAST_WEEK))


def market_median(cat, wk):
    """Platform-wide category median price in week wk (true value)."""
    c = CAT[cat]
    span = (LAST_WEEK - MARKET_START).days
    frac = (wk - MARKET_START).days / span
    base = c["median_start"] + (c["median_end"] - c["median_start"]) * frac
    return base * (1 + 0.03 * (c["season"][wk.month - 1] - 1))


def demand_index(cat, wk):
    c = CAT[cat]
    s = c["season"][wk.month - 1]
    if wk in FEST_WEEKS:
        s *= c["fest_boost"]
    return s


def ends_in_9(x):
    return int(max(49, round(x / 10.0) * 10 - 1))


# ---------------------------------------------------------------- market_weekly
mk_rows = []
for cat in CAT:
    for wk in ALL_WEEKS:
        med = market_median(cat, wk) * (1 + rng.normal(0, 0.01))
        mk_rows.append(dict(
            category=cat, week_start=wk.isoformat(),
            platform_median_price=round(med),
            platform_p25_price=round(med * 0.80),
            platform_p75_price=round(med * 1.22),
            category_demand_index=round(demand_index(cat, wk) * (1 + rng.normal(0, 0.02)), 3),
            festival_week=int(wk in FEST_WEEKS),
            campaign_week=int(wk in CAMPAIGN_WEEKS),
        ))
market_df = pd.DataFrame(mk_rows)

# ---------------------------------------------------------------- listings
prod_rows, week_rows, review_rows = [], [], []
launch_weeks = list(mondays(LAUNCH_FIRST, LAUNCH_LAST))

POS_TEXT = {
    "soap": ["Lovely fragrance, skin feels soft", "Good soap, lasts long", "Nice herbal smell, will buy again",
             "Value for money", "Gentle and nice lather", "Packing was neat, soap is good"],
    "tshirt": ["Fabric is soft and fits well", "Good quality for the price", "Colour same as photo",
               "Comfortable, nice stitching", "Value for money", "Fits perfectly, will order again"],
}
NEU_TEXT = {
    "soap": ["Okay soap, nothing special", "Average, smell fades quickly", "Fine for the price"],
    "tshirt": ["Okay t-shirt, average fabric", "Fits fine, colour slightly dull", "Average quality"],
}
NEG_PRICE_TEXT = {
    "soap": ["Too costly for such a small soap", "Not worth the price", "Same soap is cheaper elsewhere"],
    "tshirt": ["Overpriced for this fabric", "Not worth the price", "Cheaper tees are better than this"],
}
NEG_QUAL_TEXT = {
    "soap": ["Soap melted in a few days", "Fragrance not as described", "Arrived broken", "Very small, poor quality"],
    "tshirt": ["Size runs small", "Fabric is thin and rough", "Colour faded after one wash", "Stitching came off"],
}

pid = 0
for cat, c in CAT.items():
    for k in range(c["n"]):
        pid += 1
        product_id = f"{'SP' if cat == 'soap' else 'TS'}{k + 1:03d}"
        launch = launch_weeks[rng.integers(len(launch_weeks))]
        q = float(np.clip(rng.normal(0, 1), -2, 2))            # hidden true quality
        lq = float(np.clip(rng.normal(62 + 6 * q, 12), 20, 98))  # listing quality 0-100 (observable)
        lz = (lq - 62) / 12
        cost = float(c["cost_mu"] * np.exp(c["cost_sd"] * rng.normal() + 0.08 * q))
        promo = float(rng.uniform(10, 15)) if rng.random() < c["free_gift_share"] else 0.0
        strategy = rng.choice(STRATEGIES, p=STRAT_P)
        budget = float(rng.uniform(2000, 8000) * (1.5 if cat == "tshirt" else 1.0))
        cod_p = float(np.clip(rng.normal(c["cod_share"], 0.06), 0.4, 0.9))

        m0 = market_median(cat, launch)
        if strategy == "investment":
            price = ends_in_9(m0 * rng.uniform(0.84, 0.93))
        elif strategy == "market":
            price = ends_in_9(m0 * rng.uniform(0.95, 1.05))
        elif strategy == "premium":
            price = ends_in_9(m0 * rng.uniform(1.08, 1.25))
        else:
            price = ends_in_9((cost + promo) * rng.uniform(1.5, 2.0) + c["ship"])
        launch_price = price

        cum_reviews, star_sum = 0, 0.0
        conv_ema = None
        cum_contrib = 0.0
        raised = cut = False
        status, delist_week = "active", None
        weeks_live = 0

        for w, wk in enumerate(mondays(launch, LAST_WEEK), start=1):
            weeks_live = w
            mm = market_median(cat, wk)
            rating = star_sum / cum_reviews if cum_reviews else None

            # ---------------- seller pricing behaviour
            if strategy == "investment" and not raised and (cum_reviews >= 15 or w >= 8):
                if rating is None or rating >= 3.6:
                    price = ends_in_9(mm * rng.uniform(0.98, 1.08))
                raised = True
            if strategy == "premium" and not cut and w == 5:
                recent = sum(r["orders_placed"] for r in week_rows[-4:] if r["product_id"] == product_id)
                if recent < 25:
                    price = ends_in_9(mm * rng.uniform(0.95, 1.02))
                    cut = True
            if strategy == "market" and w > 1 and w % 10 == 0:
                price = ends_in_9(0.5 * price + 0.5 * mm * rng.uniform(0.97, 1.05))

            disc = 0.0
            if wk in CAMPAIGN_WEEKS and strategy != "cost_plus_stale" and rng.random() < 0.6:
                disc = float(rng.choice([0.10, 0.15, 0.20]))
            sell = round(price * (1 - disc))

            # ---------------- true buyer response (hidden mechanics)
            fair = mm * np.exp(0.08 * q + 0.04 * lz)
            pr = sell / fair
            if cum_reviews == 0:
                star_eff, trust = 0.85, 0.0
            else:
                star_eff = np.exp(0.35 * (rating - 4.0)) * (0.9 + 0.1 * min(1, cum_reviews / 30))
                trust = min(1.0, np.log1p(cum_reviews) / np.log1p(60)) * np.clip((rating - 3.0) / 1.2, 0, 1.2)
            conv_ref = c["ctr0"] * c["atc0"] * c["ord0"]
            rank = 0.55 + 0.45 * trust
            if conv_ema is not None:
                rank += 0.3 * (conv_ema / conv_ref - 1)
            rank = float(np.clip(rank, 0.3, 1.6))
            boost = 1.8 if w <= 2 else (1.3 if w <= 4 else 1.0)
            camp_impr = 1.3 if wk in CAMPAIGN_WEEKS else 1.0

            lam = c["base_impr"] * demand_index(cat, wk) * rank * boost * camp_impr * rng.lognormal(0, 0.15)
            impressions = int(rng.poisson(lam))
            ctr = np.clip(c["ctr0"] * np.exp(-c["a"] * (pr - 1)) * np.exp(0.25 * lz) * star_eff, 0.002, 0.25)
            views = int(rng.binomial(impressions, ctr))
            atc_r = np.clip(c["atc0"] * np.exp(-c["b"] * (pr - 1)) * np.exp(0.20 * q) * (0.85 + 0.15 * trust), 0.005, 0.6)
            atc = int(rng.binomial(views, atc_r))
            wl_r = np.clip(c["wl0"] * np.exp(0.6 * (pr - 1)) * np.exp(0.2 * q), 0.005, 0.4)
            wishlist = int(rng.binomial(views, wl_r))
            ord_r = np.clip(c["ord0"] * np.exp(-c["c"] * (pr - 1)) * (1 + 0.3 * disc)
                            * (1.05 if wk in FEST_WEEKS else 1.0), 0.02, 0.9)
            orders = int(rng.binomial(atc, ord_r))

            cod = int(rng.binomial(orders, cod_p))
            rto = int(rng.binomial(cod, c["rto_cod"])) + int(rng.binomial(orders - cod, c["rto_prepaid"]))
            delivered = orders - rto
            ret_r = np.clip(c["ret0"] * np.exp(c["ret_q"] * q) * (1 + c["ret_price"] * max(0, pr - 1)), 0.005, 0.6)
            returns = int(rng.binomial(delivered, ret_r))
            kept = delivered - returns

            # return reasons: lower quality -> more quality/pictured complaints
            rp = np.array([c["reasons"][r] for r in REASONS], dtype=float)
            rp[1] *= np.exp(-0.4 * q)
            rp[2] *= np.exp(-0.2 * q)
            rp = rp / rp.sum()
            reason_counts = rng.multinomial(returns, rp) if returns else np.zeros(len(REASONS), int)
            resold = int(rng.binomial(returns, c["resale_share"]))

            # ---------------- reviews
            n_rev_k = int(rng.binomial(kept, 0.14))
            n_rev_r = int(rng.binomial(returns, 0.18))
            stars_new = []
            pos = neu = neg = neg_price = neg_qual = 0
            for is_ret in [False] * n_rev_k + [True] * n_rev_r:
                mu = (2.6 + 0.3 * q) if is_ret else (3.95 + 0.45 * q - 0.5 * max(0, pr - 1.05))
                s = int(np.clip(round(rng.normal(mu, 0.8)), 1, 5))
                stars_new.append(s)
                if s >= 4:
                    sent, theme = "positive", "praise"
                    txt = rng.choice(POS_TEXT[cat]); pos += 1
                elif s == 3:
                    sent, theme = "neutral", "mixed"
                    txt = rng.choice(NEU_TEXT[cat]); neu += 1
                else:
                    sent = "negative"; neg += 1
                    p_price = 1 / (1 + np.exp(-12 * (pr - 1.10)))
                    if rng.random() < 0.15 + 0.7 * p_price:
                        theme = "price"; txt = rng.choice(NEG_PRICE_TEXT[cat]); neg_price += 1
                    else:
                        theme = "quality"; txt = rng.choice(NEG_QUAL_TEXT[cat]); neg_qual += 1
                review_rows.append(dict(product_id=product_id, category=cat, week_start=wk.isoformat(),
                                        stars=s, sentiment=sent, theme=theme, comment=str(txt),
                                        from_returned_order=int(is_ret)))
            cum_reviews += len(stars_new)
            star_sum += sum(stars_new)

            # ---------------- seller economics (hidden: needs this seller's cost)
            revenue = kept * sell / (1 + GST)
            cost_total = (orders * c["ship"]                  # forward shipping on every dispatched order
                          + returns * c["ship"]               # reverse shipping on customer returns
                          + (kept + returns - resold) * cost  # stock not recovered (RTO parcels come back intact)
                          + orders * promo)                   # free gift in every parcel
            contrib = revenue - cost_total
            cum_contrib += contrib

            conv_now = orders / impressions if impressions else 0.0
            conv_ema = conv_now if conv_ema is None else 0.6 * conv_ema + 0.4 * conv_now

            week_rows.append(dict(
                product_id=product_id, category=cat, week_start=wk.isoformat(), week_num=w,
                listing_price=price, discount_pct=disc, selling_price=sell,
                festival_week=int(wk in FEST_WEEKS), campaign_week=int(wk in CAMPAIGN_WEEKS),
                impressions=impressions, product_views=views, wishlist_adds=wishlist,
                add_to_cart=atc, orders_placed=orders, cod_orders=cod, rto_orders=rto,
                delivered_orders=delivered, customer_returns=returns,
                **{f"return_{r}": int(n) for r, n in zip(REASONS, reason_counts)},
                returns_resold=resold, kept_orders=kept,
                new_reviews=len(stars_new),
                new_reviews_avg_stars=round(float(np.mean(stars_new)), 2) if stars_new else None,
                positive_reviews=pos, neutral_reviews=neu, negative_reviews=neg,
                neg_price_mentions=neg_price, neg_quality_mentions=neg_qual,
                cumulative_reviews=cum_reviews,
                cumulative_rating=round(star_sum / cum_reviews, 2) if cum_reviews else None,
                hidden_contribution_inr=round(contrib, 2),
            ))

            # ---------------- does the seller give up?
            if w >= 6:
                last4 = sum(r["orders_placed"] for r in week_rows[-4:])
                p_quit = 0.005 if cum_contrib < 0 else 0.0   # profitable sellers don't quit at random
                if last4 < 6:
                    p_quit += 0.35
                if cum_contrib < -budget:
                    p_quit += 0.50
                if wk < LAST_WEEK and rng.random() < p_quit:
                    status, delist_week = "delisted", w
                    break

        rows_p = [r for r in week_rows if r["product_id"] == product_id]
        tot = lambda key: int(sum(r[key] for r in rows_p))
        prod_rows.append(dict(
            product_id=product_id, category=cat,
            title=("Herbal bath soap" if cat == "soap" else "Cotton round-neck t-shirt") + f" #{k + 1}",
            launch_week=launch.isoformat(), launch_price=launch_price,
            platform_median_at_launch=round(m0),
            launch_vs_median=round(launch_price / m0, 3),
            launch_position=("below" if launch_price / m0 < 0.94 else "above" if launch_price / m0 > 1.06 else "at"),
            listing_quality_score=round(lq),
            current_listing_price=rows_p[-1]["listing_price"],
            status=status, delist_week=delist_week, weeks_live=weeks_live,
            total_orders=tot("orders_placed"), total_kept=tot("kept_orders"),
            total_returns=tot("customer_returns"), total_rto=tot("rto_orders"),
            review_count=cum_reviews,
            final_rating=round(star_sum / cum_reviews, 2) if cum_reviews else None,
            hidden_unit_cost_inr=round(cost, 2), hidden_free_gift_inr=round(promo, 2),
            hidden_seller_strategy=strategy, hidden_true_quality=round(q, 3),
            hidden_launch_budget_inr=round(budget),
            hidden_cumulative_contribution_inr=round(cum_contrib, 2),
        ))

products = pd.DataFrame(prod_rows)
weekly = pd.DataFrame(week_rows)
reviews = pd.DataFrame(review_rows)
reviews.insert(0, "review_id", [f"R{i + 1:05d}" for i in range(len(reviews))])

products.to_csv("products.csv", index=False)
weekly.to_csv("weekly_panel.csv", index=False)
reviews.to_csv("reviews.csv", index=False)
market_df.to_csv("market_weekly.csv", index=False)
print(len(products), "listings |", len(weekly), "listing-weeks |", len(reviews), "reviews |",
      len(market_df), "market rows")
