"""
Price Confidence - synthetic listing dataset, generator v2 (Team Edge Case, Meesho DICE S3).

EVERYTHING HERE IS SYNTHETIC. synth_market.py defines the assumed buyer behaviour; this script runs
seller pricing strategies through it week by week and writes:

  products.csv           one row per listing (seller_* = seller-private inputs, hidden_* = generator-only truth)
  weekly_panel.csv       one row per listing per live week, funnel from impressions to reviews
  reviews.csv            one row per review
  market_weekly.csv      subcategory competitor prices and category demand per week
  dataset_manifest.json  seed, versions, configuration, row counts, file hashes and provenance

Run:
  python generate_dataset.py                       # default scenario and seed
  python generate_dataset.py --seed 7 --out /tmp/x # another draw of the same world
  python generate_dataset.py --scenario homogeneous      # every product 'moderate' (comparison world)
  python generate_dataset.py --scenario low_price_variation  # no experiments or campaign discounts

The same seed, scenario and library versions give byte-identical files; the manifest records the
versions so a mismatch can be detected (numpy has changed its random streams between releases).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

import economics as econ
import schema
import synth_market as sm

DEFAULT_SEED = 20261021
SCENARIOS = {
    "default": "Heterogeneous price response (six archetypes), mixed seller strategies.",
    "homogeneous": "Every product uses the 'moderate' archetype; for comparison with the default world.",
    "low_price_variation": "No price experiments and no campaign discounts; tests the data-quality warning on price variation.",
}


def lagged_median(market, cat, sub, wk):
    prev = wk - timedelta(days=7)
    return market.comp_median(cat, sub, prev if prev >= sm.MARKET_START else wk)


def seller_price(prod, sst, w, wk, market, rng, hist, scenario):
    """The seller's listing price for week w and the recorded reason. Uses only what the seller can see."""
    o, L = prod["obs"], prod["latent"]
    strat = L["strategy"]
    ref = lagged_median(market, o["category"], o["subcategory"], wk) * sm.TIER_FAIR[o["positioning_tier"]]
    price, reason = sst["price"], "none"
    if w == 1:
        return price, "launch"
    if strat == "investment" and not sst.get("raised") and (sst["reviews"] >= 15 or w >= 8):
        sst["raised"] = True
        if sst["rating"] is None or sst["rating"] >= 3.6:
            return sm.ends_in_9(ref * rng.uniform(0.98, 1.08)), "raise_after_reviews"
    if strat == "premium" and not sst.get("cut") and w == 5:
        sst["cut"] = True
        if sum(h["orders_placed"] for h in hist[-4:]) < 25:
            return sm.ends_in_9(ref * rng.uniform(0.95, 1.02)), "reactive_cut"
    if strat == "market" and w % 10 == 0:
        return sm.ends_in_9(0.5 * price + 0.5 * ref * rng.uniform(0.97, 1.05)), "market_follow"
    if strat == "reactive" and w >= 4 and sst.get("cuts", 0) < 3:
        recent = np.mean([h["orders_placed"] for h in hist[-2:]])
        early = np.mean([h["orders_placed"] for h in hist[:2]])
        if recent < 0.5 * early:
            sst["cuts"] = sst.get("cuts", 0) + 1
            return sm.ends_in_9(price * 0.92), "reactive_cut"
        if recent > 1.5 * early and w % 4 == 0:
            return sm.ends_in_9(price * 1.05), "reactive_raise"
    if strat == "experimenter" and scenario != "low_price_variation" and 3 <= w <= 14:
        block = (w - 3) // 2
        if (w - 3) % 2 == 0:                                   # new 2-week block: randomised arm
            arm = sst["arms"][block % len(sst["arms"])]
            return sm.ends_in_9(sst["base"] * arm), "experiment"
        return price, "none"
    if strat == "experimenter" and w == 15:
        return sm.ends_in_9(sst["base"]), "experiment"
    return price, reason


def launch_price(prod, launch, market, rng, A):
    o, L, P = prod["obs"], prod["latent"], prod["private"]
    ref = lagged_median(market, o["category"], o["subcategory"], launch) * sm.TIER_FAIR[o["positioning_tier"]]
    s = L["strategy"]
    if s == "investment":
        return sm.ends_in_9(ref * rng.uniform(0.84, 0.93))
    if s == "premium":
        return sm.ends_in_9(ref * rng.uniform(1.08, 1.25))
    if s == "cost_plus_stale":
        ship = A["economics"]["forward_shipping_inr"][o["category"]]
        return sm.ends_in_9((P["unit"] + P["gift"] + P["packaging"]) * rng.uniform(1.5, 2.0) + ship)
    return sm.ends_in_9(ref * rng.uniform(0.95, 1.05))       # market, reactive, experimenter


def run(seed, scenario, n_per_category, out_dir):
    A = econ.load_assumptions()
    rng = np.random.default_rng(seed)
    market = sm.Market(rng)
    market_rows = market.rows(rng)
    launch_weeks = list(sm.mondays(sm.LAUNCH_FIRST, sm.LAUNCH_LAST))
    prod_rows, week_rows, review_rows = [], [], []

    for cat, c in sm.CATEGORIES.items():
        for k in range(n_per_category or c["n"]):
            launch = launch_weeks[rng.integers(len(launch_weeks))]
            prod = sm.draw_product(cat, k, rng, launch)
            if scenario == "homogeneous":
                a = sm.ARCHETYPES["moderate"]
                prod["latent"].update(archetype="moderate", resp_beta=2.1 * c["subcats"][prod["obs"]["subcategory"]]["beta_mult"],
                                      resp_kappa_hi=1.0, resp_kappa_lo=0.0, resp_theta=0.0, resp_sig=0.0)
            if scenario == "low_price_variation" and prod["latent"]["strategy"] == "experimenter":
                prod["latent"]["strategy"] = "market"
            o, L, P = prod["obs"], prod["latent"], prod["private"]
            price = launch_price(prod, launch, market, rng, A)
            arms = list(rng.permutation([0.90, 1.0, 1.10]))
            sst = dict(price=price, base=price, reviews=0, rating=None, arms=arms)
            state = sm.new_state()
            hist, status, delist_week = [], "active", None
            for w, wk in enumerate(sm.mondays(launch, sm.LAST_WEEK), start=1):
                lp, reason = seller_price(prod, sst, w, wk, market, rng, hist, scenario)
                sst["price"] = lp
                disc = 0.0
                if (wk in sm.CAMPAIGN_WEEKS and L["strategy"] != "cost_plus_stale" and scenario != "low_price_variation"
                        and rng.random() < 0.6):
                    disc = float(rng.choice([0.10, 0.15, 0.20]))
                    reason = "campaign_discount" if reason in ("none",) else reason + "+campaign_discount"
                sell = int(round(lp * (1 - disc)))
                dd = int(np.clip(o["delivery_promise_days_base"] + (rng.choice([-1, 1]) if rng.random() < 0.05 else 0), 1, 9))
                row, revs = sm.step_week(prod, state, wk, sell, disc, market, rng, A, delivery_days=dd, record_reviews=True)
                if w > 1 and reason == "none" and (hist[-1]["selling_price"] != sell):
                    reason = "campaign_ended"
                row.update(listing_price=lp, discount_pct=disc, selling_price=sell, price_change_reason=reason,
                           festival_week=int(wk in sm.FEST_WEEKS), campaign_week=int(wk in sm.CAMPAIGN_WEEKS))
                week_rows.append(row)
                review_rows.extend(revs)
                hist.append(row)
                sst["reviews"] = state["cum_reviews"]
                sst["rating"] = state["star_sum"] / state["cum_reviews"] if state["cum_reviews"] else None
                if w >= 6:                                         # does the seller give up?
                    last4 = sum(h["orders_placed"] for h in hist[-4:])
                    p_quit = 0.005 if state["cum_contrib"] < 0 else 0.0
                    if last4 < 6:
                        p_quit += 0.35
                    if state["cum_contrib"] < -P["budget"]:
                        p_quit += 0.50
                    if wk < sm.LAST_WEEK and rng.random() < p_quit:
                        status, delist_week = "delisted", w
                        break
            m0 = lagged_median(market, cat, o["subcategory"], launch)
            lp0 = hist[0]["listing_price"]
            tot = lambda key: int(sum(h[key] for h in hist))
            title_noun = sm.CATEGORIES[cat]["subcats"][o["subcategory"]]["label"]
            prod_rows.append(dict(
                product_id=o["product_id"], category=cat, subcategory=o["subcategory"], positioning_tier=o["positioning_tier"],
                attribute_name=o["attribute_name"], attribute_value=o["attribute_value"],
                title=f"{title_noun} #{k + 1}", listing_quality_score=o["listing_quality_score"],
                delivery_promise_days_base=o["delivery_promise_days_base"], launch_week=launch.isoformat(),
                launch_price=lp0, platform_median_at_launch=round(m0), launch_vs_median=round(lp0 / m0, 3),
                launch_position="below" if lp0 / m0 < 0.94 else "above" if lp0 / m0 > 1.06 else "at",
                current_listing_price=hist[-1]["listing_price"], status=status, delist_week=delist_week,
                weeks_live=len(hist), total_orders=tot("orders_placed"), total_kept=tot("kept_orders"),
                total_returns=tot("customer_returns"), total_rto=tot("rto_orders"),
                review_count=state["cum_reviews"],
                final_rating=round(state["star_sum"] / state["cum_reviews"], 2) if state["cum_reviews"] else None,
                seller_unit_cost_inr=P["unit"], seller_packaging_inr=P["packaging"], seller_free_gift_inr=P["gift"],
                seller_loss_budget_inr=P["budget"],
                hidden_true_quality=round(L["quality"], 4), hidden_archetype=L["archetype"],
                **{f"hidden_resp_{kk}": round(L[f"resp_{kk}"], 4)
                   for kk in ("beta", "kappa_hi", "kappa_lo", "theta", "sig", "center", "width")},
                hidden_fair_mult=round(L["fair_mult"], 4), hidden_ret_mult=L["ret_mult"],
                hidden_demand_mult=round(L["demand_mult"], 4), hidden_conv_mult=round(L["conv_mult"], 4),
                hidden_stock_risk=round(L["stock_risk"], 4), hidden_cod_share=round(L["cod_share"], 4),
                hidden_rto_mult=round(L["rto_mult"], 4), hidden_seller_strategy=L["strategy"],
                hidden_cumulative_contribution_inr=round(state["cum_contrib"], 2),
            ))

    products = pd.DataFrame(prod_rows)
    weekly = pd.DataFrame(week_rows)[list(schema.WEEKLY)]
    reviews = pd.DataFrame(review_rows)
    reviews.insert(0, "review_id", [f"R{i + 1:06d}" for i in range(len(reviews))])
    market_df = pd.DataFrame(market_rows)
    os.makedirs(out_dir, exist_ok=True)
    files = {"products.csv": products, "weekly_panel.csv": weekly, "reviews.csv": reviews, "market_weekly.csv": market_df}
    for name, df in files.items():
        df.to_csv(os.path.join(out_dir, name), index=False, lineterminator="\n")

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    cfg = dict(categories={k: {kk: vv for kk, vv in v.items()} for k, v in sm.CATEGORIES.items()},
               archetypes=sm.ARCHETYPES, archetype_p=sm.ARCHETYPE_P, strategies=dict(zip(sm.STRATEGIES, sm.STRAT_P)),
               tier_fair=sm.TIER_FAIR, tier_q_mean=sm.TIER_Q_MEAN, calendar=dict(
                   festival=sorted(d.isoformat() for d in sm.FEST_WEEKS),
                   campaign=sorted(d.isoformat() for d in sm.CAMPAIGN_WEEKS)),
               dates=dict(market_start=sm.MARKET_START.isoformat(), launch_first=sm.LAUNCH_FIRST.isoformat(),
                          launch_last=sm.LAUNCH_LAST.isoformat(), last_week=sm.LAST_WEEK.isoformat()),
               assumptions=A["economics"])
    cfg_json = json.dumps(cfg, sort_keys=True, default=str)
    manifest = dict(
        provenance="SYNTHETIC. Generated by generate_dataset.py from assumed buyer behaviour (synth_market.py). "
                   "Not Meesho data and not public marketplace observations.",
        generator_version=sm.GENERATOR_VERSION, schema_version=schema.SCHEMA_VERSION,
        seed=seed, scenario=scenario, scenario_description=SCENARIOS[scenario],
        n_per_category=n_per_category or {k: v["n"] for k, v in sm.CATEGORIES.items()},
        config_sha256=hashlib.sha256(cfg_json.encode()).hexdigest(), config=json.loads(cfg_json),
        rows={k: int(len(v)) for k, v in files.items()},
        products_by_category=products.category.value_counts().sort_index().to_dict(),
        weeks=dict(first=weekly.week_start.min(), last=weekly.week_start.max(),
                   distinct=int(weekly.week_start.nunique())),
        files={k: sha(os.path.join(out_dir, k)) for k in files},
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        column_roles={t: {c: r for c, (r, _) in spec.items()} for t, spec in schema.TABLES.items()},
    )
    with open(os.path.join(out_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    ap.add_argument("--n-per-category", type=int, default=0, help="0 = use each category's configured n")
    ap.add_argument("--out", default=".", help="output folder")
    a = ap.parse_args(argv)
    m = run(a.seed, a.scenario, a.n_per_category, a.out)
    print(m["rows"], "| products by category", m["products_by_category"], "| seed", a.seed, "| scenario", a.scenario)


if __name__ == "__main__":
    main()
