"""
Price Confidence - data-quality report for the (synthetic) dataset.

Checks schema, missing values, duplicate keys, invalid prices and costs, negative counts,
funnel consistency, rates and denominators, missing weeks, coverage, concentration and whether
there is enough informative price variation to estimate a price response.

  python data_quality.py                 # checks the CSVs in this folder, writes reports/data_quality.{md,json}
  python data_quality.py --data /tmp/x   # another folder

Exit code 1 if any HARD check fails. WARN results are reported but do not fail the run.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import schema

COUNT_COLS = ["impressions", "clicks", "add_to_cart", "orders_placed", "cancelled_orders", "dispatched_orders",
              "cod_orders", "rto_orders", "delivered_orders", "customer_returns", "returns_resold", "kept_orders",
              "new_reviews", "positive_reviews", "neutral_reviews", "negative_reviews", "neg_price_mentions",
              "neg_quality_mentions", "cumulative_reviews"] + [f"return_{r}" for r in schema.RETURN_REASONS]
EXPECTED_MISSING = {("weekly_panel", "new_reviews_avg_stars"), ("weekly_panel", "cumulative_rating"),
                    ("products", "delist_week"), ("products", "final_rating")}
# A price change is EXOGENOUS to the listing's own sales if the reason does not depend on them.
EXOGENOUS = {"experiment", "campaign_discount", "campaign_ended", "market_follow"}
ENDOGENOUS = {"reactive_cut", "reactive_raise", "raise_after_reviews"}
THRESHOLDS = dict(min_share_products_multi_price=0.30, min_median_within_sd_logprice=0.02,
                  min_share_exogenous_changes=0.20, min_listings_per_category=50)


def load(data_dir):
    rd = lambda n: pd.read_csv(os.path.join(data_dir, n))
    return dict(products=rd("products.csv"), weekly_panel=rd("weekly_panel.csv"),
                market_weekly=rd("market_weekly.csv"), reviews=rd("reviews.csv"))


def run_checks(T):
    checks = []

    def add(name, ok, detail, level="HARD"):
        checks.append(dict(check=name, status="PASS" if ok else ("FAIL" if level == "HARD" else "WARN"),
                           level=level, detail=detail))

    P, W, Mk, R = T["products"], T["weekly_panel"], T["market_weekly"], T["reviews"]

    # schema
    for t, df in T.items():
        unknown, missing = schema.check_columns(t, list(df.columns))
        add(f"schema:{t}", not unknown and not missing, f"undeclared={unknown} missing={missing}")

    # missing values
    for t, df in T.items():
        na = df.isna().sum()
        bad = {c: int(n) for c, n in na.items() if n and (t, c) not in EXPECTED_MISSING}
        add(f"missing_values:{t}", not bad, f"unexpected missing: {bad or 'none'}; expected-missing columns: "
            + str({c: int(n) for c, n in na.items() if n and (t, c) in EXPECTED_MISSING}))
    expl = W[W.new_reviews_avg_stars.isna()].new_reviews.eq(0).all() and W[W.cumulative_rating.isna()].cumulative_reviews.eq(0).all()
    add("missing_values:explained", bool(expl), "missing stars/rating only where there are no reviews")

    # duplicate keys
    add("duplicates:products", not P.product_id.duplicated().any(), f"{int(P.product_id.duplicated().sum())} duplicate product_id")
    add("duplicates:weekly", not W.duplicated(["product_id", "week_num"]).any() and not W.duplicated(["product_id", "week_start"]).any(),
        f"{int(W.duplicated(['product_id', 'week_num']).sum())} duplicate (product, week)")
    add("duplicates:market", not Mk.duplicated(["category", "subcategory", "week_start"]).any(), "unique (category, subcategory, week)")
    add("duplicates:reviews", not R.review_id.duplicated().any(), "unique review_id")

    # prices and costs
    bad_price = int(((W.selling_price <= 0) | (W.listing_price <= 0) | (W.selling_price > W.listing_price)
                     | (W.discount_pct < 0) | (W.discount_pct > 0.9)).sum())
    add("prices:valid", bad_price == 0, f"{bad_price} rows with a non-positive price, price above list price or invalid discount")
    cost_cols = ["seller_unit_cost_inr", "seller_packaging_inr", "seller_free_gift_inr", "seller_loss_budget_inr"]
    bad_cost = int((P[cost_cols] < 0).any(axis=1).sum() + (P.seller_unit_cost_inr <= 0).sum())
    add("costs:valid", bad_cost == 0, f"{bad_cost} listings with a negative cost or non-positive unit cost")

    # counts
    neg = {c: int((W[c] < 0).sum()) for c in COUNT_COLS if (W[c] < 0).any()}
    nonint = {c: int((W[c] % 1 != 0).sum()) for c in COUNT_COLS if (W[c] % 1 != 0).any()}
    add("counts:non_negative_integers", not neg and not nonint, f"negative={neg or 'none'} non-integer={nonint or 'none'}")

    # funnel identities and orderings
    rules = {
        "clicks<=impressions": W.clicks <= W.impressions,
        "carts<=clicks": W.add_to_cart <= W.clicks,
        "orders<=carts": W.orders_placed <= W.add_to_cart,
        "dispatched=orders-cancelled": W.dispatched_orders == W.orders_placed - W.cancelled_orders,
        "cod<=dispatched": W.cod_orders <= W.dispatched_orders,
        "rto<=dispatched": W.rto_orders <= W.dispatched_orders,
        "delivered=dispatched-rto": W.delivered_orders == W.dispatched_orders - W.rto_orders,
        "returns<=delivered": W.customer_returns <= W.delivered_orders,
        "resold<=returns": W.returns_resold <= W.customer_returns,
        "kept=delivered-returns": W.kept_orders == W.delivered_orders - W.customer_returns,
        "reasons sum to returns": W[[f"return_{r}" for r in schema.RETURN_REASONS]].sum(axis=1) == W.customer_returns,
        "review sentiment sums": W.positive_reviews + W.neutral_reviews + W.negative_reviews == W.new_reviews,
        "negative themes sum": W.neg_price_mentions + W.neg_quality_mentions == W.negative_reviews,
        "reviews<=kept+returns": W.new_reviews <= W.kept_orders + W.customer_returns,
    }
    for name, ok in rules.items():
        add(f"funnel:{name}", bool(ok.all()), f"{int((~ok).sum())} violating rows")
    cum = W.sort_values(["product_id", "week_num"]).groupby("product_id").new_reviews.cumsum()
    ok = (cum.sort_index() == W.cumulative_reviews.sort_index())
    add("funnel:cumulative_reviews=running sum", bool(ok.all()), f"{int((~ok).sum())} rows")
    rv = R.groupby(["product_id", "week_start"]).size()
    wn = W.set_index(["product_id", "week_start"]).new_reviews
    add("reviews table matches panel", bool(rv.reindex(wn.index, fill_value=0).eq(wn).all()), "reviews.csv rows = new_reviews per week")

    # rates in range
    rate_bad = int(((W.in_stock_share <= 0) | (W.in_stock_share > 1)).sum()
                   + ((W.cumulative_rating < 1) | (W.cumulative_rating > 5)).sum() + (~R.stars.between(1, 5)).sum())
    add("rates:in_range", rate_bad == 0, f"{rate_bad} out-of-range stock shares, ratings or stars")

    # time series regularity
    W["_ws"] = pd.to_datetime(W.week_start)
    g = W.sort_values(["product_id", "week_num"]).groupby("product_id")
    gaps = int(g.week_num.apply(lambda s: (s.diff().dropna() != 1).sum()).sum()
               + g._ws.apply(lambda s: (s.diff().dropna() != pd.Timedelta(days=7)).sum()).sum())
    starts = int((g.week_num.min() != 1).sum())
    first = g._ws.min().dt.date.astype(str)
    lw = P.set_index("product_id").launch_week.reindex(first.index)
    add("time:regular_weeks", gaps == 0 and starts == 0 and bool((first == lw).all()),
        f"{gaps} gaps, {starts} series not starting at week 1, launch_week mismatches={int((first != lw).sum())}")
    mk_counts = Mk.groupby(["category", "subcategory"]).week_start.nunique()
    add("time:market_complete", mk_counts.nunique() == 1, f"weeks per subcategory: {sorted(set(mk_counts))}")
    W.drop(columns="_ws", inplace=True)

    # coverage
    cov_cat = P.groupby("category").size().to_dict()
    add("coverage:listings_per_category", min(cov_cat.values()) >= THRESHOLDS["min_listings_per_category"], cov_cat, level="WARN")
    cov_cell = P.groupby(["category", "subcategory", "positioning_tier"]).size()
    add("coverage:category x subcategory x tier", int(cov_cell.min()) >= 3,
        f"{len(cov_cell)} cells, smallest {int(cov_cell.min())}", level="WARN")
    Wm = W.merge(Mk[["category", "subcategory", "week_start", "platform_median_price"]], on=["category", "subcategory", "week_start"])
    rel = Wm.selling_price / Wm.platform_median_price
    seg = pd.cut(rel, [0, 0.85, 0.95, 1.05, 1.15, 10], labels=["<0.85", "0.85-0.95", "0.95-1.05", "1.05-1.15", ">1.15"])
    seg_tab = pd.crosstab(Wm.category, seg)
    add("coverage:price segments", bool((seg_tab > 20).all().all()), seg_tab.to_dict(orient="index"), level="WARN")

    # concentration
    po = W.groupby(["category", "product_id"]).orders_placed.sum()
    top = {cat: round(float(s.sort_values(ascending=False).head(max(1, len(s) // 10)).sum() / s.sum()), 3)
           for cat, s in po.groupby(level=0)}
    hhi = {cat: round(float(((s / s.sum()) ** 2).sum()), 4) for cat, s in po.groupby(level=0)}
    add("concentration:top10pct_order_share", max(top.values()) < 0.6, f"top-10% listings' share of orders {top}; HHI {hhi}",
        level="WARN")

    # price variation
    gp = W.groupby("product_id")
    multi = float((gp.selling_price.nunique() > 1).mean())
    sd_lp = float(gp.selling_price.apply(lambda s: np.log(s).std(ddof=0)).median())
    ch = W[W.price_change_reason != "none"].price_change_reason
    base = ch.str.split("+").str[0]
    base = base[base != "launch"]
    exo = float(base.isin(EXOGENOUS).mean()) if len(base) else 0.0
    endo = float(base.isin(ENDOGENOUS).mean()) if len(base) else 0.0
    informative = (multi >= THRESHOLDS["min_share_products_multi_price"] and sd_lp >= THRESHOLDS["min_median_within_sd_logprice"]
                   and exo >= THRESHOLDS["min_share_exogenous_changes"])
    add("price_variation:informative", informative,
        f"{multi:.0%} of listings have >1 selling price; median within-listing SD of log price {sd_lp:.3f}; "
        f"price changes: {exo:.0%} exogenous (experiments, campaigns, scheduled market follow), {endo:.0%} reacting to own sales",
        level="WARN")

    summary = dict(
        rows={t: int(len(df)) for t, df in T.items()},
        products=int(P.product_id.nunique()), categories=sorted(P.category.unique().tolist()),
        weeks=int(W.week_start.nunique()), first_week=str(W.week_start.min()), last_week=str(W.week_start.max()),
        missing_cells={t: int(df.isna().sum().sum()) for t, df in T.items()},
        price_variation=dict(share_multi_price=round(multi, 3), median_within_sd_logprice=round(sd_lp, 4),
                             share_exogenous_changes=round(exo, 3), share_endogenous_changes=round(endo, 3),
                             change_reasons=base.value_counts().to_dict()),
        distributions={cat: dict(
            orders_per_week=dict(zip(["p10", "p50", "p90"], [float(x) for x in d.orders_placed.quantile([.1, .5, .9])])),
            orders_per_1000_impressions=round(float(d.orders_placed.sum() / d.impressions.sum() * 1000), 3),
            cancel_rate=round(float(d.cancelled_orders.sum() / max(1, d.orders_placed.sum())), 3),
            rto_rate=round(float(d.rto_orders.sum() / max(1, d.dispatched_orders.sum())), 3),
            return_rate=round(float(d.customer_returns.sum() / max(1, d.delivered_orders.sum())), 3),
            variance_to_mean_orders=round(float(d.orders_placed.var() / d.orders_placed.mean()), 1),
        ) for cat, d in W.groupby("category")},
        price_segments=seg_tab.to_dict(orient="index"),
    )
    return checks, summary


def write_reports(checks, summary, out_dir, manifest=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "data_quality.json"), "w", encoding="utf-8") as f:
        json.dump(dict(checks=checks, summary=summary, thresholds=THRESHOLDS,
                       provenance=(manifest or {}).get("provenance")), f, indent=1, default=str)
    lines = ["# Data-quality report", ""]
    if manifest:
        lines += [f"**{manifest['provenance']}**", "",
                  f"Generator {manifest['generator_version']}, schema {manifest['schema_version']}, seed {manifest['seed']}, "
                  f"scenario `{manifest['scenario']}`, numpy {manifest['environment']['numpy']}.", ""]
    s = summary
    lines += [f"Rows: {s['rows']}. Listings: {s['products']} in {', '.join(s['categories'])}. "
              f"Weeks: {s['weeks']} ({s['first_week']} to {s['last_week']}).", "",
              "| Check | Level | Result | Detail |", "|---|---|---|---|"]
    for c in checks:
        d = str(c["detail"]).replace("|", "/")
        lines.append(f"| {c['check']} | {c['level']} | {c['status']} | {d[:300]} |")
    pv = s["price_variation"]
    lines += ["", "## Price variation", "",
              f"- Listings with more than one selling price: {pv['share_multi_price']:.0%}",
              f"- Median within-listing SD of log price: {pv['median_within_sd_logprice']:.3f}",
              f"- Price changes not driven by the listing's own sales: {pv['share_exogenous_changes']:.0%}; "
              f"driven by its own sales: {pv['share_endogenous_changes']:.0%}",
              f"- Reasons: {pv['change_reasons']}", "",
              "## Key distributions by category", "", "| Category | Orders/week p10-p50-p90 | Orders per 1,000 impressions | Cancel | RTO | Returns | Var/mean of weekly orders |",
              "|---|---|---|---|---|---|---|"]
    for cat, d in s["distributions"].items():
        o = d["orders_per_week"]
        lines.append(f"| {cat} | {o['p10']:.0f} / {o['p50']:.0f} / {o['p90']:.0f} | {d['orders_per_1000_impressions']} | "
                     f"{d['cancel_rate']:.1%} | {d['rto_rate']:.1%} | {d['return_rate']:.1%} | {d['variance_to_mean_orders']} |")
    lines += ["", "More rows do not make synthetic data more realistic; these checks only show the files are internally "
                  "consistent and contain the variation the models need."]
    with open(os.path.join(out_dir, "data_quality.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    T = load(a.data)
    checks, summary = run_checks(T)
    mpath = os.path.join(a.data, "dataset_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else None
    write_reports(checks, summary, a.out, manifest)
    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    print(f"{len(checks)} checks: {len(fails)} FAIL, {len(warns)} WARN -> {a.out}/data_quality.md")
    for c in fails + warns:
        print(f"  {c['status']} {c['check']}: {str(c['detail'])[:160]}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
