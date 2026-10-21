"""
Price Confidence - evaluation v2 (Team Edge Case, Meesho DICE S3).

EVERYTHING HERE RUNS ON SYNTHETIC DATA. Good numbers show that the method recovers the assumptions built
into synth_market.py; they do not show accuracy on real Meesho sellers.

Design
  Splits      listings are split 60 / 20 / 20 into train, validation and a final holdout (stratified by category,
              fixed seed). Models are fitted on TRAIN listings using only weeks before T0. Validation is used for
              nothing except reporting (no tuning was done on it); the holdout is reported once.
  Task A      one-week-ahead orders for validation / holdout listings, weeks >= T0. Inputs are known at the start of
              the week: the listing's own past, last week's market, this week's price and calendar.
  Task B      launch-time prediction for listings launched on or after T0: week-1 and weeks 1-4 orders from
              launch-time facts and the planned prices only.
  Task C      recursive simulation of the first 12 weeks (or fewer if the listing stopped) with engine.js, without
              any of the listing's realised outcomes; market frozen at launch.
  Decisions   for every holdout listing, prices are judged by the generator itself (synth_market.simulate_plan with
              the listing's hidden parameters and the seller's private costs) at a common diagnostic launch date:
              regret against the best price, true chance of loss, break-even and budget violations.

  python evaluate.py                     # full run (several minutes), writes reports/evaluation.{md,json}
  python evaluate.py --quick             # smaller samples, for a smoke test
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import economics as econ
import features as F
import schema
import synth_market as smk
import train_models as T

HERE = os.path.dirname(os.path.abspath(__file__))
T0 = "2026-03-02"
SPLIT_SEED = 11
DIAG_LAUNCH = date(2026, 6, 29)
TRUTH_REPS = 40
HORIZON = 12


# ----------------------------------------------------------------------------------------------- helpers
def split_products(P, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    out = {}
    for cat in sorted(P.category.unique()):
        ids = sorted(P[P.category == cat].product_id)
        ids = [ids[i] for i in rng.permutation(len(ids))]
        n = len(ids)
        a, b = int(round(0.6 * n)), int(round(0.8 * n))
        for i in ids[:a]:
            out[i] = "train"
        for i in ids[a:b]:
            out[i] = "validation"
        for i in ids[b:]:
            out[i] = "holdout"
    return out


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if len(y) == 0:
        return dict(n=0)
    err = pred - y
    tot = y.sum()
    return dict(n=int(len(y)), mae=float(np.mean(np.abs(err))), wape=float(np.abs(err).sum() / tot) if tot > 0 else None,
                bias=float(err.sum() / tot) if tot > 0 else None, mean_actual=float(y.mean()))


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi)))) if len(y) else None


def nb_interval(mean, var, q=(0.1, 0.9)):
    """Quantiles of a negative binomial with the given mean and variance (Poisson if var <= mean)."""
    mean, var = np.asarray(mean, float), np.asarray(var, float)
    lo, hi = np.empty_like(mean), np.empty_like(mean)
    for i, (m, v) in enumerate(zip(mean, var)):
        if m <= 0:
            lo[i] = hi[i] = 0
        elif v <= m * 1.0001:
            lo[i], hi[i] = stats.poisson.ppf(q[0], m), stats.poisson.ppf(q[1], m)
        else:
            r = m * m / (v - m)
            p = r / (r + m)
            lo[i], hi[i] = stats.nbinom.ppf(q[0], r, p), stats.nbinom.ppf(q[1], r, p)
    return lo, hi


def price_segment(rel):
    return np.where(rel < -0.10, "below median (>10%)", np.where(rel > 0.10, "above median (>10%)", "near median (+-10%)"))


def sliced(df, ycol, preds, by):
    out = {}
    for key, g in df.groupby(by, observed=True):
        out[str(key)] = {name: metrics(g[ycol], g[p]) for name, p in preds.items()}
    return out


# ----------------------------------------------------------------------------------------------- python mirror of the model
def price_term(rel, tier, cv, support):
    rel = np.asarray(rel, float)
    s1 = cv["rel"] + np.where(tier == "premium", cv["rel_x_premium"], 0.0) + np.where(tier == "basic", cv["rel_x_basic"], 0.0)
    s2 = cv["rel2"]
    lo = np.array([support.get(t, support["all"])[0] for t in tier])
    hi = np.array([support.get(t, support["all"])[1] for t in tier])
    g = lambda r: s1 * r + s2 * r * r
    dg = lambda r: s1 + 2 * s2 * r
    return np.where(rel > hi, g(hi) + np.minimum(dg(hi), 0) * (rel - hi),
                    np.where(rel < lo, g(lo) + np.maximum(dg(lo), 0) * (rel - lo), g(rel)))


def second_base(coef, fr):
    b = np.full(len(fr), coef.get("const", 0.0)) + coef.get("lq", 0.0) * fr.lq.values
    for name, col in (("tier", "positioning_tier"), ("sub", "subcategory"), ("attr", "attribute_value")):
        vals = fr[col].astype(str).values
        for k, v in coef.items():
            if k.startswith(name + ":"):
                b = b + v * (vals == k.split(":", 1)[1])
    return b


def expected_orders(model, cat, fr):
    """Model mean of weekly orders given history-based features (zero listing effect)."""
    C = model["categories"][cat]
    cv, iv = C["conv"]["coef"], C["impr"]["coef"]
    fr = fr.copy()
    for col, (lo, hi) in C["state_ranges"].items():            # same guard as engine.js
        fr[col] = fr[col].clip(lo if col != "logrev" else None, hi)
    lin_c = sum(cv[c] * fr[c].values for c in schema.MODEL_FEATURES["conversion"] if not c.startswith("rel"))
    lc = second_base(C["conv"]["second"], fr) + lin_c + price_term(fr.rel.values, fr.positioning_tier.values, cv, C["price_support"])
    li = second_base(C["impr"]["second"], fr) + sum(iv[c] * fr[c].values for c in schema.MODEL_FEATURES["visibility"]) \
        + math.log(C["in_stock_mean"])
    return np.exp(li) * np.minimum(0.5, np.exp(lc)), np.exp(li)


def effect_prior(model, cat):
    pe = model["categories"][cat]["product_effects"]
    cov = np.array(pe["cov"])
    i, j = pe["names"].index("conv"), pe["names"].index("impr")
    v = cov[i, i] + cov[j, j] + 2 * cov[i, j]
    return 1.0 / (math.exp(v) - 1.0)


# ----------------------------------------------------------------------------------------------- v1-style specification
def v1_rating_c(d, prior_all, prior_k):
    """Rating shown at the start of the week, but shrunk towards v1's full-sample prior instead of the training prior."""
    n = np.expm1(d.logrev.values)
    pk = d.category.map(prior_k).values
    return d.rating_c.values + (prior_all - pk) * F.SHRINK_M / (n + F.SHRINK_M)


def v1_style(train_fr, prior_all, prior_k):
    """The v1 specification refitted on the same training rows, per category, for comparison only.

    v1 used this week's market median and demand (not known when the week starts) and a rating prior taken from the
    final ratings of ALL listings (including evaluation listings, and their future). No tier, subcategory or attribute terms.
    """
    out = {}
    for cat, d in train_fr.groupby("category"):
        d = d.copy()
        d["rel_c"] = d.selling_price / d.platform_median_price - 1
        d["rel_c2"] = d.rel_c ** 2
        d["log_di_c"] = np.log(d.category_demand_index)
        d["rating_c"] = v1_rating_c(d, prior_all, prior_k)
        m = {}
        tot = d.groupby("product_id").orders_placed.sum()
        dc = d[(d.impressions > 0) & d.product_id.isin(tot[tot >= 5].index)].reset_index(drop=True)
        Xc = dc[["rel_c", "rel_c2", "rating_c", "logrev", "festival_week"]].astype(float)
        rc, cc, fe = T.fe_glm(dc.orders_placed, Xc, dc.product_id, sm.families.Poisson(), np.log(dc.impressions))
        lqp = dc.groupby("product_id").lq.first().loc[fe.index]
        s2 = sm.OLS(fe.values, sm.add_constant(lqp.values)).fit()
        m["conv"] = dict(coef={c: float(rc.params[c]) for c in cc}, a=float(s2.params[0]), b_lq=float(s2.params[1]))
        dv = d[d.in_stock_share > 0].reset_index(drop=True)
        Xi = dv[["ph12", "ph34", "logrev", "rating_c", "convratio", "log_di_c"]].astype(float)
        ri, ci, fei = T.fe_glm(dv.impressions, Xi, dv.product_id, sm.families.Poisson(), np.log(dv.in_stock_share))
        lqi = dv.groupby("product_id").lq.first().loc[fei.index]
        s2i = sm.OLS(fei.values, sm.add_constant(lqi.values)).fit()
        m["impr"] = dict(coef={c: float(ri.params[c]) for c in ci}, a=float(s2i.params[0]), b_lq=float(s2i.params[1]))
        m["conv_ref"] = float(d.orders_placed.sum() / d.impressions.sum())
        out[cat] = m
    out["prior_all"], out["prior_k"] = prior_all, prior_k
    return out


def v1_teacher_forced(v1, fr):
    pred = np.zeros(len(fr))
    rating_c = v1_rating_c(fr, v1["prior_all"], v1["prior_k"])
    for cat in fr.category.unique():
        idx = (fr.category == cat).values
        d, m = fr[idx], v1[cat]
        rel_c = d.selling_price / d.platform_median_price - 1
        rc_ = rating_c[idx]
        cv, iv = m["conv"]["coef"], m["impr"]["coef"]
        lc = m["conv"]["a"] + m["conv"]["b_lq"] * d.lq + cv["rel_c"] * rel_c + cv["rel_c2"] * rel_c ** 2 + cv["rating_c"] * rc_ \
            + cv["logrev"] * d.logrev + cv["festival_week"] * d.festival_week
        li = m["impr"]["a"] + m["impr"]["b_lq"] * d.lq + iv["ph12"] * d.ph12 + iv["ph34"] * d.ph34 + iv["logrev"] * d.logrev \
            + iv["rating_c"] * rc_ + iv["convratio"] * d.convratio + iv["log_di_c"] * np.log(d.category_demand_index)
        pred[idx] = np.exp(li + lc).values
    return pred


def v1_recursive(m, lq, prices, medians, demand, rating_prior, review_rate, keep):
    """v1's deterministic simulation: expected path, rating fixed at the category prior, no listing uncertainty."""
    cv, iv = m["conv"]["coef"], m["impr"]["coef"]
    cum_rev, ema, out = 0.0, None, []
    for w, (p, med) in enumerate(zip(prices, medians), start=1):
        rel = p / med - 1
        shown = (rating_prior * cum_rev + rating_prior * F.SHRINK_M) / (cum_rev + F.SHRINK_M)
        rc, lr = shown - 4, math.log1p(cum_rev)
        cr = 0.0 if ema is None else max(-1, min(3, ema / m["conv_ref"] - 1))
        impr = math.exp(m["impr"]["a"] + m["impr"]["b_lq"] * lq + iv["ph12"] * (w <= 2) + iv["ph34"] * (3 <= w <= 4)
                        + iv["logrev"] * lr + iv["rating_c"] * rc + iv["convratio"] * cr + iv["log_di_c"] * math.log(demand))
        conv = math.exp(m["conv"]["a"] + m["conv"]["b_lq"] * lq + cv["rel_c"] * rel + cv["rel_c2"] * rel * rel
                        + cv["rating_c"] * rc + cv["logrev"] * lr)
        o = impr * conv
        out.append(o)
        cum_rev += review_rate * o * keep
        ema = conv if ema is None else 0.6 * ema + 0.4 * conv
    return out


# ----------------------------------------------------------------------------------------------- engine runs
def run_engine(tasks, model_path, tmp):
    spec = os.path.join(tmp, "tasks.json")
    res = os.path.join(tmp, "results.json")
    with open(spec, "w") as f:
        json.dump(dict(model=model_path, tasks=tasks), f)
    subprocess.run(["node", os.path.join(HERE, "eval_runner.js"), spec, res], check=True)
    return {r["id"]: r for r in json.load(open(res))}


def engine_input(prow, costs=True):
    d = dict(category=prow.category, subcategory=prow.subcategory, tier=prow.positioning_tier,
             attribute=str(prow.attribute_value), listingScore=float(prow.listing_quality_score),
             deliveryDays=int(prow.delivery_promise_days_base))
    if costs:
        d.update(product=float(prow.seller_unit_cost_inr), packaging=float(prow.seller_packaging_inr),
                 gift=float(prow.seller_free_gift_inr))
    return d


# ----------------------------------------------------------------------------------------------- truth (generator) for decisions
_TRUTH = {}


def _truth_init(seed):
    rng = np.random.default_rng(seed)
    _TRUTH["market"] = smk.Market(rng)
    _TRUTH["A"] = econ.load_assumptions()


def _truth_job(job):
    prow, plans, reps = job
    prod = smk.product_from_row(prow)
    budget = prod["private"]["budget"]
    out = {}
    base = abs(hash(prod["obs"]["product_id"])) % (2 ** 31)
    for key, plan in plans.items():
        tot, mins, orders = [], [], []
        for r in range(reps):
            s = smk.simulate_plan(prod, plan, _TRUTH["market"], DIAG_LAUNCH, _TRUTH["A"], seed=(int(prow["_seed"]) * 1000 + r))
            tot.append(s["cum"][-1])
            mins.append(min(0.0, min(s["cum"])))
            orders.append(sum(s["orders"]))
        tot, mins = np.array(tot), np.array(mins)
        out[key] = dict(mean=float(tot.mean()), se=float(tot.std(ddof=1) / math.sqrt(reps)), p_loss=float((tot < 0).mean()),
                        p_budget=float((mins < -budget).mean()), orders=float(np.mean(orders)),
                        per_order=float(tot.sum() / max(1, sum(orders))))
    return prow["product_id"], out


# ----------------------------------------------------------------------------------------------- main evaluation
def evaluate(data_dir, out_dir, quick=False):
    os.makedirs(out_dir, exist_ok=True)
    P, W, Mk, R, manifest = T.load(data_dir)
    A = econ.load_assumptions()
    split = split_products(P)
    P["split"] = P.product_id.map(split)
    W["split"] = W.product_id.map(split)
    train_ids = sorted(P[P.split == "train"].product_id)
    model = T.fit(P, W, Mk, R, manifest, train_ids=train_ids, cutoff=T0)
    tmp = tempfile.mkdtemp(prefix="pc_eval_")
    model_path = os.path.join(tmp, "model_eval.json")
    with open(model_path, "w") as f:
        json.dump(model, f)
    report = dict(
        provenance="SYNTHETIC evaluation. Metrics measure recovery of the generator's assumptions, not real-world accuracy.",
        design=dict(train_cutoff=T0, split_seed=SPLIT_SEED, split_counts=P.split.value_counts().to_dict(),
                    diagnostic_launch=DIAG_LAUNCH.isoformat(), truth_reps=TRUTH_REPS, generator_version=manifest.get("generator_version"),
                    seed=manifest.get("seed"), quick=quick, tuning="none: no parameter was chosen using validation or holdout results"))

    # ---------- frames with history-only features, built with TRAINING constants
    frames = []
    for cat in sorted(P.category.unique()):
        C = model["categories"][cat]
        k = dict(prior_rating=C["constants"]["prior_rating"], conv_ref=C["constants"]["conv_ref"],
                 lq_mean=C["constants"]["lq_mean"], lq_sd=C["constants"]["lq_sd"])
        Pc, Wc, Mc = P[P.category == cat], W[W.category == cat], Mk[Mk.category == cat]
        fr = F.build_frame(Wc, Pc, Mc, k)
        fr = fr.merge(Mc[["subcategory", "week_start", "platform_median_price", "category_demand_index"]],
                      on=["subcategory", "week_start"], how="left")
        mu, mu_i = expected_orders(model, cat, fr)
        fr["mu"] = mu
        a = effect_prior(model, cat)
        fr = fr.sort_values(["product_id", "week_num"]).reset_index(drop=True)
        g = fr.groupby("product_id")
        sy = g.orders_placed.cumsum() - fr.orders_placed
        smu = g.mu.cumsum() - fr.mu
        fr["mhat"] = (a + sy) / (a + smu)
        fr["mshape"] = a + sy
        fr["mrate"] = a + smu
        fr["prev_orders"] = g.orders_placed.shift(1)
        disp = C["conv"]["dispersion"] + C["impr"]["dispersion"]
        fr["disp"] = disp
        frames.append(fr)
    fr = pd.concat(frames, ignore_index=True)
    fr = fr.merge(P[["product_id", "hidden_archetype", "hidden_seller_strategy", "launch_week"]], on="product_id", how="left")

    # ---------- baselines fitted on training rows only
    tr = fr[(fr.split == "train") & (fr.week_start < T0)]
    cat_week = tr.assign(wk=tr.week_num.clip(upper=20)).groupby(["category", "wk"]).orders_placed.mean()
    Xp = lambda d: pd.concat([pd.get_dummies(d.category, prefix="cat", dtype=float).reindex(columns=[f"cat_{c}" for c in sorted(P.category.unique())], fill_value=0.0),
                              d[["ph12", "ph34", "rel", "rel2"]].astype(float)], axis=1)
    pooled = sm.GLM(tr.orders_placed, Xp(tr), family=sm.families.Poisson()).fit()
    prior_all = float(P.final_rating.median())        # v1's leak: final ratings of every listing
    v1 = v1_style(tr, prior_all, {c: model["categories"][c]["constants"]["prior_rating"] for c in model["categories"]})

    def add_baselines(d):
        d = d.copy()
        d["pred_cat_week"] = [cat_week.get((c, min(int(w), 20)), np.nan) for c, w in zip(d.category, d.week_num)]
        d["pred_pooled"] = pooled.predict(Xp(d))
        d["pred_v1_tf"] = v1_teacher_forced(v1, d)
        return d

    # ================================================================ Task A: one week ahead
    ta = fr[fr.split.isin(["validation", "holdout"]) & (fr.week_start >= T0) & (fr.week_num >= 2)].copy()
    ta = add_baselines(ta)
    ta["pred_v2_launch"] = ta.mu
    ta["pred_v2_online"] = ta.mu * ta.mhat
    ta["pred_persist"] = ta.prev_orders
    # no-price-change: same model, but last week's price instead of this week's
    prev_sell = fr.groupby("product_id").selling_price.shift(1)
    ta["prev_sell"] = prev_sell.loc[ta.index]
    tmp_fr = ta.copy()
    tmp_fr["rel"] = tmp_fr.prev_sell / tmp_fr.platform_median_price_lag1 - 1
    tmp_fr["rel2"] = tmp_fr.rel ** 2
    tmp_fr["rel_x_premium"] = np.where(tmp_fr.positioning_tier == "premium", tmp_fr.rel, 0)
    tmp_fr["rel_x_basic"] = np.where(tmp_fr.positioning_tier == "basic", tmp_fr.rel, 0)
    tmp_fr["discount_pct"] = 0.0
    mu_np = np.zeros(len(tmp_fr))
    for cat in tmp_fr.category.unique():
        idx = (tmp_fr.category == cat).values
        mu_np[idx] = expected_orders(model, cat, tmp_fr[idx])[0]
    ta["pred_v2_noprice"] = mu_np * ta.mhat.values
    m_mean = ta.mhat
    m_var = ta.mshape / ta.mrate ** 2
    pm = ta.mu * m_mean
    var = pm + ta.mu ** 2 * (m_var + m_mean ** 2 * ta.disp)
    lo, hi = nb_interval(pm.values, var.values)
    ta["lo"], ta["hi"] = lo, hi
    ta["segment"] = price_segment(ta.rel.values)
    preds_a = {"category x week average": "pred_cat_week", "last week's orders": "pred_persist",
               "pooled price response": "pred_pooled", "v1 spec (same-week market, teacher-forced)": "pred_v1_tf",
               "v2, launch-time effect only": "pred_v2_launch", "v2, no price change": "pred_v2_noprice",
               "v2, online listing effect": "pred_v2_online"}
    task_a = {}
    for part in ["validation", "holdout"]:
        d = ta[ta.split == part]
        task_a[part] = dict(overall={k: metrics(d.orders_placed, d[v]) for k, v in preds_a.items()},
                            coverage_80=coverage(d.orders_placed, d.lo, d.hi))
    dh = ta[ta.split == "holdout"]
    keep = {k: preds_a[k] for k in ["category x week average", "pooled price response", "v2, online listing effect"]}
    task_a["holdout_by_category"] = sliced(dh, "orders_placed", keep, "category")
    task_a["holdout_by_archetype"] = sliced(dh, "orders_placed", keep, "hidden_archetype")
    task_a["holdout_by_price_segment"] = sliced(dh, "orders_placed", keep, "segment")
    report["task_a_one_week_ahead"] = task_a

    # ================================================================ Tasks B and C: from launch, no realised outcomes
    launched = P[P.split.isin(["validation", "holdout"]) & (P.launch_week >= T0)].copy()
    if quick:
        launched = launched.groupby(["split", "category"]).head(6)
    mkl = F.lagged_market(Mk)
    tasks, meta = [], {}
    for prow in launched.itertuples():
        rows = W[W.product_id == prow.product_id].sort_values("week_num").head(HORIZON)
        m0 = mkl[(mkl.subcategory == prow.subcategory) & (mkl.week_start == prow.launch_week)].iloc[0]
        base_week = lambda r, price, disc: dict(price=float(price), discount=float(disc), festival=int(r.festival_week),
                                                campaign=int(r.campaign_week), delivery=int(r.delivery_promise_days),
                                                median=float(m0.platform_median_price_lag1), demand=float(m0.category_demand_index_lag1))
        weeks = [base_week(r, r.selling_price, r.discount_pct) for r in rows.itertuples()]
        weeks_np = [base_week(r, rows.iloc[0].listing_price, 0.0) for r in rows.itertuples()]
        inp = engine_input(prow)
        inp["market"] = dict(median=float(m0.platform_median_price_lag1), p25=float(m0.platform_p25_price_lag1),
                             p75=float(m0.platform_p75_price_lag1), demand=float(m0.category_demand_index_lag1))
        tasks.append(dict(id=prow.product_id, kind="forecast", input=inp, weeks=weeks, n=400))
        tasks.append(dict(id=prow.product_id + "|np", kind="forecast", input=inp, weeks=weeks_np, n=400))
        meta[prow.product_id] = dict(rows=rows, m0=m0)
    eng = run_engine(tasks, model_path, tmp)

    recs = []
    C_rr = {c: model["categories"][c]["review_rate"] for c in model["categories"]}
    for prow in launched.itertuples():
        rows, m0 = meta[prow.product_id]["rows"], meta[prow.product_id]["m0"]
        n = len(rows)
        f, fnp = eng[prow.product_id], eng[prow.product_id + "|np"]
        wk = np.array(f["weekly"])
        sub = fr[fr.product_id == prow.product_id].sort_values("week_num").head(HORIZON)
        sub = add_baselines(sub)
        z = (prow.listing_quality_score - model["categories"][prow.category]["constants"]["lq_mean"]) / model["categories"][prow.category]["constants"]["lq_sd"]
        v1p = v1_recursive(v1[prow.category], z, rows.selling_price.values, [m0.platform_median_price_lag1] * n,
                           m0.category_demand_index_lag1, prior_all, C_rr[prow.category], 0.85)
        # totals over the observed weeks (a listing that stopped early is compared over the weeks it was live)
        tot_draws = np.array(f["orderTotals"]) if n == HORIZON else None
        rec = dict(product_id=prow.product_id, split=prow.split, category=prow.category, archetype=prow.hidden_archetype,
                   strategy=prow.hidden_seller_strategy, weeks=n, segment=str(price_segment(np.array([prow.launch_vs_median - 1]))[0]),
                   y1=float(rows.orders_placed.iloc[0]), y4=float(rows.orders_placed.head(4).sum()), y12=float(rows.orders_placed.sum()),
                   e1=float(wk[0, 5]), e4=float(np.median(np.array(f["cum4"]))), e12=float(np.median(np.array(f["cumN"]))),
                   m1=float(wk[0, 0]), m4=float(wk[:min(4, n), 0].sum()), m12=float(wk[:n, 0].sum()),
                   np12=float(np.median(np.array(fnp["cumN"]))),
                   cw1=float(sub.pred_cat_week.iloc[0]), cw4=float(sub.pred_cat_week.head(4).sum()), cw12=float(sub.pred_cat_week.sum()),
                   pl1=float(sub.pred_pooled.iloc[0]), pl4=float(sub.pred_pooled.head(4).sum()), pl12=float(sub.pred_pooled.sum()),
                   v1_1=float(v1p[0]), v1_4=float(sum(v1p[:4])), v1_12=float(sum(v1p)),
                   tf12=float(sub.mu.sum()),
                   contrib=float(rows.seller_contribution_inr.sum()), e_contrib=float(np.median(f["totals"])), m_contrib=float(np.mean(f["totals"])),
                   lo12=float(np.quantile(f["orderTotals"], 0.1)) if n == HORIZON else None,
                   hi12=float(np.quantile(f["orderTotals"], 0.9)) if n == HORIZON else None,
                   c_lo=float(np.quantile(f["totals"], 0.1)) if n == HORIZON else None,
                   c_hi=float(np.quantile(f["totals"], 0.9)) if n == HORIZON else None,
                   p_loss=float(np.mean(np.array(f["totals"]) < 0)) if n == HORIZON else None,
                   pit12=float(np.mean(np.array(f["orderTotals"]) < rows.orders_placed.sum())) if n == HORIZON else None,
                   lo4=float(np.quantile(f["cum4"], 0.1)), hi4=float(np.quantile(f["cum4"], 0.9)))
        recs.append(rec)
    bc = pd.DataFrame(recs)
    task_b, task_c = {}, {}
    for part in ["validation", "holdout"]:
        d = bc[bc.split == part]
        task_b[part] = dict(
            n_listings=int(len(d)),
            week1={"category x week average": metrics(d.y1, d.cw1), "pooled price response": metrics(d.y1, d.pl1),
                   "v1 spec (deterministic path)": metrics(d.y1, d.v1_1), "v2 engine (median)": metrics(d.y1, d.e1),
                   "v2 engine (mean)": metrics(d.y1, d.m1)},
            weeks1_4={"category x week average": metrics(d.y4, d.cw4), "pooled price response": metrics(d.y4, d.pl4),
                      "v1 spec (deterministic path)": metrics(d.y4, d.v1_4), "v2 engine (median)": metrics(d.y4, d.e4),
                      "v2 engine (mean)": metrics(d.y4, d.m4)},
            coverage_80_weeks1_4=coverage(d.y4, d.lo4, d.hi4))
        full = d[d.weeks == HORIZON]
        task_c[part] = dict(
            n_listings=int(len(d)), n_full_12_weeks=int(len(full)),
            orders_observed_weeks={"category x week average": metrics(d.y12, d.cw12), "pooled price response": metrics(d.y12, d.pl12),
                                   "v1 spec (deterministic path)": metrics(d.y12, d.v1_12),
                                   "v2 engine, launch price held (no price change)": metrics(d.y12, d.np12),
                                   "v2 engine, recursive (median)": metrics(d.y12, d.e12),
                                   "v2 engine, recursive (mean)": metrics(d.y12, d.m12),
                                   "v2 teacher-forced (uses realised history)": metrics(d.y12, d.tf12)},
            contribution_observed_weeks=dict(mae=float(np.mean(np.abs(d.e_contrib - d.contrib))),
                                             bias=float(np.mean(d.e_contrib - d.contrib)), bias_of_mean=float(np.mean(d.m_contrib - d.contrib)),
                                             mean_actual=float(d.contrib.mean())),
            coverage_80_orders_12w=coverage(full.y12, full.lo12, full.hi12),
            pit_12w_orders=dict(bins=["0-0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-1"],
                                observed=[int(x) for x in np.histogram(full.pit12.astype(float), bins=[0, .1, .25, .5, .75, .9, 1.0001])[0]],
                                expected=[round(len(full) * w, 1) for w in [.1, .15, .25, .25, .15, .1]]),
            coverage_80_contribution_12w=coverage(full.contrib, full.c_lo, full.c_hi),
            p_loss=dict(mean_predicted=float(full.p_loss.mean()) if len(full) else None,
                        observed_share=float((full.contrib < 0).mean()) if len(full) else None,
                        brier=float(np.mean((full.p_loss - (full.contrib < 0)) ** 2)) if len(full) else None))
    hb = bc[bc.split == "holdout"]
    keep_c = {"category x week average": "cw12", "pooled price response": "pl12", "v2 engine, recursive (median)": "e12"}
    task_c["holdout_by_category"] = sliced(hb, "y12", keep_c, "category")
    task_c["holdout_by_archetype"] = sliced(hb, "y12", keep_c, "archetype")
    task_c["holdout_by_launch_price"] = sliced(hb, "y12", keep_c, "segment")
    task_c["holdout_by_seller_strategy"] = sliced(hb, "y12", keep_c, "strategy")
    report["task_b_launch_time"] = task_b
    report["task_c_recursive_12_weeks"] = task_c

    # ================================================================ parameter recovery (price sensitivity)
    # Compared where the data is: each listing's slope at its own typical (median) observed relative price.
    rec_rows = []
    trP = P[P.split == "train"]
    typ = fr[fr.split == "train"].groupby("product_id").rel.median()
    tl = []
    for _, r in trP.iterrows():
        prod = smk.product_from_row(r)
        rel_m = float(typ.get(r.product_id, 0.0))
        L = prod["latent"]
        rel_f = (1 + rel_m) / L["fair_mult"] - 1
        h = 1e-4
        sl = (smk.price_response(rel_f + h, smk.response_params(prod)) - smk.price_response(rel_f - h, smk.response_params(prod))) / (2 * h) / L["fair_mult"]
        tl.append((sl, rel_m))
    trP = trP.assign(true_slope=[x[0] for x in tl], rel_typ=[x[1] for x in tl])
    for cat, C in model["categories"].items():
        cv = C["conv"]["coef"]
        for tier in ["basic", "standard", "premium"]:
            g = trP[(trP.category == cat) & (trP.positioning_tier == tier)]
            s1 = cv["rel"] + (cv["rel_x_premium"] if tier == "premium" else 0) + (cv["rel_x_basic"] if tier == "basic" else 0)
            fitted = float(np.mean(s1 + 2 * cv["rel2"] * g.rel_typ)) if len(g) else None
            rec_rows.append(dict(category=cat, tier=tier, n=int(len(g)), typical_rel=round(float(g.rel_typ.mean()), 3) if len(g) else None,
                                 true_mean_slope=round(float(g.true_slope.mean()), 2) if len(g) else None,
                                 fitted_slope=None if fitted is None else round(fitted, 2),
                                 true_sd_within_tier=round(float(g.true_slope.std()), 2) if len(g) > 1 else None,
                                 fitted_listing_sd=round(C["price_slope"]["sd"], 2), naive_between_listing=round(C["stats"]["naive_price_coef"], 2)))
    report["parameter_recovery"] = rec_rows

    # ================================================================ decisions judged by the generator
    hold = P[P.split == "holdout"].copy()
    if quick:
        hold = hold.groupby("category").head(5)
    mrow = Mk[Mk.week_start == (DIAG_LAUNCH - timedelta(days=7)).isoformat()].set_index(["category", "subcategory"])
    tasks = []
    for prow in hold.itertuples():
        m = mrow.loc[(prow.category, prow.subcategory)]
        inp = engine_input(prow)
        inp.update(budget=float(prow.seller_loss_budget_inr), lite=True, policy="balanced",
                   market=dict(median=float(m.platform_median_price), p25=float(m.platform_p25_price),
                               p75=float(m.platform_p75_price), demand=float(m.category_demand_index)))
        tasks.append(dict(id=prow.product_id, kind="recommend", input=inp))
    recs = run_engine(tasks, model_path, tmp)
    jobs = []
    hold["_seed"] = np.arange(len(hold)) + 1
    for i, (_, prow) in enumerate(hold.iterrows()):
        r = recs[prow.product_id]
        if "error" in r:
            raise RuntimeError(f"engine failed on {prow.product_id}: {r['error']}")
        g = r["grid"]
        prices = sorted(set(range(g["lo"], g["hi"] + 1, 20)) | {p for p in [r.get("balanced"), r.get("conservative"), r.get("point"),
                                                                          smk.ends_in_9(mrow.loc[(prow.category, prow.subcategory)].platform_median_price)] if p})
        plans = {f"p{p}": [p] * HORIZON for p in prices}
        if r.get("experiment"):
            plans["experiment"] = r["experiment"]["plan"]
        jobs.append((prow, plans, 12 if quick else TRUTH_REPS))
    seed = manifest.get("seed", 20261021)
    with Pool(processes=max(1, min(4, os.cpu_count() or 1)), initializer=_truth_init, initargs=(seed,)) as pool:
        truth = dict(pool.map(_truth_job, jobs, chunksize=1))

    dec = []
    for _, prow in hold.iterrows():
        r, t = recs[prow.product_id], truth[prow.product_id]
        g = r["grid"]
        med = smk.ends_in_9(mrow.loc[(prow.category, prow.subcategory)].platform_median_price)
        capped = {k: v for k, v in t.items() if k.startswith("p") and int(k[1:]) <= g["cap"]}
        orc_key = max(capped, key=lambda k: capped[k]["mean"])
        orc_all = max((k for k in t if k.startswith("p")), key=lambda k: t[k]["mean"])
        best = capped[orc_key]["mean"]
        row = dict(product_id=prow.product_id, category=prow.category, archetype=prow.hidden_archetype, tier=prow.positioning_tier,
                   status=r["status"], learning=(r.get("learning") or {}).get("action"), confident=r["decision"]["confident"],
                   oracle_p=int(orc_key[1:]), oracle_mean=best, oracle_uncapped_p=int(orc_all[1:]),
                   median_p=med, cap=g["cap"], floor=r["floor"])
        # the generator's break-even per placed order, interpolated on the price grid it evaluated
        grid = sorted((int(k[1:]), v["per_order"]) for k, v in t.items() if k.startswith("p") and v["orders"] > 0)
        tbe = None
        for (p0, c0), (p1, c1) in zip(grid, grid[1:]):
            if c0 < 0 <= c1:
                tbe = p0 + (p1 - p0) * (-c0) / (c1 - c0)
                break
        if tbe is None and grid and grid[0][1] >= 0:
            tbe = float(grid[0][0])
        row["true_break_even"] = tbe
        for pol in ["balanced", "conservative", "point", "median"]:
            p = med if pol == "median" else r.get(pol)
            if p is None:
                row[f"{pol}_p"] = None
                continue
            tt = t[f"p{p}"]
            row.update({f"{pol}_p": p, f"{pol}_mean": tt["mean"], f"{pol}_regret": best - tt["mean"], f"{pol}_ploss": tt["p_loss"],
                        f"{pol}_pbudget": tt["p_budget"], f"{pol}_neg_mean": tt["mean"] < 0,
                        f"{pol}_ploss_over_limit": tt["p_loss"] > A["policy"]["max_loss_probability"]})
        if r["status"] == "ok" and r.get("pick"):
            row["engine_ploss"] = r["pick"]["probLoss"]
            row["engine_mean"] = r["pick"]["mean"]
        if "experiment" in t:
            row.update(exp_mean=t["experiment"]["mean"], exp_regret=best - t["experiment"]["mean"], exp_ploss=t["experiment"]["p_loss"])
        # calibration of the engine's loss probability on every price the generator also evaluated
        for c in r["curve"]:
            k = f"p{c[0]}"
            if k in t:
                row.setdefault("_cal", []).append((c[3], t[k]["p_loss"], c[1], t[k]["mean"]))
        dec.append(row)
    D = pd.DataFrame(dec)
    pol_names = {"balanced": "Balanced (v2)", "conservative": "Conservative (v2)", "point": "v1 rule (point estimate, no risk limits)",
                 "median": "Market median, held"}
    # every policy is compared on the SAME listings: those v2 priced (status ok). Listings v2 declined are reported apart.
    common = D[D.status == "ok"]
    dsum = {}
    for pol, name in pol_names.items():
        d = common[common[f"{pol}_p"].notna()]
        rel_reg = d[f"{pol}_regret"] / d.oracle_mean.abs().clip(lower=200)
        dsum[name] = dict(
            n=int(len(d)), mean_regret_inr=float(d[f"{pol}_regret"].mean()), median_regret_inr=float(d[f"{pol}_regret"].median()),
            median_relative_regret=float(rel_reg.median()), share_within_10pct_of_best=float((rel_reg <= 0.10).mean()),
            mean_true_p_loss=float(d[f"{pol}_ploss"].mean()), share_true_mean_negative=float(d[f"{pol}_neg_mean"].mean()),
            share_true_p_loss_over_limit=float(d[f"{pol}_ploss_over_limit"].mean()),
            share_budget_breach_over_10pct=float((d[f"{pol}_pbudget"] > 0.10).mean()),
            mean_price_vs_oracle=float((d[f"{pol}_p"] / d.oracle_p).mean()))
    fl = D[D.floor.notna() & D.true_break_even.notna()]
    floor_acc = dict(n=int(len(fl)), mean_abs_error_inr=float((fl.floor - fl.true_break_even).abs().mean()),
                     mean_abs_error_share=float(((fl.floor - fl.true_break_even).abs() / fl.true_break_even).mean()),
                     mean_error_inr=float((fl.floor - fl.true_break_even).mean()),
                     share_engine_floor_more_than_5pct_below_truth=float((fl.floor < 0.95 * fl.true_break_even).mean()))
    ok_bal = common[common.engine_ploss.notna()]
    pick_cal = dict(n=int(len(ok_bal)), mean_predicted_p_loss=float(ok_bal.engine_ploss.mean()), mean_true_p_loss=float(ok_bal.balanced_ploss.mean()))
    cal = [x for lst in D["_cal"].dropna() for x in lst] if "_cal" in D else []
    cal = pd.DataFrame(cal, columns=["pred_ploss", "true_ploss", "pred_mean", "true_mean"])
    cal["bin"] = pd.cut(cal.pred_ploss, [-0.01, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0])
    cal_tab = cal.groupby("bin", observed=True).agg(n=("pred_ploss", "size"), predicted=("pred_ploss", "mean"),
                                                     observed=("true_ploss", "mean")).reset_index()
    cal_tab["bin"] = cal_tab["bin"].astype(str)
    by_arch = D.groupby("archetype").agg(n=("product_id", "size"), oracle_vs_median=("oracle_p", lambda s: float((s / D.loc[s.index, "median_p"]).mean())),
                                         balanced_vs_median=("balanced_p", lambda s: float((s / D.loc[s.index, "median_p"]).mean())),
                                         balanced_regret=("balanced_regret", "mean"), point_regret=("point_regret", "mean"),
                                         uncertain_share=("confident", lambda s: float((~s.astype(bool)).mean()))).reset_index()
    declined = D[D.status == "no_feasible_candidate"]
    declined_cmp = dict(n=int(len(declined)),
                        v1_rule_priced=int(declined.point_p.notna().sum()),
                        v1_rule_true_mean=float(declined.point_mean.mean()) if declined.point_p.notna().any() else None,
                        v1_rule_true_p_loss=float(declined.point_ploss.mean()) if declined.point_p.notna().any() else None,
                        v1_rule_share_true_mean_negative=float(declined.point_neg_mean.mean()) if declined.point_p.notna().any() else None,
                        oracle_true_mean=float(declined[declined.point_p.notna()].oracle_mean.mean()) if declined.point_p.notna().any() else None)
    report["decisions"] = dict(declined_by_v2=declined_cmp, 
        n_listings=int(len(D)), status=D.status.value_counts().to_dict(), learning_policy=D.learning.value_counts().to_dict(),
        confident_share=float(D.confident.mean()), uncertain_share_priced=float((~common.confident.astype(bool)).mean()),
        common_n=int(len(common)), policies=dsum, floor_accuracy=floor_acc, balanced_pick_p_loss=pick_cal,
        experiment=dict(n_designed=int(D.exp_mean.notna().sum()) if "exp_mean" in D else 0,
                        n_chosen=int((D.learning == "experiment").sum()),
                        mean_regret_inr_chosen=float(D[D.learning == "experiment"].exp_regret.mean()) if "exp_regret" in D else None,
                        mean_true_p_loss_chosen=float(D[D.learning == "experiment"].exp_ploss.mean()) if "exp_ploss" in D else None,
                        mean_regret_inr_all_designed=float(D.exp_regret.mean()) if "exp_regret" in D else None),
        oracle_above_cap_share=float((D.oracle_uncapped_p > D.cap).mean()),
        p_loss_calibration=cal_tab.to_dict(orient="records"),
        mean_contribution_prediction=dict(corr=float(np.corrcoef(cal.pred_mean, cal.true_mean)[0, 1]) if len(cal) > 2 else None,
                                          mean_predicted=float(cal.pred_mean.mean()) if len(cal) else None,
                                          mean_true=float(cal.true_mean.mean()) if len(cal) else None),
        by_archetype=by_arch.to_dict(orient="records"))

    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    D.drop(columns=["_cal"], errors="ignore").to_csv(os.path.join(out_dir, "evaluation_decisions.csv"), index=False)
    bc.to_csv(os.path.join(out_dir, "evaluation_forecasts.csv"), index=False)
    write_markdown(report, os.path.join(out_dir, "evaluation.md"))
    return report


# ----------------------------------------------------------------------------------------------- report
def _fmt(x, kind="num"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if kind == "pct":
        return f"{100 * x:.1f}%"
    if kind == "inr":
        return f"₹{x:,.0f}"
    return f"{x:.2f}" if abs(x) < 100 else f"{x:,.0f}"


def _table(rows, cols):
    out = ["| " + " | ".join(c[0] for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c[1]), c[2]) if len(c) > 2 else str(r.get(c[1], "-")) for c in cols) + " |")
    return "\n".join(out)


def write_markdown(rep, path):
    L = ["# Evaluation report (synthetic data)", "",
         "> **Everything below is measured on synthetic data generated by `synth_market.py`.** The numbers show how well the "
         "pipeline recovers assumptions that were built into the generator. They are not evidence of accuracy on real Meesho "
         "listings, and good synthetic scores must not be quoted as real-world performance.", "",
         f"Design: models trained on the 60% training listings using weeks before **{rep['design']['train_cutoff']}**; "
         f"validation and final holdout listings never enter training. Split counts: {rep['design']['split_counts']}. "
         f"No setting was tuned on validation or holdout results. Quick mode: {rep['design']['quick']}.", ""]
    mcols = [("Method", "name"), ("n", "n"), ("MAE (orders)", "mae", "num"), ("WAPE", "wape", "pct"), ("Bias", "bias", "pct")]

    def mt(d):
        return _table([dict(name=k, **v) for k, v in d.items()], mcols)
    a = rep["task_a_one_week_ahead"]
    L += ["## Task A: one-week-ahead orders", "",
          "Listings in validation / holdout, weeks on or after the cutoff, week 2 onwards. Inputs known at the start of the week only.", ""]
    for part in ["validation", "holdout"]:
        L += [f"**{part.title()}** (80% interval coverage of the v2 online prediction: {_fmt(a[part]['coverage_80'], 'pct')})", "", mt(a[part]["overall"]), ""]
    L += ["Holdout by category, archetype (hidden, evaluation only) and price segment:", ""]
    for key in ["holdout_by_category", "holdout_by_archetype", "holdout_by_price_segment"]:
        rows = []
        for g, d in a[key].items():
            for meth, v in d.items():
                rows.append(dict(group=g, name=meth, **v))
        L += [_table(rows, [("Group", "group"), ("Method", "name"), ("n", "n"), ("MAE", "mae", "num"), ("WAPE", "wape", "pct"), ("Bias", "bias", "pct")]), ""]
    b = rep["task_b_launch_time"]
    L += ["## Task B: new listing, launch-time prediction", "",
          "Listings launched on or after the cutoff. Inputs: launch-time listing facts, the planned weekly prices and calendar, market as of launch.", ""]
    for part in ["validation", "holdout"]:
        L += [f"**{part.title()}** ({b[part]['n_listings']} listings), week 1:", "", mt(b[part]["week1"]), "",
              f"Weeks 1-4 (80% interval coverage {_fmt(b[part]['coverage_80_weeks1_4'], 'pct')}):", "", mt(b[part]["weeks1_4"]), ""]
    c = rep["task_c_recursive_12_weeks"]
    L += ["## Task C: recursive 12-week simulation", "",
          "engine.js simulates each listing from launch with no realised outcomes. Listings that stopped before week 12 are compared over the weeks they were live. "
          "The teacher-forced row uses the realised history of reviews and sales and is shown only to measure how much error recursion adds.", ""]
    for part in ["validation", "holdout"]:
        cc = c[part]
        L += [f"**{part.title()}** ({cc['n_listings']} listings, {cc['n_full_12_weeks']} with all 12 weeks)", "", mt(cc["orders_observed_weeks"]), "",
              f"- Contribution over the observed weeks (seller's private costs, evaluation only): MAE of the median {_fmt(cc['contribution_observed_weeks']['mae'], 'inr')}, "
              f"bias of the median {_fmt(cc['contribution_observed_weeks']['bias'], 'inr')}, bias of the mean {_fmt(cc['contribution_observed_weeks']['bias_of_mean'], 'inr')} "
              f"(mean actual {_fmt(cc['contribution_observed_weeks']['mean_actual'], 'inr')}).",
              f"- 80% interval coverage, 12-week orders: {_fmt(cc['coverage_80_orders_12w'], 'pct')}; 12-week contribution: {_fmt(cc['coverage_80_contribution_12w'], 'pct')}.",
              f"- Where the actual 12-week orders fall in the simulated distribution (PIT; listings with all 12 weeks), bins {cc['pit_12w_orders']['bins']}: "
              f"observed {cc['pit_12w_orders']['observed']}, expected if calibrated {cc['pit_12w_orders']['expected']}.",
              f"- Probability of a loss: mean predicted {_fmt(cc['p_loss']['mean_predicted'], 'pct')}, observed {_fmt(cc['p_loss']['observed_share'], 'pct')}, Brier {_fmt(cc['p_loss']['brier'])}.", ""]
    for key in ["holdout_by_category", "holdout_by_archetype", "holdout_by_launch_price", "holdout_by_seller_strategy"]:
        rows = []
        for g, d in c[key].items():
            for meth, v in d.items():
                rows.append(dict(group=g, name=meth, **v))
        L += [f"Holdout, {key.replace('holdout_by_', '').replace('_', ' ')}:", "",
              _table(rows, [("Group", "group"), ("Method", "name"), ("n", "n"), ("MAE", "mae", "num"), ("WAPE", "wape", "pct"), ("Bias", "bias", "pct")]), ""]
    L += ["## Price-sensitivity recovery (training listings)", "",
          "Slope = d log(conversion) / d(price / subcategory median), evaluated at each listing's own typical price (where its data is). "
          "True = from the listing's hidden response curve; fitted = the v2 conversion model (tier slope plus curvature). "
          "Naive = the slope a pooled model comparing different listings finds (confounded by quality).", "",
          _table(rep["parameter_recovery"], [("Category", "category"), ("Tier", "tier"), ("n", "n"), ("Typical price vs median", "typical_rel", "num"),
                                             ("True mean", "true_mean_slope", "num"),
                                             ("Fitted", "fitted_slope", "num"), ("True SD in tier", "true_sd_within_tier", "num"),
                                             ("Fitted listing SD", "fitted_listing_sd", "num"), ("Naive", "naive_between_listing", "num")]), ""]
    d = rep["decisions"]
    L += ["## Pricing decisions judged by the generator", "",
          f"{d['n_listings']} holdout listings, each launched at {rep['design']['diagnostic_launch']} in the generator with its hidden "
          f"response curve and the seller's private costs, {rep['design']['truth_reps']} replications per price. Regret is measured against the best "
          f"price at or below the cap. Status counts: {d['status']}. Learning policy: {d['learning_policy']}. Flagged uncertain: "
          f"{_fmt(1 - d['confident_share'], 'pct')} of all listings, {_fmt(d['uncertain_share_priced'], 'pct')} of the {d['common_n']} priced ones. "
          f"Best uncapped price lies above the cap for {_fmt(d['oracle_above_cap_share'], 'pct')} of listings.", "",
          f"**All policies below are compared on the same {d['common_n']} listings** (those v2 priced); the listings v2 declined are reported after the table.", "",
          _table([dict(name=k, **v) for k, v in d["policies"].items()],
                 [("Policy", "name"), ("n", "n"), ("Mean regret", "mean_regret_inr", "inr"), ("Median rel. regret", "median_relative_regret", "pct"),
                  ("Within 10% of best", "share_within_10pct_of_best", "pct"), ("True P(loss)", "mean_true_p_loss", "pct"),
                  ("True mean < 0", "share_true_mean_negative", "pct"), ("True P(loss) > limit", "share_true_p_loss_over_limit", "pct"),
                  ("Budget breach > 10%", "share_budget_breach_over_10pct", "pct")]), "",
          f"Balanced picks: engine-predicted P(loss) {_fmt(d['balanced_pick_p_loss']['mean_predicted_p_loss'], 'pct')} on average, generator {_fmt(d['balanced_pick_p_loss']['mean_true_p_loss'], 'pct')}.", "",
          f"Break-even accuracy ({d['floor_accuracy']['n']} listings): the engine's break-even price is off from the generator's by {_fmt(d['floor_accuracy']['mean_abs_error_inr'], 'inr')} "
          f"on average ({_fmt(d['floor_accuracy']['mean_abs_error_share'], 'pct')}; mean signed error {_fmt(d['floor_accuracy']['mean_error_inr'], 'inr')}); it is more than 5% below the "
          f"generator's for {_fmt(d['floor_accuracy']['share_engine_floor_more_than_5pct_below_truth'], 'pct')} of listings. (With no fixed costs in the generator, a price below "
          "break-even is the same thing as a negative expected contribution, so that column is not repeated.)", "",
          (f"v2 returned 'No feasible candidate under current constraints' for {d['declined_by_v2']['n']} listings. The v1 rule still priced "
           f"{d['declined_by_v2']['v1_rule_priced']} of them; at those prices the generator gives a mean 12-week contribution of "
           f"{_fmt(d['declined_by_v2']['v1_rule_true_mean'], 'inr')}, a true P(loss) of {_fmt(d['declined_by_v2']['v1_rule_true_p_loss'], 'pct')}, and a negative "
           f"expected contribution for {_fmt(d['declined_by_v2']['v1_rule_share_true_mean_negative'], 'pct')} of them (best achievable mean on the same listings "
           f"{_fmt(d['declined_by_v2']['oracle_true_mean'], 'inr')}).") if d['declined_by_v2']['n'] else "v2 priced every holdout listing.", "",
          f"Price tests chosen by the learning policy ({d['experiment']['n_chosen']} listings): mean regret over 12 weeks "
          f"{_fmt(d['experiment']['mean_regret_inr_chosen'], 'inr')}, true P(loss) {_fmt(d['experiment']['mean_true_p_loss_chosen'], 'pct')} "
          f"(over all {d['experiment']['n_designed']} listings with a test design: {_fmt(d['experiment']['mean_regret_inr_all_designed'], 'inr')}). "
          "The value of what the test teaches is not counted.", "",
          "Loss-probability calibration (every candidate price the generator also evaluated):", "",
          _table(d["p_loss_calibration"], [("Predicted P(loss) bin", "bin"), ("n", "n"), ("Mean predicted", "predicted", "pct"), ("Observed", "observed", "pct")]), "",
          "By archetype (hidden, evaluation only):", "",
          _table(d["by_archetype"], [("Archetype", "archetype"), ("n", "n"), ("Best price / median", "oracle_vs_median", "num"),
                                     ("Balanced / median", "balanced_vs_median", "num"), ("Balanced regret", "balanced_regret", "inr"),
                                     ("v1-rule regret", "point_regret", "inr"), ("Flagged uncertain", "uncertain_share", "pct")]), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    rep = evaluate(a.data, a.out, a.quick)
    print(json.dumps(rep["decisions"]["policies"], indent=1))
    print("written", os.path.join(a.out, "evaluation.md"))


if __name__ == "__main__":
    main()
