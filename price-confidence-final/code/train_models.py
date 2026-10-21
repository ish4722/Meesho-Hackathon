"""
Price Confidence - model training v2 (Team Edge Case, Meesho DICE S3).

Fits, per category, on whitelisted and lagged features only (features.py, schema.py):

  conversion  orders per impression. Poisson GLM, product fixed effects, log(impressions) offset.
              Price enters as rel and rel^2 plus rel x tier interactions, so premium and basic listings
              can respond differently. Second stage: product effects on launch-time facts
              (listing quality, tier, subcategory, attribute) give a new listing its starting level.
  visibility  impressions per week. Poisson GLM, product fixed effects, log(in-stock share) offset;
              launch phase, reviews, rating, recent conversion, last week's demand, campaigns, delivery promise.
  returns     return rate: within-listing price effect (fixed effects, constrained >= 0) + launch-time baseline.
  rto, cancel pooled logistic rates by price position and delivery promise.
  stars       star level of new reviews: launch-time baseline + within-listing effect of paying above the market.

and the pieces the risk engine needs to simulate honestly:

  param_blocks      coefficient means + Cholesky factors of their cluster-robust (by listing) covariance
  product_effects   covariance across listings of the unexplained conversion, visibility, return and star
                    effects (sampling noise removed from the diagonal)
  price_slope_sd    spread of listing-specific price sensitivity around the common slope
  dispersion        extra-Poisson week-to-week noise in impressions and in orders given impressions

All data is synthetic, so these models can only rediscover the generator's assumptions. A price
coefficient fitted within listings removes fixed product differences; it does not remove time-varying
confounding (reactive price cuts, campaign timing), so it is a predictive association, not a proven
causal effect.

  python train_models.py                       # all data -> model.json
  python train_models.py --train-ids ids.txt --cutoff 2026-06-01 --out model_eval.json
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import economics as econ
import features as F
import model_schema
import schema

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*divide by zero.*")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
except ImportError:          # older statsmodels
    pass

MODEL_SCHEMA_VERSION = model_schema.MODEL_SCHEMA_VERSION
HERE = os.path.dirname(os.path.abspath(__file__))
MIN_ORDERS_FOR_FE = 5


def load(data_dir):
    rd = lambda n: pd.read_csv(os.path.join(data_dir, n))
    P, W, Mk, R = rd("products.csv"), rd("weekly_panel.csv"), rd("market_weekly.csv"), rd("reviews.csv")
    P["attribute_value"] = P.attribute_value.astype(str)
    man = os.path.join(data_dir, "dataset_manifest.json")
    manifest = json.load(open(man)) if os.path.exists(man) else {}
    return P, W, Mk, R, manifest


def chol_psd(cov):
    cov = np.asarray(cov, float)
    cov = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-10, None)
    fixed = vecs @ np.diag(vals) @ vecs.T
    return np.linalg.cholesky(fixed + 1e-12 * np.eye(len(cov))), fixed


def block(name, names, mean, cov):
    L, fixed = chol_psd(cov)
    return dict(model=name, names=list(names), mean=[float(x) for x in mean], chol=L.tolist(),
                se=[float(np.sqrt(fixed[i, i])) for i in range(len(names))])


def fe_glm(y, X, groups, family, offset=None):
    """GLM with group fixed effects; cluster-robust covariance by group for the shared coefficients."""
    fe = pd.get_dummies(groups, prefix="fe", dtype=float)
    Xf = pd.concat([X.reset_index(drop=True), fe.reset_index(drop=True)], axis=1)
    codes = pd.factorize(groups)[0]
    m = sm.GLM(np.asarray(y), Xf, family=family, offset=None if offset is None else np.asarray(offset))
    r = m.fit(cov_type="cluster", cov_kwds={"groups": codes})
    cols = list(X.columns)
    fes = r.params[[c for c in Xf.columns if c.startswith("fe_")]]
    fes.index = [c[3:] for c in fes.index]
    return r, cols, fes


def pearson_dispersion(y, mu):
    y, mu = np.asarray(y, float), np.asarray(mu, float)
    ok = mu > 0.5
    return float(max(0.0, np.mean(((y[ok] - mu[ok]) ** 2 - mu[ok]) / mu[ok] ** 2)))


def second_stage(fes, noise_var, prod_tab, levels):
    """Regress product effects on launch-time facts. Every listing counts equally (a typical listing).

    Returns coefficients, their covariance, residuals, and the residual SD with sampling noise removed.
    """
    pt = prod_tab.loc[fes.index]
    X = F.second_stage_matrix(pt, levels)
    keep = X.columns[(X != 0).any(axis=0) | (X.columns == "const")]
    X = X[keep]
    r = sm.OLS(fes.values, X).fit(cov_type="HC1")
    resid = pd.Series(fes.values - r.fittedvalues.values, index=fes.index)
    w = 1.0 / (noise_var.loc[fes.index].values + 0.05)
    raw = float(np.average(resid.values ** 2, weights=w))
    sd = float(np.sqrt(max(raw - np.average(noise_var.loc[fes.index].values, weights=w), 0.1 * raw)))
    coef = {c: float(r.params[c]) for c in X.columns}
    return coef, r.cov_params().loc[list(X.columns), list(X.columns)].values, resid, sd


def fit_category(cat, Pc, Wc, Mc, Rc, A, k):
    fr = F.build_frame(Wc, Pc, Mc, k)
    levels = F.levels_for(Pc)
    prod_tab = Pc.set_index("product_id")
    prod_tab["lq"] = (prod_tab.listing_quality_score - k["lq_mean"]) / k["lq_sd"]
    blocks = []

    # ---------------- conversion
    tot_o = fr.groupby("product_id").orders_placed.sum()
    dconv = fr[(fr.impressions > 0) & fr.product_id.isin(tot_o[tot_o >= MIN_ORDERS_FOR_FE].index)].reset_index(drop=True)
    Xc = F.design(dconv, "conversion")
    rc, cc, fes_c = fe_glm(dconv.orders_placed, Xc, dconv.product_id, sm.families.Poisson(), np.log(dconv.impressions))
    phi_c = float(rc.pearson_chi2 / rc.df_resid)
    noise_c = phi_c / dconv.groupby("product_id").orders_placed.sum().clip(lower=1)
    c2, c2cov, res_c, sd_c = second_stage(fes_c, noise_c, prod_tab, levels)
    conv = dict(coef={c: float(rc.params[c]) for c in cc}, second=c2, fe_sd=sd_c,
                dispersion=pearson_dispersion(dconv.orders_placed, rc.fittedvalues), n_products=int(len(fes_c)),
                n_rows=int(len(dconv)))
    blocks.append(block("conv", cc, [rc.params[c] for c in cc], rc.cov_params().loc[cc, cc].values))
    blocks.append(block("conv2", list(c2), list(c2.values()), c2cov))

    # naive (between-listing) slope for the "why fixed effects" comparison
    nv = sm.GLM(dconv.orders_placed, sm.add_constant(dconv[["rel", "discount_pct", "festival_week"]].astype(float)),
                family=sm.families.Poisson(), offset=np.log(dconv.impressions)).fit()

    # ---------------- price-slope heterogeneity (listing-specific deviations from the common slope)
    mu_c = rc.fittedvalues.values
    e = np.log((dconv.orders_placed.values + 0.5) / (mu_c + 0.5))
    slopes, ses = [], []
    for pid, idx in dconv.groupby("product_id").indices.items():
        if len(idx) < 8:
            continue
        x = dconv.rel.values[idx]
        if np.std(x) < 0.03:
            continue
        wts = mu_c[idx] / (1 + conv["dispersion"] * mu_c[idx])
        xm = np.average(x, weights=wts)
        sxx = np.sum(wts * (x - xm) ** 2)
        b = np.sum(wts * (x - xm) * (e[idx] - np.average(e[idx], weights=wts))) / sxx
        slopes.append(b)
        ses.append(1.0 / np.sqrt(sxx))
    slopes, ses = np.array(slopes), np.array(ses)
    # DerSimonian-Laird random-effects estimate of the spread of listing-specific slopes: precise slopes
    # count more, and the expected sampling spread (Q's k-1) is removed before anything is called heterogeneity
    if len(slopes) >= 10:
        wv = 1.0 / ses ** 2
        bbar = np.sum(wv * slopes) / np.sum(wv)
        Q = float(np.sum(wv * (slopes - bbar) ** 2))
        denom = float(np.sum(wv) - np.sum(wv ** 2) / np.sum(wv))
        tau2 = max(0.0, (Q - (len(slopes) - 1)) / denom)
    else:
        tau2 = 0.0
    tau2_data = tau2
    pr = A["modeling"]
    n0, nk = pr["price_slope_prior_listings"], len(slopes)
    tau2 = (nk * tau2_data + n0 * pr["price_slope_prior_sd"] ** 2) / (nk + n0)    # few listings -> close to the prior
    price_slope = dict(sd=float(np.sqrt(tau2)), sd_data=float(np.sqrt(tau2_data)), n_listings=int(nk),
                       method="DerSimonian-Laird, pooled with a prior (assumptions.json: modeling)",
                       raw_sd=float(np.std(slopes)) if len(slopes) else None,
                       median_se=float(np.median(ses)) if len(slopes) else None)

    # ---------------- visibility
    dvis = fr[fr.in_stock_share > 0].reset_index(drop=True)
    Xi = F.design(dvis, "visibility")
    ri, ci, fes_i = fe_glm(dvis.impressions, Xi, dvis.product_id, sm.families.Poisson(), np.log(dvis.in_stock_share))
    phi_i = float(ri.pearson_chi2 / ri.df_resid)
    noise_i = phi_i / dvis.groupby("product_id").impressions.sum().clip(lower=1)
    i2, i2cov, res_i, sd_i = second_stage(fes_i, noise_i, prod_tab, levels)
    impr = dict(coef={c: float(ri.params[c]) for c in ci}, second=i2, fe_sd=sd_i,
                dispersion=pearson_dispersion(dvis.impressions, ri.fittedvalues), n_products=int(len(fes_i)))
    blocks.append(block("impr", ci, [ri.params[c] for c in ci], ri.cov_params().loc[ci, ci].values))
    blocks.append(block("impr2", list(i2), list(i2.values()), i2cov))

    # ---------------- returns (within-listing price effect, then launch-time baseline)
    dret = fr[fr.delivered_orders > 0].reset_index(drop=True)
    Xr = F.design(dret, "returns")
    yb = np.column_stack([dret.customer_returns, dret.delivered_orders - dret.customer_returns])
    has = dret.groupby("product_id").customer_returns.transform("sum") > 0
    rr, cr, _ = fe_glm(pd.DataFrame(yb[has.values]), Xr[has.values].reset_index(drop=True),
                       dret.product_id[has.values].reset_index(drop=True), sm.families.Binomial())
    ret_coef = {c: float(rr.params[c]) for c in cr}
    ret_coef["rel_pos"] = max(0.0, ret_coef["rel_pos"])            # paying more never lowers returns
    off_r = sum(ret_coef[c] * dret[c] for c in cr)
    X2r = F.second_stage_matrix(dret.assign(lq=dret.lq), levels)
    X2r = X2r[X2r.columns[(X2r != 0).any(axis=0) | (X2r.columns == "const")]]
    br = sm.GLM(yb, X2r, family=sm.families.Binomial(), offset=off_r).fit(cov_type="cluster",
                                                                          cov_kwds={"groups": pd.factorize(dret.product_id)[0]})
    p_hat = br.fittedvalues
    agg = pd.DataFrame(dict(pid=dret.product_id, y=dret.customer_returns, n=dret.delivered_orders, ph=p_hat * dret.delivered_orders))
    agg = agg.groupby("pid").sum()
    agg = agg[agg.n >= 30]
    obs = (agg.y + 0.5) / (agg.n + 1)
    pred = (agg.ph / agg.n).clip(1e-4, 1 - 1e-4)
    res_r = np.log(obs / (1 - obs)) - np.log(pred / (1 - pred))
    noise_r = 1.0 / (agg.n * obs * (1 - obs))
    ret_sd = float(np.sqrt(max(np.var(res_r) - np.mean(noise_r), 0.1 * np.var(res_r))))
    returns = dict(coef=ret_coef, second={c: float(br.params[c]) for c in X2r.columns}, fe_sd=ret_sd)
    blocks.append(block("ret", cr, [ret_coef[c] for c in cr], rr.cov_params().loc[cr, cr].values))
    blocks.append(block("ret2", list(X2r.columns), list(br.params[X2r.columns]),
                        br.cov_params().loc[list(X2r.columns), list(X2r.columns)].values))

    # ---------------- RTO and cancellations (pooled)
    def pooled(dsub, num, den, model):
        Xp = sm.add_constant(F.design(dsub, model))
        y = np.column_stack([dsub[num], dsub[den] - dsub[num]])
        r = sm.GLM(y, Xp, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                              cov_kwds={"groups": pd.factorize(dsub.product_id)[0]})
        names = list(Xp.columns)
        return r, names
    drto = fr[fr.dispatched_orders > 0].reset_index(drop=True)
    rt, nt = pooled(drto, "rto_orders", "dispatched_orders", "rto")
    a2 = pd.DataFrame(dict(pid=drto.product_id, y=drto.rto_orders, n=drto.dispatched_orders, ph=rt.fittedvalues * drto.dispatched_orders)).groupby("pid").sum()
    a2 = a2[a2.n >= 30]
    o2 = (a2.y + 0.5) / (a2.n + 1)
    p2 = (a2.ph / a2.n).clip(1e-4, 1 - 1e-4)
    rres = np.log(o2 / (1 - o2)) - np.log(p2 / (1 - p2))
    rto_sd = float(np.sqrt(max(np.var(rres) - np.mean(1 / (a2.n * o2 * (1 - o2))), 0.1 * np.var(rres))))
    rto = dict(coef={c: float(rt.params[c]) for c in nt}, fe_sd=rto_sd)
    blocks.append(block("rto", nt, [rt.params[c] for c in nt], rt.cov_params().loc[nt, nt].values))
    dcan = fr[fr.orders_placed > 0].reset_index(drop=True)
    rn, nn = pooled(dcan, "cancelled_orders", "orders_placed", "cancel")
    cancel = dict(coef={c: float(rn.params[c]) for c in nn})
    blocks.append(block("cancel", nn, [rn.params[c] for c in nn], rn.cov_params().loc[nn, nn].values))

    # ---------------- stars of new reviews
    dst = fr[fr.new_reviews > 0].copy()
    wts = dst.new_reviews.astype(float)
    dm = lambda s: s - (s * wts).groupby(dst.product_id).transform("sum") / wts.groupby(dst.product_id).transform("sum")
    yy, xx = dm(dst.new_reviews_avg_stars), dm(dst.rel_pos)
    b_rel = float(np.sum(wts * xx * yy) / max(np.sum(wts * xx * xx), 1e-9))
    b_rel = min(0.0, b_rel)                                         # paying more never improves stars
    se_rel = float(np.sqrt(np.sum(wts * (yy - b_rel * xx) ** 2) / max(np.sum(wts) - 1, 1) / max(np.sum(wts * xx * xx), 1e-9)))
    star_sd_review = float(Rc.groupby("product_id").stars.std().median()) if len(Rc) else 0.9
    ps = dst.assign(adj=dst.new_reviews_avg_stars - b_rel * dst.rel_pos).groupby("product_id").apply(
        lambda d: pd.Series(dict(m=np.average(d.adj, weights=d.new_reviews), n=d.new_reviews.sum())), include_groups=False)
    ps = ps[ps.n >= 5]
    s2, s2cov, res_s, sd_s = second_stage(ps.m, star_sd_review ** 2 / ps.n, prod_tab, levels)
    stars = dict(coef=dict(rel_pos=b_rel), second=s2, fe_sd=sd_s, sd_review=star_sd_review)
    blocks.append(block("stars", ["rel_pos"], [b_rel], [[se_rel ** 2]]))
    blocks.append(block("stars2", list(s2), list(s2.values()), s2cov))

    # ---------------- covariance of unexplained product effects
    effects = pd.concat(dict(conv=res_c, impr=res_i, ret=res_r, stars=res_s), axis=1).dropna()
    names = ["conv", "impr", "ret", "stars"]
    if len(effects) >= 15:
        cov = np.cov(effects[names].values.T)
        target = dict(conv=sd_c, impr=sd_i, ret=ret_sd, stars=sd_s)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        sds = np.array([target[n] for n in names])
        cov = corr * np.outer(sds, sds)                              # noise-corrected SDs, observed correlations
    else:
        cov = np.diag([sd_c ** 2, sd_i ** 2, ret_sd ** 2, sd_s ** 2])
    L, fixed = chol_psd(cov)
    product_effects = dict(names=names, cov=fixed.tolist(), chol=L.tolist(), n_listings=int(len(effects)),
                           corr=(fixed / np.outer(np.sqrt(np.diag(fixed)), np.sqrt(np.diag(fixed)))).round(3).tolist())

    # ---------------- market as of the last data week, per subcategory
    subcats = {}
    for sub, m in Mc.groupby("subcategory"):
        m = m.sort_values("week_start")
        last8 = m.tail(8)
        age = (pd.to_datetime(m.week_start.max()) - pd.to_datetime(last8.week_start)).dt.days / 7
        wt = 0.5 ** (age / 2.0)
        wavg = lambda col: float(np.average(last8[col], weights=wt))
        subcats[sub] = dict(market=dict(
            median=round(wavg("platform_median_price")), p25=round(wavg("platform_p25_price")),
            p75=round(wavg("platform_p75_price")),
            median_prev4=round(float(m.iloc[-8:-4].platform_median_price.mean())),
            median_last4=round(float(m.tail(4).platform_median_price.mean())),
            demand_prev4=round(float(m.iloc[-8:-4].category_demand_index.mean()), 3),
            demand_last4=round(float(m.tail(4).category_demand_index.mean()), 3),
            as_of=str(m.week_start.max())), n_listings=int((Pc.subcategory == sub).sum()))

    # ---------------- complaints by price position (descriptive)
    rv = Rc[Rc.sentiment == "negative"].merge(fr[["product_id", "week_start", "rel"]], on=["product_id", "week_start"])
    labels = ["more than 5% below median", "within 5% of median", "5-15% above median", "more than 15% above median"]
    rv["bucket"] = pd.cut(1 + rv.rel, [0, 0.95, 1.05, 1.15, 10], labels=labels)
    complaints = []
    for b in labels:
        t = rv[rv.bucket == b].theme
        complaints.append(dict(bucket=b, price_share=round(float((t == "price").mean()), 3) if len(t) else None, n=int(len(t))))

    # ---------------- similar-listing cohorts (descriptive: realised outcomes of every listing)
    first12 = fr[fr.week_num <= 12].assign(kxp=lambda d: d.kept_orders * d.selling_price)
    agg = first12.groupby("product_id").agg(kxp=("kxp", "sum"), orders=("orders_placed", "sum"),
                                            dispatched=("dispatched_orders", "sum"), rto=("rto_orders", "sum"),
                                            returns=("customer_returns", "sum"), kept=("kept_orders", "sum"),
                                            resold=("returns_resold", "sum"), weeks=("week_num", "max"))
    last_week = pd.to_datetime(Wc.week_start).max()
    cohorts = []
    for r in Pc.merge(agg, left_on="product_id", right_index=True).itertuples():
        eligible = pd.to_datetime(r.launch_week) <= last_week - pd.Timedelta(weeks=11)
        cohorts.append(dict(id=r.product_id, sub=r.subcategory, tier=r.positioning_tier, pos=r.launch_position,
                            lvm=round(r.launch_vs_median, 3), eligible=bool(eligible), survived12=bool(eligible and r.weeks >= 12),
                            rating=None if pd.isna(r.final_rating) else round(float(r.final_rating), 2),
                            kxp=round(float(r.kxp)), orders=int(r.orders), dispatched=int(r.dispatched), rto=int(r.rto),
                            returns=int(r.returns), kept=int(r.kept), resold=int(r.resold)))

    attr_freq = Pc.attribute_value.value_counts(normalize=True).round(4).to_dict()
    # observed price range (relative to the lagged subcategory median) per tier: outside it the price
    # curve is extrapolated conservatively by the engine and candidates are flagged
    support = {t: [round(float(g.rel.quantile(0.01)), 3), round(float(g.rel.quantile(0.99)), 3)]
               for t, g in fr.groupby("positioning_tier")}
    support["all"] = [round(float(fr.rel.quantile(0.01)), 3), round(float(fr.rel.quantile(0.99)), 3)]
    # range of the dynamic state seen in training: the engine keeps simulated listings inside it, so the
    # log-linear models are not extrapolated into feedback loops the data never showed
    state_ranges = {c: [round(float(fr[c].quantile(0.01)), 3), round(float(fr[c].quantile(0.99)), 3)]
                    for c in ("convratio", "logrev", "rating_c")}
    return dict(
        levels=levels, attr_freq=attr_freq, subcats=subcats, price_support=support, state_ranges=state_ranges,
        in_stock_mean=round(float(fr.in_stock_share.mean()), 4),
        constants=dict(prior_rating=k["prior_rating"], conv_ref=k["conv_ref"], lq_mean=k["lq_mean"], lq_sd=k["lq_sd"]),
        rating=dict(prior=round(k["prior_rating"], 3), weak=round(k["rating_weak"], 2),
                    expected=round(k["rating_expected"], 2), good=round(k["rating_good"], 2)),
        conv=conv, impr=impr, returns=returns, rto=rto, cancel=cancel,
        resale=float(fr.returns_resold.sum() / max(1, fr.customer_returns.sum())),
        review_rate=float(fr.new_reviews.sum() / max(1, (fr.kept_orders + fr.customer_returns).sum())),
        stars=stars, product_effects=product_effects, price_slope=price_slope, param_blocks=blocks,
        stats=dict(naive_price_coef=float(nv.params["rel"]), within_price_coef=float(rc.params["rel"]),
                   listings=int(len(Pc)), rows=int(len(fr)), conv_listings=int(len(fes_c))),
        complaints=complaints, cohorts=cohorts,
    )


def fit(P, W, Mk, R, manifest=None, train_ids=None, cutoff=None, categories=None):
    """Fit every category. Restricting to train_ids and weeks before cutoff gives an evaluation model."""
    import synth_market as smk
    A = econ.load_assumptions()
    if train_ids is not None:
        P = P[P.product_id.isin(train_ids)]
        W = W[W.product_id.isin(train_ids)]
        R = R[R.product_id.isin(train_ids)]
    if cutoff is not None:
        W = W[W.week_start < cutoff]
        R = R[R.week_start < cutoff]
        Mk = Mk[Mk.week_start < cutoff]
        P = P[P.product_id.isin(W.product_id.unique())]
    out = dict(model_schema_version=MODEL_SCHEMA_VERSION,
               meta=dict(synthetic=True, provenance=(manifest or {}).get("provenance", "SYNTHETIC"),
                         generator_version=(manifest or {}).get("generator_version"), seed=(manifest or {}).get("seed"),
                         scenario=(manifest or {}).get("scenario"), config_sha256=(manifest or {}).get("config_sha256"),
                         listings=int(len(P)), listing_weeks=int(len(W)), reviews=int(len(R)),
                         first_launch=str(P.launch_week.min()), data_to=str(W.week_start.max()),
                         train_cutoff=cutoff, n_train_ids=None if train_ids is None else int(len(set(train_ids))),
                         shrink_m=F.SHRINK_M, horizon_weeks=A["policy"]["horizon_weeks"],
                         feature_whitelist=schema.MODEL_FEATURES, schema_version=schema.SCHEMA_VERSION),
               assumptions=A, categories={})
    for cat in (categories or sorted(P.category.unique())):
        Pc, Wc, Rc = P[P.category == cat], W[W.category == cat], R[R.category == cat]
        Mc = Mk[Mk.category == cat]
        k = F.fit_constants(Wc, Pc)
        m = fit_category(cat, Pc, Wc, Mc, Rc, A, k)
        cfg = smk.CATEGORIES[cat]
        m.update(label=cfg["label"], plural=cfg["plural"], group=cfg["group"], keywords=cfg["keywords"],
                 attr_name=cfg["attribute"]["name"], ship=A["economics"]["forward_shipping_inr"][cat],
                 subcat_labels={s: v["label"] for s, v in cfg["subcats"].items()})
        for s in m["subcats"]:
            m["subcats"][s]["label"] = cfg["subcats"][s]["label"]
        out["categories"][cat] = m
    model_schema.validate(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--train-ids", help="file with one product_id per line (default: all)")
    ap.add_argument("--cutoff", help="only use weeks before this date (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    P, W, Mk, R, manifest = load(a.data)
    ids = None
    if a.train_ids:
        ids = [x.strip() for x in open(a.train_ids) if x.strip()]
    out = fit(P, W, Mk, R, manifest, ids, a.cutoff)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    for cat, m in out["categories"].items():
        c = m["conv"]["coef"]
        print(f"{cat:9s} rel {c['rel']:+.2f} rel2 {c['rel2']:+.2f} x_prem {c['rel_x_premium']:+.2f} x_basic {c['rel_x_basic']:+.2f}"
              f" | naive {m['stats']['naive_price_coef']:+.2f} | slope sd {m['price_slope']['sd']:.2f} (n={m['price_slope']['n_listings']})"
              f" | effects sd conv {m['conv']['fe_sd']:.2f} impr {m['impr']['fe_sd']:.2f} | disp o {m['conv']['dispersion']:.3f} i {m['impr']['dispersion']:.3f}")
    print(a.out, "written,", round(os.path.getsize(a.out) / 1024), "KB")


if __name__ == "__main__":
    main()
