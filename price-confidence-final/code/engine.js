/*
 * Price Confidence - risk-aware pricing engine v2 (Team Edge Case, Meesho DICE S3)
 *
 * Pure functions over model.json (written by train_models.py). Runs in the browser app and under node
 * (tests/engine.test.js, eval_runner.js). No network, no storage, no Math.random: every simulation is seeded.
 *
 * ALL MODELS ARE FITTED ON SYNTHETIC DATA. Outputs show how the method reasons under stated assumptions;
 * they are not forecasts for real Meesho sellers, and the price effects are predictive associations,
 * not proven causal effects.
 *
 *   view            resolve one listing: category, subcategory, tier, attribute, listing quality, costs, market
 *   perOrder        expected money from one placed order at a price (mirrors economics.py)
 *   breakEven       lowest price whose expected contribution per placed order is >= 0 (NOT a guarantee of no loss)
 *   simulate        Monte Carlo of weekly impressions -> orders -> cancellations / RTO / returns -> reviews,
 *                   with parameter uncertainty, listing-level uncertainty and weekly noise; common random
 *                   numbers across candidate prices so differences between prices are not noise
 *   recommend       candidate grid, risk metrics per candidate, three policies (balanced, conservative,
 *                   learning), boundary flags, infeasibility reasons, strategy comparison
 */
(function (root) {
  "use strict";

  var ENGINE_VERSION = "2.0.0";
  var MODEL_SCHEMA_VERSION = "2.0";

  // ------------------------------------------------------------------ helpers
  var ends9 = function (x) { return Math.max(49, Math.round(x / 10) * 10 - 1); };
  var sig = function (x) { return 1 / (1 + Math.exp(-x)); };
  var clamp = function (x, a, b) { return Math.min(b, Math.max(a, x)); };
  function quantile(arr, q) {
    if (!arr.length) return null;
    var s = Array.prototype.slice.call(arr).sort(function (a, b) { return a - b; });
    var pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }
  function mean(arr) { var t = 0; for (var i = 0; i < arr.length; i++) t += arr[i]; return arr.length ? t / arr.length : null; }
  function share(arr, f) { var n = 0; for (var i = 0; i < arr.length; i++) if (f(arr[i], i)) n++; return arr.length ? n / arr.length : null; }
  var median = function (a) { return quantile(a, 0.5); };

  // ------------------------------------------------------------------ seeded randomness
  function hash3(a, b, c) {
    var h = (a >>> 0) ^ 0x9E3779B9;
    h = Math.imul(h ^ (b >>> 0), 0x85EBCA6B) >>> 0; h ^= h >>> 13;
    h = Math.imul(h ^ (c >>> 0), 0xC2B2AE35) >>> 0; h ^= h >>> 16;
    return h >>> 0;
  }
  function rng(seed) {                       // mulberry32, values strictly inside (0, 1)
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
    };
  }
  function qnorm(p) {                        // Acklam's inverse normal CDF
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    var q, r;
    if (p < 0.02425) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  // Each sampler uses exactly one uniform, so every simulated week consumes the same number of draws and
  // candidate prices see the same random events (common random numbers).
  function poisson(lam, u) {
    if (!(lam > 0)) return 0;
    if (lam > 40) return Math.max(0, Math.floor(lam + Math.sqrt(lam) * qnorm(u) + 0.5));
    var p = Math.exp(-lam), F = p, k = 0;
    while (u > F && k < 400) { k++; p *= lam / k; F += p; }
    return k;
  }
  function binom(n, p, u) {
    if (n <= 0 || p <= 0) return 0;
    if (p >= 1) return n;
    if (p > 0.5) return n - binom(n, 1 - p, 1 - u);
    var m = n * p;
    if (m > 25 && n * (1 - p) > 25) return clamp(Math.floor(m + Math.sqrt(m * (1 - p)) * qnorm(u) + 0.5), 0, n);
    var q = 1 - p, pr = Math.pow(q, n), F = pr, k = 0, ratio = p / q;
    if (pr < 1e-280) return clamp(Math.floor(m + Math.sqrt(m * q) * qnorm(u) + 0.5), 0, n);
    while (u > F && k < n) { pr *= ratio * (n - k) / (k + 1); k++; F += pr; }
    return k;
  }
  function gammaMult(d, u) {                 // mean-1 gamma multiplier with variance d (Wilson-Hilferty)
    if (!(d > 1e-6)) return 1;
    var c = d / 9, v = 1 - c + qnorm(u) * Math.sqrt(c);
    return v <= 0 ? 0 : v * v * v;
  }

  // ------------------------------------------------------------------ model checks
  function validateModel(M) {
    var need = function (o, keys, where) {
      if (!o || typeof o !== "object") throw new Error(where + ": missing");
      keys.forEach(function (k) { if (!(k in o)) throw new Error(where + ": missing " + k); });
    };
    need(M, ["model_schema_version", "meta", "assumptions", "categories"], "model");
    if (M.model_schema_version !== MODEL_SCHEMA_VERSION) throw new Error("model schema " + M.model_schema_version + " is not " + MODEL_SCHEMA_VERSION);
    if (M.meta.synthetic !== true) throw new Error("model.meta.synthetic must be true");
    need(M.assumptions, ["economics", "policy"], "assumptions");
    Object.keys(M.categories).forEach(function (cat) {
      var C = M.categories[cat], w = "categories." + cat;
      need(C, ["levels", "subcats", "constants", "conv", "impr", "returns", "rto", "cancel", "stars", "product_effects",
               "price_slope", "param_blocks", "price_support", "in_stock_mean", "state_ranges", "resale", "review_rate", "rating"], w);
      ["conv", "conv2", "impr", "impr2", "ret", "ret2", "rto", "cancel", "stars", "stars2"].forEach(function (b) {
        var blk = C.param_blocks.filter(function (x) { return x.model === b; })[0];
        if (!blk) throw new Error(w + ".param_blocks: missing " + b);
        if (blk.names.length !== blk.mean.length || blk.chol.length !== blk.names.length) throw new Error(w + ".param_blocks." + b + ": sizes differ");
      });
      C.levels.sub.forEach(function (s) { need(C.subcats[s] && C.subcats[s].market, ["median", "p25", "p75"], w + ".subcats." + s + ".market"); });
    });
    return true;
  }

  // ------------------------------------------------------------------ listing view
  function detectCategory(M, title) {
    var t = (title || "").toLowerCase(), best = null;
    Object.keys(M.categories).forEach(function (cat) {
      (M.categories[cat].keywords || []).forEach(function (k) {
        var kk = k.toLowerCase(), re = new RegExp("(^|[^a-z])" + kk.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "($|[^a-z])");
        if (!best && (/[a-z]/.test(kk) ? re.test(t) : t.indexOf(kk) >= 0)) best = cat;
      });
    });
    return best;
  }
  function listingScore(checks) { return 50 + 7.5 * clamp(checks, 0, 4); }     // 50..80, inside the data range

  function view(M, input) {
    var cat = input.category, C = M.categories[cat];
    if (!C) throw new Error("unknown category " + cat);
    var sub = input.subcategory && C.subcats[input.subcategory] ? input.subcategory :
      C.levels.sub.slice().sort(function (a, b) { return C.subcats[b].n_listings - C.subcats[a].n_listings; })[0];
    var tier = C.levels.tier.indexOf(input.tier) >= 0 ? input.tier : "standard";
    var attr = input.attribute !== undefined && input.attribute !== null && C.levels.attr.indexOf(String(input.attribute)) >= 0 ? String(input.attribute) :
      Object.keys(C.attr_freq).sort(function (a, b) { return C.attr_freq[b] - C.attr_freq[a]; })[0];
    var mk = {}, base = C.subcats[sub].market;
    Object.keys(base).forEach(function (k) { mk[k] = base[k]; });
    if (input.market) Object.keys(input.market).forEach(function (k) { if (input.market[k] !== undefined && input.market[k] !== null) mk[k] = input.market[k]; });
    if (mk.demand === undefined) mk.demand = mk.demand_last4;
    var score = input.listingScore !== undefined && input.listingScore !== null ? +input.listingScore : listingScore(input.checks || 0);
    var z = (score - C.constants.lq_mean) / C.constants.lq_sd;
    var x2 = { "const": 1, lq: z };
    x2["tier:" + tier] = 1; x2["sub:" + sub] = 1; x2["attr:" + attr] = 1;
    var costs = { unit: +(input.product !== undefined ? input.product : input.unit) || 0, packaging: +input.packaging || 0, gift: +input.gift || 0 };
    var budgetGiven = input.budget !== null && input.budget !== undefined && input.budget !== "";
    var pol = M.assumptions.policy;
    return {
      M: M, C: C, A: M.assumptions, cat: cat, sub: sub, tier: tier, attr: attr, market: mk, score: score, z: z, x2: x2,
      costs: costs, delivery: input.deliveryDays ? +input.deliveryDays : 4,
      budget: budgetGiven ? Math.max(0, +input.budget || 0) : null,
      launchCost: +input.launchCost || 0, adWeekly: +input.adWeekly || 0,
      support: C.price_support[tier] || C.price_support.all,
      seed: input.seed !== undefined ? input.seed >>> 0 : pol.seed, lite: !!input.lite,
      nSims: input.nSims || pol.n_simulations, uncertaintyScale: input.uncertaintyScale || 1
    };
  }

  // ------------------------------------------------------------------ parameters
  function meanBlocks(C) {
    var P = {};
    C.param_blocks.forEach(function (b) { P[b.model] = {}; b.names.forEach(function (n, i) { P[b.model][n] = b.mean[i]; }); });
    return P;
  }
  function lin(coef, x) { var s = 0; for (var k in coef) if (x[k]) s += coef[k] * x[k]; return s; }
  function bases(V, P) {
    return { conv: lin(P.conv2, V.x2), impr: lin(P.impr2, V.x2), ret: lin(P.ret2, V.x2), stars: lin(P.stars2, V.x2) };
  }
  var ZERO_EFF = { conv: 0, impr: 0, ret: 0, stars: 0, rto: 0 };

  // One draw = plausible model coefficients + this listing's unexplained level and price sensitivity.
  function makeDraws(V, n, seed, opts) {
    opts = opts || {};
    var C = V.C, out = [], pe = C.product_effects, sc = V.uncertaintyScale;
    var tau = C.price_slope.sd * sc, rtoSd = (C.rto.fe_sd || 0) * sc;
    for (var s = 0; s < n; s++) {
      var r = rng(hash3(seed, s, 1)), P = {};
      C.param_blocks.forEach(function (b) {
        var k = b.names.length, zz = new Array(k), o = {};
        for (var i = 0; i < k; i++) zz[i] = opts.params === false ? 0 : qnorm(r());
        for (i = 0; i < k; i++) {
          var v = b.mean[i];
          for (var j = 0; j <= i; j++) v += b.chol[i][j] * zz[j];
          o[b.names[i]] = v;
        }
        P[b.model] = o;
      });
      P.ret.rel_pos = Math.max(0, P.ret.rel_pos);
      P.stars.rel_pos = Math.min(0, P.stars.rel_pos);
      var ze = pe.names.map(function () { return qnorm(r()); }), eff = { rto: rtoSd * qnorm(r()) }, delta = tau * qnorm(r());
      pe.names.forEach(function (nm, i) { var v = 0; for (var j = 0; j <= i; j++) v += pe.chol[i][j] * ze[j]; eff[nm] = v * sc; });
      if (opts.effects === false) { eff = ZERO_EFF; delta = 0; }
      out.push({ P: P, eff: eff, delta: delta, base: bases(V, P), noiseSeed: hash3(seed, s, 2) });
    }
    return out;
  }
  function meanDraw(V) { var P = meanBlocks(V.C); return { P: P, eff: ZERO_EFF, delta: 0, base: bases(V, P), noiseSeed: hash3(V.seed, 0, 2) }; }

  // log-conversion contribution of relative price rel; outside the observed price range the curve is
  // continued in the direction that does NOT reward going further out (no extrapolated gains)
  function priceTerm(V, d, rel) {
    var cv = d.P.conv;
    var s1 = cv.rel + (V.tier === "premium" ? cv.rel_x_premium : 0) + (V.tier === "basic" ? cv.rel_x_basic : 0) + d.delta, s2 = cv.rel2;
    var g = function (r) { return s1 * r + s2 * r * r; }, dg = function (r) { return s1 + 2 * s2 * r; };
    var lo = V.support[0], hi = V.support[1];
    if (rel > hi) return g(hi) + Math.min(dg(hi), 0) * (rel - hi);
    if (rel < lo) return g(lo) + Math.max(dg(lo), 0) * (rel - lo);
    return g(rel);
  }
  function eventRates(V, d, price, median, dd) {
    var P = d.P, rp = Math.max(0, price / median - 1);
    return {
      p_cancel: sig(P.cancel["const"] + P.cancel.rel_pos * rp),
      p_rto: sig(P.rto["const"] + P.rto.rel_pos * rp + P.rto.delivery_days_c * dd + d.eff.rto),
      p_return: sig(d.base.ret + d.eff.ret + P.ret.rel_pos * rp + P.ret.delivery_days_c * dd),
      resale_share: V.C.resale
    };
  }

  // ------------------------------------------------------------------ economics (mirrors economics.py)
  function netPrice(A, price) {
    var e = A.economics;
    return price / (1 + e.gst_rate_included_in_price) * (1 - e.platform_commission_rate - e.payment_fee_rate);
  }
  function weekMoney(V, price, o) {
    var e = V.A.economics, ship = e.forward_shipping_inr[V.cat], c = V.costs;
    var revenue = o.kept * netPrice(V.A, price);
    var parcel = o.dispatched * (ship + c.packaging + c.gift);
    var reverse = o.returns * ship * e.return_pickup_fee_multiplier + o.rto * e.rto_return_fee_inr;
    var cancel = o.cancelled * e.cancellation_cost_inr;
    var unitsLost = o.kept + o.returns - o.resold + (e.rto_units_restockable ? 0 : o.rto);
    var product = unitsLost * c.unit + o.resold * e.resale_refurb_cost_inr;
    var cost = parcel + reverse + cancel + product + (o.ad || 0);
    return { revenue: revenue, cost: cost, contribution: revenue - cost };
  }
  function perOrderAt(V, price, rates) {
    var e = V.A.economics, ship = e.forward_shipping_inr[V.cat], c = V.costs;
    var dispatched = 1 - rates.p_cancel, rto = dispatched * rates.p_rto, delivered = dispatched - rto;
    var returns = delivered * rates.p_return, kept = delivered - returns, resold = returns * rates.resale_share;
    var parts = {
      product: (kept + returns - resold + (e.rto_units_restockable ? 0 : rto)) * c.unit,
      packaging: dispatched * c.packaging, forward: dispatched * ship,
      reverse: returns * ship * e.return_pickup_fee_multiplier, rto_fee: rto * e.rto_return_fee_inr,
      gift: dispatched * c.gift, cancel: rates.p_cancel * e.cancellation_cost_inr, refurb: resold * e.resale_refurb_cost_inr
    };
    var cost = 0; for (var k in parts) cost += parts[k];
    var revenue = kept * netPrice(V.A, price), contrib = revenue - cost;
    // cancel / rto / ret are RATES (of placed, dispatched and delivered orders); the other quantities are per placed order
    return {
      price: price, cancel: rates.p_cancel, rto: rates.p_rto, ret: rates.p_return, dispatched: dispatched, delivered: delivered,
      rtoParcels: rto, returns: returns, kept: kept, resold: resold, revenue: revenue, parts: parts, cost: cost,
      contribution_per_order: contrib, contribution_per_kept: kept > 0 ? contrib / kept : -Infinity,
      gst_per_kept: price - price / (1 + V.A.economics.gst_rate_included_in_price)
    };
  }
  function perOrder(V, price, d) { d = d || meanDraw(V); return perOrderAt(V, price, eventRates(V, d, price, V.market.median, V.delivery - 4)); }
  // Break-even price: expected contribution per placed order >= 0 under the stated assumptions.
  // It is NOT a guarantee of no loss: weekly noise, launch costs and unsold stock can still lose money above it.
  function breakEven(V, d) {
    var cap = Math.ceil(V.market.median * 8);
    for (var p = 1; p <= cap; p++) if (perOrder(V, p, d).contribution_per_order >= 0) return p;
    return null;
  }

  // ------------------------------------------------------------------ simulation
  function planWeeks(V, prices) {
    return prices.map(function (p) {
      return { price: p, discount: 0, festival: 0, campaign: 0, delivery: V.delivery, median: V.market.median, demand: V.market.demand };
    });
  }

  // One listing path. noise=false gives the expected path (no weekly randomness) for the same draw.
  // opts.stopLoss = {budget, until, to}: during weeks < until, if running contribution falls below -budget,
  // switch to price `to` for the rest of the horizon (a seller's stop rule, applied inside each simulation).
  function runPath(V, d, weeks, noise, opts, keepWeekly) {
    var C = V.C, P = d.P, cv = P.conv, iv = P.impr, A = V.A;
    var r = noise ? rng(d.noiseSeed) : null;
    var prior = C.constants.prior_rating, m = V.M.meta.shrink_m, convRef = C.constants.conv_ref;
    var dispO = C.conv.dispersion || 0, dispI = C.impr.dispersion || 0, sdRev = C.stars.sd_review || 0.9;
    var logStock = Math.log(C.in_stock_mean || 1), SR = C.state_ranges;
    var adImpr = V.adWeekly > 0 ? V.adWeekly / A.economics.ads_cost_per_1000_impressions_inr * 1000 : 0;
    var nRev = 0, starSum = 0, ema = null, cum = -V.launchCost, minCum = Math.min(0, cum);
    var tot = { orders: 0, kept: 0, dispatched: 0, returns: 0, rto: 0, cancelled: 0 }, stopped = false, stopWeek = null;
    var wk = keepWeekly ? [] : null, stop = opts && opts.stopLoss;
    for (var i = 0; i < weeks.length; i++) {
      var x = weeks[i], w = i + 1, price = x.price;
      if (stop && !stopped && i < stop.until && i > 0 && cum < -stop.budget) { stopped = true; stopWeek = w; }
      if (stopped && stop) price = stop.to;
      var rel = price / x.median - 1, rp = Math.max(0, rel), dd = x.delivery - 4;
      var shown = (starSum + prior * m) / (nRev + m), rc = clamp(shown - 4, SR.rating_c[0], SR.rating_c[1]);
      var lr = Math.min(Math.log1p(nRev), SR.logrev[1]);
      var cr = ema === null ? 0 : clamp(ema / convRef - 1, SR.convratio[0], SR.convratio[1]);
      var lImpr = d.base.impr + d.eff.impr + iv.ph12 * (w <= 2 ? 1 : 0) + iv.ph34 * (w >= 3 && w <= 4 ? 1 : 0) + iv.logrev * lr +
        iv.rating_c * rc + iv.convratio * cr + iv.log_di_lag * Math.log(x.demand) + iv.campaign_week * x.campaign +
        iv.delivery_days_c * dd + logStock;
      var lConv = d.base.conv + d.eff.conv + priceTerm(V, d, rel) + cv.discount_pct * x.discount + cv.festival_week * x.festival +
        cv.campaign_week * x.campaign + cv.delivery_days_c * dd + cv.rating_c * rc + cv.logrev * lr;
      var lam = Math.exp(lImpr) + adImpr, conv = Math.min(0.5, Math.exp(lConv));
      var ev = eventRates(V, d, price, x.median, dd);
      var muStar = clamp(d.base.stars + d.eff.stars + P.stars.rel_pos * rp, 1, 5);
      var o;
      if (noise) {
        var impr = poisson(lam * gammaMult(dispI, r()), r());
        var orders = poisson(impr * conv * gammaMult(dispO, r()), r());
        var canc = binom(orders, ev.p_cancel, r()), disp = orders - canc;
        var rto = binom(disp, ev.p_rto, r()), deliv = disp - rto;
        var ret = binom(deliv, ev.p_return, r()), resold = binom(ret, ev.resale_share, r()), kept = deliv - ret;
        var nNew = binom(kept + ret, C.review_rate, r()), u = r();
        var sNew = nNew > 0 ? clamp(nNew * muStar + Math.sqrt(nNew) * sdRev * qnorm(u), nNew, 5 * nNew) : 0;
        o = { impressions: impr, orders: orders, cancelled: canc, dispatched: disp, rto: rto, delivered: deliv, returns: ret, resold: resold, kept: kept, nNew: nNew, sNew: sNew };
      } else {
        var oo = lam * conv, dsp = oo * (1 - ev.p_cancel), rt = dsp * ev.p_rto, dl = dsp - rt, rr = dl * ev.p_return, kp = dl - rr;
        var nn = (kp + rr) * C.review_rate;
        o = { impressions: lam, orders: oo, cancelled: oo - dsp, dispatched: dsp, rto: rt, delivered: dl, returns: rr, resold: rr * ev.resale_share, kept: kp, nNew: nn, sNew: nn * muStar };
      }
      o.ad = V.adWeekly;
      var money = weekMoney(V, price, o).contribution;
      cum += money; if (cum < minCum) minCum = cum;
      nRev += o.nNew; starSum += o.sNew;
      var cNow = o.impressions > 0 ? o.orders / o.impressions : 0;
      ema = ema === null ? cNow : 0.6 * ema + 0.4 * cNow;
      tot.orders += o.orders; tot.kept += o.kept; tot.dispatched += o.dispatched; tot.returns += o.returns; tot.rto += o.rto; tot.cancelled += o.cancelled;
      if (wk) wk.push({ w: w, price: price, impressions: o.impressions, orders: o.orders, kept: o.kept, profit: money, cum: cum, reviews: nRev,
                        rating: nRev > 0 ? starSum / nRev : null });
    }
    return { total: cum, minCum: minCum, orders: tot.orders, kept: tot.kept, dispatched: tot.dispatched, returns: tot.returns, rto: tot.rto,
             cancelled: tot.cancelled, reviews: nRev, rating: nRev > 0 ? starSum / nRev : null, stopped: stopped, stopWeek: stopWeek, weeks: wk };
  }

  function summarise(paths, budget) {
    var t = paths.map(function (p) { return p.total; });
    return {
      n: t.length, mean: mean(t), median: quantile(t, 0.5), p10: quantile(t, 0.1), p90: quantile(t, 0.9),
      probLoss: share(t, function (x) { return x < 0; }),
      probBudget: budget === null || budget === undefined ? null : share(paths, function (p) { return p.minCum < -budget; }),
      minCumP10: quantile(paths.map(function (p) { return p.minCum; }), 0.1),
      orders: mean(paths.map(function (p) { return p.orders; })), kept: mean(paths.map(function (p) { return p.kept; })),
      reviews: mean(paths.map(function (p) { return p.reviews; })),
      stopShare: share(paths, function (p) { return p.stopped; })
    };
  }

  // Public: Monte Carlo of one weekly plan. Returns the summary plus per-week P10/P50/P90 when opts.weekly.
  function simulate(M, input, plan, opts) {
    opts = opts || {};
    var V = input.cat && input.C ? input : view(M, input);
    var weeks = typeof plan[0] === "number" ? planWeeks(V, plan) : plan;
    var draws = opts.draws || makeDraws(V, opts.n || V.nSims, opts.seed !== undefined ? opts.seed : V.seed, opts);
    var paths = draws.map(function (d) { return runPath(V, d, weeks, opts.noise !== false, opts, !!opts.weekly); });
    var out = summarise(paths, V.budget);
    if (opts.weekly) {
      out.weekly = weeks.map(function (_, i) {
        var o = paths.map(function (p) { return p.weeks[i].orders; }), c = paths.map(function (p) { return p.weeks[i].cum; }),
            k = paths.map(function (p) { return p.weeks[i].kept; }), pr = paths.map(function (p) { return p.weeks[i].profit; });
        return { w: i + 1, orders: mean(o), ordersP10: quantile(o, 0.1), ordersP50: quantile(o, 0.5), ordersP90: quantile(o, 0.9),
                 kept: mean(k), profit: mean(pr), cum: mean(c), cumP10: quantile(c, 0.1), cumP50: quantile(c, 0.5), cumP90: quantile(c, 0.9) };
      });
      out.totals = paths.map(function (p) { return p.total; });
      out.orderTotals = paths.map(function (p) { return p.orders; });
    }
    out.paths = opts.keepPaths ? paths : undefined;
    return out;
  }

  // Expected path at the typical listing (mean coefficients, no listing effect): used for the charts.
  function expectedPath(V, prices, d) { return runPath(V, d || meanDraw(V), planWeeks(V, prices), false, null, true); }

  function flat(price, n) { var a = []; for (var i = 0; i < n; i++) a.push(price); return a; }

  // ------------------------------------------------------------------ recommendation
  function recommend(M, input) {
    validateModel(M);
    var V = view(M, input), C = V.C, mk = V.market, pol = M.assumptions.policy, H = pol.horizon_weeks;
    var md = meanDraw(V);
    var floor = breakEven(V, md);
    var draws = makeDraws(V, V.nSims, V.seed);
    var J = Math.min(pol.n_parameter_draws_for_decision_uncertainty, draws.length);

    // break-even across plausible return / RTO / cancellation rates
    var floors = draws.slice(0, J).map(function (d) { return breakEven(V, d); }).filter(function (x) { return x !== null; });
    var floorRange = floors.length ? [Math.round(quantile(floors, 0.1)), Math.round(quantile(floors, 0.9))] : null;

    // candidate grid: prices ending in 9; recommendations capped at p75 x cap, simulated up to p75 x explore
    var step = pol.candidate_step_inr, lo = ends9(mk.p25 * pol.candidate_low_vs_p25);
    var cap = ends9(mk.p75 * pol.candidate_high_vs_p75), hi = ends9(mk.p75 * pol.explore_high_vs_p75);
    var supLo = mk.median * (1 + V.support[0]), supHi = mk.median * (1 + V.support[1]);
    var grid = [];
    for (var p = lo; p <= hi; p += step) grid.push(p);
    var basePrice = pol.baseline === "offline" && +input.offline > 0 ? +input.offline : ends9(mk.median);
    var basePaths = draws.map(function (d) { return runPath(V, d, planWeeks(V, flat(basePrice, H)), true); });
    var baseSum = summarise(basePaths, V.budget);

    var cands = grid.map(function (price) {
      var paths = draws.map(function (d) { return runPath(V, d, planWeeks(V, flat(price, H)), true); });
      var s = summarise(paths, V.budget), u = perOrder(V, price, md);
      s.p = price;
      s.probBeatBaseline = share(paths, function (pp, i) { return pp.total > basePaths[i].total; });
      s.perOrder = u.contribution_per_order; s.perKept = u.contribution_per_kept;
      s.inSupport = price >= supLo - 0.5 && price <= supHi + 0.5;
      s.withinCap = price <= cap;
      s.aboveFloor = floor !== null && price >= floor;
      s.fails = [];
      if (!s.inSupport) s.fails.push("outside_observed_prices");
      if (!s.withinCap) s.fails.push("above_cap");
      if (!s.aboveFloor) s.fails.push("below_break_even");
      if (s.probLoss > pol.max_loss_probability) s.fails.push("loss_probability");
      if (V.budget !== null && s.probBudget > pol.max_budget_breach_probability) s.fails.push("budget_breach");
      s.eligible = s.fails.length === 0;
      s.budgetNeeded = -quantile(paths.map(function (pp) { return pp.minCum; }), pol.max_budget_breach_probability);
      return s;
    });
    var E = cands.filter(function (c) { return c.eligible; });

    // how sure is the best price? argmax of the expected 12-week contribution across plausible draws
    var inSup = cands.filter(function (c) { return c.inSupport; });
    var argmax = draws.slice(0, J).map(function (d) {
      var best = null, bv = -Infinity;
      inSup.forEach(function (c) { var t = runPath(V, d, planWeeks(V, flat(c.p, H)), false).total; if (t > bv) { bv = t; best = c.p; } });
      return best;
    }).filter(function (x) { return x !== null; });
    var dq = { p10: quantile(argmax, 0.1), p50: quantile(argmax, 0.5), p90: quantile(argmax, 0.9) };
    var width = argmax.length ? dq.p90 - dq.p10 : null;
    var decision = { draws: argmax.length, p10: dq.p10, p50: dq.p50, p90: dq.p90, width: width, widthShare: width === null ? null : width / mk.median,
                     threshold: pol.learning_max_optimal_price_width_vs_median };
    decision.confident = decision.widthShare !== null && decision.widthShare <= decision.threshold;

    var out = {
      engine: ENGINE_VERSION, synthetic: true, category: V.cat, subcategory: V.sub, tier: V.tier, attribute: V.attr,
      label: C.label, plural: C.plural, costs: V.costs, listingScore: V.score, market: mk, ship: M.assumptions.economics.forward_shipping_inr[V.cat],
      rating: C.rating, horizon: H, budget: V.budget, floor: floor, floorRange: floorRange,
      floorEcon: floor !== null ? perOrder(V, floor, md) : null,
      grid: { lo: lo, hi: hi, cap: cap, step: step, supportLo: Math.round(supLo), supportHi: Math.round(supHi) },
      curve: cands, baseline: { p: basePrice, summary: baseSum }, decision: decision, confident: decision.confident,
      constraints: { maxLossProbability: pol.max_loss_probability, maxBudgetBreachProbability: pol.max_budget_breach_probability, budget: V.budget },
      policy: ["balanced", "conservative", "learning"].indexOf(input.policy) >= 0 ? input.policy : "balanced"
    };
    if (+input.offline > 0) out.offline = { price: +input.offline, econ: perOrder(V, +input.offline, md) };
    // the v1 decision rule on the same model, kept for comparison: best expected 12-week contribution at the
    // typical listing (no uncertainty, no risk limits), between break-even and the cap
    var pt = null, ptv = -Infinity;
    cands.forEach(function (c) {
      if (!c.withinCap || !c.aboveFloor) return;
      var t = runPath(V, md, planWeeks(V, flat(c.p, H)), false).total;
      if (t > ptv) { ptv = t; pt = c.p; }
    });
    out.pointPolicy = { p: pt, expected: pt === null ? null : ptv };

    if (!E.length) return infeasible(V, out, cands, md);

    // Balanced: highest mean 12-week contribution among eligible prices; if a lower eligible price is within
    // `prefer_lower` of it, take the lower one (more orders, reviews sooner; the gap is inside model error)
    var best = E.reduce(function (a, b) { return b.mean > a.mean ? b : a; });
    var tol = pol.prefer_lower_within_share_of_best_mean * Math.abs(best.mean);
    var start = best;
    E.forEach(function (c) { if (c.p < start.p && c.mean >= best.mean - tol) start = c; });
    var bi = cands.indexOf(best), a = bi, b = bi, gs = pol.good_range_share_of_best_mean;
    var good = function (c) { return c.eligible && (best.mean > 0 ? c.mean >= gs * best.mean : c.mean >= best.mean - Math.abs(best.mean) * (1 - gs)); };
    while (a - 1 >= 0 && good(cands[a - 1])) a--;
    while (b + 1 < cands.length && good(cands[b + 1])) b++;
    if (start.p < cands[a].p) a = cands.indexOf(start);
    var cons = E.reduce(function (x, y) { return y.p10 > x.p10 || (y.p10 === x.p10 && y.p < x.p) ? y : x; });
    out.policies = {
      balanced: { p: start.p, modelBest: best.p, range: [cands[a].p, cands[b].p], metrics: start },
      conservative: { p: cons.p, metrics: cons }
    };

    // boundary report: is the chosen price pinned by the grid, the cap, the data range or the break-even?
    var eMin = E[0].p, eMax = E[E.length - 1].p;
    var beyondCap = cands.filter(function (c) { return !c.withinCap && c.inSupport && c.mean > best.mean; });
    out.boundary = {
      eligibleMin: eMin, eligibleMax: eMax, atUpperBound: best.p === eMax, atLowerBound: start.p === eMin,
      optimumBeyondCap: beyondCap.length > 0, bestBeyondCap: beyondCap.length ? beyondCap.reduce(function (x, y) { return y.mean > x.mean ? y : x; }).p : null,
      upperLimitedBy: eMax === cap ? "cap" : (eMax >= supHi - step ? "observed_prices" : "risk_limits"),
      lowerLimitedBy: floor !== null && eMin - step < floor ? "break_even" : (eMin <= supLo + step ? "observed_prices" : "risk_limits"),
      atGridEdge: best.p === grid[0] || best.p === grid[grid.length - 1]
    };

    // Learning / experiment policy: when the best price is uncertain, test it instead of guessing
    out.experiment = experiment(V, out, cands, draws, md, start.p);
    var lrn;
    if (decision.confident) lrn = { action: "price", p: start.p, why: "best price is well determined" };
    else if (out.experiment && out.experiment.affordable) lrn = { action: "experiment", arms: out.experiment.arms, then: start.p, why: "best price is uncertain; the test is affordable" };
    else lrn = { action: "abstain", why: out.experiment ? "best price is uncertain and the test fails the risk limits" : "best price is uncertain and no test design fits between break-even and the cap" };
    out.policies.learning = lrn;

    var chosen = out.policy === "conservative" ? out.policies.conservative : out.policies.balanced;
    out.status = "ok";
    if (out.policy === "learning" && lrn.action !== "price") out.status = lrn.action === "experiment" ? "experiment" : "abstain";
    out.start = chosen.p;
    out.pick = chosen.metrics;
    out.modelBest = best.p;
    out.range = out.policies.balanced.range;
    out.cappedAtBand = out.boundary.optimumBeyondCap;
    out.startEcon = perOrder(V, out.start, md);
    out.startSim = expectedPath(V, flat(out.start, H), md);
    if (!input.lite) out.startFan = simulate(M, V, flat(out.start, H), { draws: draws, weekly: true });
    out.position = out.start / mk.median < 0.94 ? "below" : out.start / mk.median > 1.06 ? "above" : "at";
    out.thinMargin = out.startEcon.contribution_per_kept < 0.08 * out.start;
    var lower = null;
    cands.forEach(function (c) { if (c.p <= out.start - 30 && c.inSupport && (!lower || c.p > lower.p)) lower = c; });
    out.compareLower = lower ? { p: lower.p, orders: lower.orders, total: lower.mean, probLoss: lower.probLoss } : null;
    out.cohorts = cohortEvidence(V, C.rating.expected);
    var rel = out.start / mk.median, bucket = rel < 0.95 ? 0 : rel <= 1.05 ? 1 : rel <= 1.15 ? 2 : 3;
    out.complaint = { at: C.complaints[bucket], above: C.complaints[3], base: C.complaints[1] };
    var upPct = trustRoom(V, C.rating.good - C.rating.expected);
    out.triggers = {
      raiseAtRating: C.rating.good, raisePct: upPct, raiseTo: ends9(out.start * (1 + upPct)),
      saleFloorCheck: { discount: 0.2, priceAfter: Math.round(out.start * 0.8), breaksFloor: out.start * 0.8 < floor },
      expectedReturnRate: out.startEcon.ret, expectedRto: out.startEcon.rto, expectedCancel: out.startEcon.cancel
    };
    if (input.lite) return out;                    // evaluation: skip the launch and strategy views
    var Ld = ends9(floor * 0.9);
    if (Ld >= out.start - 10) Ld = out.start - 20;
    out.launchBounds = { min: Math.min(Math.max(49, ends9(floor * 0.7)), Ld), max: out.start - 10, step: 10 };
    out.launch = launchOption(M, input, out, Ld, 4, V);
    out.strategies = compareStrategies(M, input, out, V);
    return out;
  }

  // No price passes the limits: say which limits bind and what would have to change. Nothing is relaxed.
  function infeasible(V, out, cands, md) {
    var pol = V.A.policy, mk = V.market;
    out.status = "no_feasible_candidate";
    out.message = "No feasible candidate under current constraints";
    var inSup = cands.filter(function (c) { return c.inSupport && c.withinCap; });
    var count = function (f) { return inSup.filter(function (c) { return c.fails.indexOf(f) >= 0; }).length; };
    var reasons = [];
    if (out.floor === null || out.floor > out.grid.cap) reasons.push({ code: "break_even_above_cap", floor: out.floor, cap: out.grid.cap });
    if (count("below_break_even")) reasons.push({ code: "below_break_even", n: count("below_break_even"), of: inSup.length, floor: out.floor });
    if (count("loss_probability")) reasons.push({ code: "loss_probability", n: count("loss_probability"), of: inSup.length, limit: pol.max_loss_probability });
    if (count("budget_breach")) reasons.push({ code: "budget_breach", n: count("budget_breach"), of: inSup.length, limit: pol.max_budget_breach_probability, budget: V.budget });
    out.reasons = reasons;
    var tradeoffs = [];
    var aboveFloor = inSup.filter(function (c) { return c.aboveFloor; });
    if (aboveFloor.length) {
      var lp = aboveFloor.reduce(function (a, b) { return b.probLoss < a.probLoss ? b : a; });
      if (lp.probLoss > pol.max_loss_probability) tradeoffs.push({ code: "accept_loss_probability", p: lp.p, probLoss: lp.probLoss, mean: lp.mean });
      var okLoss = aboveFloor.filter(function (c) { return c.probLoss <= pol.max_loss_probability; });
      if (V.budget !== null && okLoss.length) {
        var nb = okLoss.reduce(function (a, b) { return b.budgetNeeded < a.budgetNeeded ? b : a; });
        tradeoffs.push({ code: "larger_budget", p: nb.p, budgetNeeded: Math.ceil(Math.max(0, nb.budgetNeeded) / 100) * 100 });
      }
    }
    var maxCost = null;
    for (var c = Math.floor(V.costs.unit); c >= 0; c--) {
      var V2 = Object.create(V); V2.costs = { unit: c, packaging: V.costs.packaging, gift: V.costs.gift };
      var f = breakEven(V2, md);
      if (f !== null && f <= mk.median) { maxCost = c; break; }
    }
    if (maxCost !== null && maxCost < V.costs.unit) tradeoffs.push({ code: "lower_unit_cost", unitCost: maxCost, atPrice: ends9(mk.median) });
    out.tradeoffs = tradeoffs;
    out.maxProductCostAtMedian = maxCost;
    out.medianEcon = perOrder(V, ends9(mk.median), md);
    var top = ends9(Math.min(out.grid.cap, mk.median * (1 + V.support[1])));
    out.bleed = { price: top, econ: perOrder(V, top, md), sim: V.lite ? null : simulate(V.M, V, flat(top, V.A.policy.long_horizon_weeks), { weekly: true, n: 200 }) };
    return out;
  }

  // ------------------------------------------------------------------ learning policy: a price test
  function experiment(V, out, cands, draws, md, after) {
    var pol = V.A.policy, d = out.decision, H = pol.horizon_weeks;
    var okP = function (p) { return p >= (out.floor || Infinity) && p <= out.grid.cap && p >= out.grid.supportLo && p <= out.grid.supportHi; };
    var pool = cands.filter(function (c) { return okP(c.p); }).map(function (c) { return c.p; });
    if (pool.length < 3) return null;
    // three distinct arms spanning the plausible best prices, at least two grid steps apart
    var idx = function (x) { var bi = 0; pool.forEach(function (q, i) { if (Math.abs(q - x) < Math.abs(pool[bi] - x)) bi = i; }); return bi; };
    var n = pool.length, gap = n >= 5 ? 2 : 1;
    var iMid = clamp(idx(d.p50), gap, n - 1 - gap);
    var iLow = Math.min(idx(Math.min(d.p10, pool[iMid] * 0.88)), iMid - gap);
    var iHigh = Math.max(idx(Math.max(d.p90, pool[iMid] * 1.12)), iMid + gap);
    var low = pool[Math.max(0, iLow)], mid = pool[iMid], high = pool[Math.min(n - 1, iHigh)];
    var blk = pol.experiment_block_weeks, order = [mid, low, high, mid], plan = [];
    for (var i = 0; i < pol.experiment_weeks; i++) plan.push(order[Math.floor(i / blk) % order.length]);
    while (plan.length < H) plan.push(after);
    var paths = draws.map(function (dr) { return runPath(V, dr, planWeeks(V, plan), true); });
    var s = summarise(paths, V.budget);
    var flatAfter = cands.filter(function (c) { return c.p === after; })[0];
    // how much would the test teach about this listing's own price sensitivity? (Poisson information)
    var ep = expectedPath(V, plan.slice(0, pol.experiment_weeks), md), disp = V.C.conv.dispersion || 0;
    var xs = plan.slice(0, pol.experiment_weeks).map(function (p) { return p / V.market.median - 1; });
    var ws = ep.weeks.map(function (w) { return w.orders / (1 + disp * w.orders); });
    var sw = 0, sx = 0; ws.forEach(function (wt, k) { sw += wt; sx += wt * xs[k]; });
    var xbar = sx / sw, info = 0; ws.forEach(function (wt, k) { info += wt * (xs[k] - xbar) * (xs[k] - xbar); });
    var tau = V.C.price_slope.sd * V.uncertaintyScale;
    var post = 1 / Math.sqrt(1 / (tau * tau) + info);
    return {
      arms: [low, mid, high], plan: plan, weeks: pol.experiment_weeks, blockWeeks: blk, after: after, summary: s,
      costVsFlat: flatAfter ? flatAfter.mean - s.mean : null,
      affordable: s.probLoss <= pol.max_loss_probability && (V.budget === null || s.probBudget <= pol.max_budget_breach_probability),
      slopeSdBefore: tau, slopeSdAfter: post, expectedOrdersInTest: ep.orders
    };
  }

  // ------------------------------------------------------------------ launching low, and a fair strategy comparison
  function launchOption(M, input, rec, L, k, V0) {
    var V = V0 || view(M, input), pol = M.assumptions.policy, H = pol.long_horizon_weeks, T = rec.start;
    var n = Math.min(V.nSims, 250), draws = makeDraws(V, n, V.seed);
    var planL = flat(L, k).concat(flat(T, H - k)), planT = flat(T, H);
    var stop = V.budget !== null ? { stopLoss: { budget: V.budget * pol.stop_loss_share_of_budget, until: k, to: T } } : null;
    var pl = draws.map(function (d) { return runPath(V, d, planWeeks(V, planL), true, stop, true); });
    var pt = draws.map(function (d) { return runPath(V, d, planWeeks(V, planT), true, null, true); });
    var sl = summarise(pl, V.budget), st = summarise(pt, V.budget);
    var diff = pl.map(function (p, i) { return p.total - pt[i].total; });
    var atK = function (ps, key) { return mean(ps.map(function (p) { return p.weeks[k - 1][key]; })); };
    var cumBand = function (ps) {
      return ps[0].weeks.map(function (_, i) {
        var c = ps.map(function (p) { return p.weeks[i].cum; });
        return { w: i + 1, mean: mean(c), p10: quantile(c, 0.1), p90: quantile(c, 0.9) };
      });
    };
    var eL = perOrder(V, L), extra = atK(pl, "reviews") - atK(pt, "reviews"), cost = atK(pt, "cum") - atK(pl, "cum");
    var gain = mean(diff);
    return {
      launch: L, weeks: k, horizon: H, start: T, floor: rec.floor, belowFloor: rec.floor !== null && L < rec.floor,
      perKept: eL.contribution_per_kept, perOrder: eL.contribution_per_order,
      launchPlan: sl, flatPlan: st, gainMean: gain, gainP10: quantile(diff, 0.1), gainP90: quantile(diff, 0.9),
      probBetter: share(diff, function (x) { return x > 0; }),
      reviewsLaunch: atK(pl, "reviews"), reviewsFlat: atK(pt, "reviews"), extraReviews: extra,
      ordersLaunch: mean(pl.map(function (p) { var s = 0; for (var i = 0; i < k; i++) s += p.weeks[i].orders; return s; })),
      cumAtK: { launch: atK(pl, "cum"), flat: atK(pt, "cum") }, launchCost: cost,
      costPerReview: extra > 0.5 && cost > 0 ? cost / extra : null,
      budget: V.budget, stopShare: sl.stopShare, bandLaunch: cumBand(pl), bandFlat: cumBand(pt),
      worthIt: gain >= Math.max(500, 0.05 * Math.abs(st.mean)) && share(diff, function (x) { return x > 0; }) >= 0.6
    };
  }

  // Every strategy under the same draws, horizon, costs and constraints.
  function compareStrategies(M, input, rec, V0) {
    var V = V0 || view(M, input), pol = M.assumptions.policy, H = pol.long_horizon_weeks;
    var n = Math.min(V.nSims, 300), draws = makeDraws(V, n, V.seed), T = rec.start;
    var defs = [
      { key: "flat_recommended", label: "Recommended price, held", plan: flat(T, H) },
      { key: "flat_median", label: "Market median, held", plan: flat(ends9(V.market.median), H) }
    ];
    if (rec.floor !== null) {
      var L = ends9(rec.floor * 0.9);
      if (L < T) defs.push({ key: "launch_at_loss", label: "4 weeks at " + L + " then " + T, plan: flat(L, 4).concat(flat(T, H - 4)),
                             stop: V.budget !== null ? { budget: V.budget * pol.stop_loss_share_of_budget, until: 4, to: T } : null });
    }
    if (rec.offline) defs.push({ key: "no_change_offline", label: "Offline price, held", plan: flat(rec.offline.price, H) });
    var ref = null;
    var res = defs.map(function (s) {
      var paths = draws.map(function (d) { return runPath(V, d, planWeeks(V, s.plan), true, s.stop ? { stopLoss: s.stop } : null); });
      if (!ref) ref = paths;
      var sm = summarise(paths, V.budget);
      sm.key = s.key; sm.label = s.label; sm.firstPrice = s.plan[0];
      sm.probBeatRecommended = share(paths, function (p, i) { return p.total > ref[i].total; });
      sm.budgetEnforced = !!s.stop;
      return sm;
    });
    return { horizon: H, n: n, conditions: "same draws, costs, market, horizon and limits for every strategy", results: res };
  }

  // ------------------------------------------------------------------ evidence and triggers
  function cohortEvidence(V, split) {
    var C = V.C, e = V.A.economics, ship = e.forward_shipping_inr[V.cat], c = V.costs;
    var rows = C.cohorts.filter(function (r) { return r.eligible && r.rating !== null; }).map(function (r) {
      var o = { kept: r.kept, dispatched: r.dispatched, returns: r.returns, rto: r.rto, cancelled: r.orders - r.dispatched, resold: r.resold };
      var revenue = r.kxp / (1 + e.gst_rate_included_in_price) * (1 - e.platform_commission_rate - e.payment_fee_rate);
      var cost = weekMoney(V, 0, o).cost;
      return { pos: r.pos, tier: r.tier, good: r.rating >= split, survived: r.survived12, orders: r.orders, profit: revenue - cost };
    });
    var groups = [];
    ["below", "at", "above"].forEach(function (pos) {
      [true, false].forEach(function (good) {
        var g = rows.filter(function (r) { return r.pos === pos && r.good === good; });
        groups.push({
          pos: pos, good: good, n: g.length,
          survival: g.length ? share(g, function (r) { return r.survived; }) : null,
          orders: median(g.map(function (r) { return r.orders; })),
          profit: median(g.map(function (r) { return r.profit; })),
          lossShare: g.length ? share(g, function (r) { return r.profit < 0; }) : null
        });
      });
    });
    return groups;
  }
  // price rise (share of the median) that the same orders can bear when the rating rises by dr stars
  function trustRoom(V, dr) {
    var P = meanBlocks(V.C), gain = (P.conv.rating_c + P.impr.rating_c) * dr, d = meanDraw(V);
    var lo = 0, hi = 1;
    for (var i = 0; i < 40; i++) { var mid = (lo + hi) / 2; if (priceTerm(V, d, 0) - priceTerm(V, d, mid) < gain) lo = mid; else hi = mid; }
    return lo;
  }
  function priceEffect(M, input, x) { var V = view(M, input), d = meanDraw(V); return Math.exp(priceTerm(V, d, x) - priceTerm(V, d, 0)) - 1; }

  // ------------------------------------------------------------------ forecasting (evaluation, Task C)
  // weeks: [{price, discount, festival, campaign, delivery, median, demand}]; returns weekly P10/P50/P90
  function forecast(M, input, weeks, opts) {
    opts = opts || {};
    var V = view(M, input);
    return simulate(M, V, weeks, { n: opts.n || 300, seed: opts.seed !== undefined ? opts.seed : V.seed, weekly: true, params: opts.params,
                                   effects: opts.effects, keepPaths: !!opts.keepPaths });
  }

  var api = {
    ENGINE_VERSION: ENGINE_VERSION, MODEL_SCHEMA_VERSION: MODEL_SCHEMA_VERSION,
    ends9: ends9, detectCategory: detectCategory, listingScore: listingScore, validateModel: validateModel,
    view: view, perOrder: perOrder, breakEven: breakEven, netPrice: netPrice, weekMoney: weekMoney, perOrderAt: perOrderAt,
    simulate: simulate, expectedPath: expectedPath, forecast: forecast, recommend: recommend,
    launchOption: launchOption, compareStrategies: compareStrategies, cohortEvidence: cohortEvidence,
    trustRoom: trustRoom, priceEffect: priceEffect, makeDraws: makeDraws, meanDraw: meanDraw,
    _internal: { rng: rng, qnorm: qnorm, poisson: poisson, binom: binom, gammaMult: gammaMult, hash3: hash3, quantile: quantile, priceTerm: priceTerm, runPath: runPath, planWeeks: planWeeks }
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.PriceEngine = api;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
