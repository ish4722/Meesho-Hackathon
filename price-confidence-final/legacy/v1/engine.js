/*
 * Price Confidence - pricing engine (Team Edge Case, Meesho DICE S3)
 *
 * Pure functions over model.json (written by train_models.py). Used by the web app,
 * and by test_engine.js under node. No network, no storage.
 *
 * Steps for one seller:
 *   1. unitEconomics  - what one dispatched order costs, and what comes back, at a price
 *   2. floorPrice     - lowest listing price where a paid order covers every dispatched order
 *   3. simulate       - 12 weeks of impressions -> orders -> reviews -> visibility, at a price plan
 *   4. recommend      - best flat launch price inside the market band, a good range,
 *                       the "launch lower first" check, similar-listing evidence, and triggers
 */
(function (root) {
  "use strict";

  var ends9 = function (x) { return Math.max(49, Math.round(x / 10) * 10 - 1); };
  var sig = function (x) { return 1 / (1 + Math.exp(-x)); };
  var clamp = function (x, a, b) { return Math.min(b, Math.max(a, x)); };
  var median = function (a) {
    if (!a.length) return null;
    var s = a.slice().sort(function (x, y) { return x - y; });
    var m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };

  function detectCategory(title) {
    var t = (title || "").toLowerCase();
    if (/t[\s-]?shirt|\btees?\b|tshirt/.test(t)) return "tshirt";
    if (/soap|sabun|साबुन|bath bar/.test(t)) return "soap";
    return null;
  }

  // Listing checklist (0-4 boxes) -> listing quality score used by the models
  function listingScore(checks) { return 50 + 7.5 * clamp(checks, 0, 4); }   // 50..80, inside the data range

  // A product scenario: the rating buyers give it, and how far its own baseline sits from a typical listing
  // (shift in SD units: +0.674 = 75th percentile "good", -0.674 = 25th percentile "weak")
  var QSHIFT = 0.674;
  function scen(R, which) {
    if (which === "good") return { rating: R.good, shift: QSHIFT };
    if (which === "weak") return { rating: R.weak, shift: -QSHIFT };
    return { rating: R.expected, shift: 0 };
  }
  function asScen(s) { return typeof s === "number" ? { rating: s, shift: 0 } : s; }

  // log-conversion change from price p, relative to the market median
  function priceTerm(C, p) {
    if (C.price_form === "rel") { var r = p / C.market.median - 1; return C.conv.rel * r + C.conv.rel2 * r * r; }
    var lp = Math.log(p / C.market.median); return C.conv.lp * lp + C.conv.lp2 * lp * lp;
  }
  // effect on orders per impression of a +x price change at the typical price
  function priceEffect(C, x) { return Math.exp(priceTerm(C, C.market.median * (1 + x)) - priceTerm(C, C.market.median)) - 1; }

  function lqz(M, cat, score) {
    var C = M.categories[cat];
    return (score - C.lq.mean) / C.lq.sd;
  }

  // 1. What one dispatched order costs and returns at listing price `price`
  function unitEconomics(M, cat, price, costs, z, shift) {
    var C = M.categories[cat], k = C.keep, gst = M.meta.gst;
    var lp = Math.log(price / C.market.median);
    var ret = sig(k.ret_intercept + k.ret_lp_pos * Math.max(0, lp) + k.ret_lq * z - (shift || 0) * (k.ret_fe_sd || 0));
    var rto = k.rto;
    var delivered = 1 - rto;
    var returns = delivered * ret;
    var kept = delivered - returns;
    var lostUnits = kept + returns * (1 - k.resale);     // RTO parcels come back intact
    var parts = {
      product: costs.product * lostUnits,
      packaging: costs.packaging,                         // every parcel
      forward: C.ship,                                    // every parcel, incl. RTO and returns
      reverse: returns * C.ship,                          // return pickup
      gift: costs.gift
    };
    var cost = parts.product + parts.packaging + parts.forward + parts.reverse + parts.gift;
    var revenue = kept * price / (1 + gst);
    return {
      price: price, rto: rto, ret: ret, delivered: delivered, returns: returns, kept: kept,
      parts: parts, costPerDispatch: cost, revenue: revenue, profit: revenue - cost,
      costPerPaid: cost / kept, gstPerPaid: price - price / (1 + gst)
    };
  }

  // 2. Floor: lowest whole-rupee listing price with profit >= 0 per dispatched order
  function floorPrice(M, cat, costs, z) {
    var cap = M.categories[cat].market.median * 8;
    for (var p = 1; p <= cap; p++) {
      if (unitEconomics(M, cat, p, costs, z).profit >= 0) return p;
    }
    return null;
  }

  // 3. Week-by-week simulation of a price plan (array of weekly prices)
  function simulate(M, cat, plan, costs, z, scenario) {
    var sc = asScen(scenario), rating = sc.rating, sh = sc.shift || 0;
    var C = M.categories[cat], I = C.impr, V = C.conv;
    var prior = C.rating.prior, m = M.meta.shrink_m, logdi = Math.log(C.market.demand_last4);
    var cumRev = 0, ema = null, cum = 0, minCum = 0, orders = 0, weeks = [];
    for (var w = 1; w <= plan.length; w++) {
      var p = plan[w - 1], lp = Math.log(p / C.market.median);
      var rShown = (rating * cumRev + prior * m) / (cumRev + m);
      var rc = rShown - 4, lr = Math.log1p(cumRev);
      var convratio = ema === null ? 0 : clamp(ema / C.conv_ref - 1, -1, 3);
      var impr = Math.exp(I.intercept + I.lq * z + sh * (I.fe_sd || 0) + I.ph12 * (w <= 2 ? 1 : 0) + I.ph34 * (w >= 3 && w <= 4 ? 1 : 0)
        + I.logrev * lr + I.rating_c * rc + I.convratio * convratio + I.log_di * logdi);
      var conv = Math.exp(V.a + V.b_lq * z + sh * (V.fe_sd || 0) + priceTerm(C, p) + V.rating_c * rc + V.logrev * lr);
      var o = impr * conv;
      var u = unitEconomics(M, cat, p, costs, z, sh);
      var profit = o * u.profit;
      cumRev += C.keep.review_rate * o * (u.kept + u.returns);
      ema = ema === null ? conv : 0.6 * ema + 0.4 * conv;
      cum += profit;
      minCum = Math.min(minCum, cum);
      orders += o;
      weeks.push({ w: w, price: p, impressions: impr, orders: o, profit: profit, cum: cum, reviews: cumRev });
    }
    return { weeks: weeks, total: cum, minCum: minCum, orders: orders, reviews: cumRev };
  }

  function flat(price, n) { var a = []; for (var i = 0; i < n; i++) a.push(price); return a; }

  var LONG = 26;   // horizon for launch-at-a-loss decisions: payback often takes about 13 weeks

  // extra price the same sales can bear when the rating rises by dr stars (conversion + visibility effects)
  function trustRoom(C, dr) {
    var gain = (C.conv.rating_c + C.impr.rating_c) * dr;          // log-orders gained from the better rating
    // find the price rise (as a share of the typical price) that loses the same log-orders
    var lo = 0, hi = 3;
    for (var i = 0; i < 40; i++) { var mid = (lo + hi) / 2; if (-(Math.log(1 + priceEffect(C, mid))) < gain) lo = mid; else hi = mid; }
    return lo;
  }

  // Launch at price L for k weeks, then the start price; followed for 26 weeks under good, typical and weak ratings
  function launchOption(M, input, rec, L, k) {
    var cat = rec.category, C = M.categories[cat], R = C.rating, H = LONG, T = rec.start, floor = rec.floor;
    var costs = rec.costs, z = lqz(M, cat, rec.listingScore);
    var budgetGiven = input.budget !== null && input.budget !== undefined && input.budget !== "";
    var budget = budgetGiven ? (+input.budget || 0) : null;
    var planL = flat(L, k).concat(flat(T, H - k)), planT = flat(T, H);
    var sim = function (plan, r) { return simulate(M, cat, plan, costs, z, r); };
    var SG = scen(R, "good"), SW = scen(R, "weak"), SX = scen(R, "expected");
    var gL = sim(planL, SG), wL = sim(planL, SW), xL = sim(planL, SX);
    var gT = sim(planT, SG), wT = sim(planT, SW), xT = sim(planT, SX);
    var eL = unitEconomics(M, cat, L, costs, z);
    // launch-window figures use the good-rating runs, the same lines the chart shows
    var ordersLaunch = 0;
    for (var i = 0; i < k; i++) ordersLaunch += gL.weeks[i].orders;
    var extraReviews = gL.weeks[k - 1].reviews - gT.weeks[k - 1].reviews;
    var launchCost = gT.weeks[k - 1].cum - gL.weeks[k - 1].cum;
    var backToZero = function (s) {
      var dipped = false;
      for (var j = 0; j < s.weeks.length; j++) { if (s.weeks[j].cum < 0) dipped = true; else if (dipped) return s.weeks[j].w; }
      return dipped ? null : 0;
    };
    var catchUp = function (a, b) {
      for (var j = k; j < H; j++) if (a.weeks[j].cum >= b.weeks[j].cum) return j + 1;
      return null;
    };
    var maxWeeks = budgetGiven ? 0 : null;
    for (var kk = 1; budgetGiven && kk <= 12; kk++) {
      var s = sim(flat(L, kk).concat(flat(T, H - kk)), SW);
      if (-s.minCum <= budget) maxWeeks = kk; else break;
    }
    var worstDip = -Math.min(gL.minCum, wL.minCum, 0);
    var gain = gL.total - gT.total;
    return {
      launch: L, weeks: k, horizon: H, start: T, floor: floor, belowFloor: L < floor,
      perPaid: eL.profit / eL.kept, ordersLaunch: ordersLaunch,
      reviewsLaunch: gL.weeks[k - 1].reviews, reviewsFlat: gT.weeks[k - 1].reviews, extraReviews: extraReviews,
      cumAtK: { launch: gL.weeks[k - 1].cum, flat: gT.weeks[k - 1].cum },
      launchCost: launchCost, costPerReview: extraReviews > 0.5 && launchCost > 0 ? launchCost / extraReviews : null,
      good: gL, weak: wL, flatGood: gT, flatWeak: wT, gainIfGood: gain, changeIfWeak: wL.total - wT.total,
      worstDip: worstDip, budget: budget, withinBudget: budgetGiven ? worstDip <= budget : null, maxWeeks: maxWeeks,
      backToZeroGood: backToZero(gL), backToZeroWeak: backToZero(wL),
      catchUpGood: catchUp(gL, gT), catchUpWeak: catchUp(wL, wT),
      // "worth it" is about the money; whether the seller can afford the dip is reported separately
      worthIt: gain >= Math.max(500, 0.05 * Math.abs(gT.total))
    };
  }

  // Similar listings, re-costed with THIS seller's costs (competitor costs are unknown)
  function cohortEvidence(M, cat, costs, split) {
    var C = M.categories[cat], gst = M.meta.gst;
    var rows = C.cohorts.filter(function (r) { return r.eligible && r.rating !== null; }).map(function (r) {
      var profit = r.kxp / (1 + gst) - r.orders * (C.ship + costs.packaging + costs.gift)
        - r.returns * C.ship - (r.kept + r.returns - r.resold) * costs.product;
      return { pos: r.pos, good: r.rating >= split, survived: r.survived12, orders: r.orders, profit: profit };
    });
    var groups = [];
    ["below", "at", "above"].forEach(function (pos) {
      [true, false].forEach(function (good) {
        var g = rows.filter(function (r) { return r.pos === pos && r.good === good; });
        groups.push({
          pos: pos, good: good, n: g.length,
          survival: g.length ? g.filter(function (r) { return r.survived; }).length / g.length : null,
          orders: median(g.map(function (r) { return r.orders; })),
          profit: median(g.map(function (r) { return r.profit; })),
          lossShare: g.length ? g.filter(function (r) { return r.profit < 0; }).length / g.length : null
        });
      });
    });
    return groups;
  }

  function positionOf(price, med) {
    var r = price / med;
    return r < 0.94 ? "below" : r > 1.06 ? "above" : "at";
  }

  // 4. Everything the seller sees
  function recommend(M, input) {
    var cat = input.category, C = M.categories[cat], mk = C.market, H = M.meta.horizon_weeks;
    var costs = { product: +input.product || 0, packaging: +input.packaging || 0, gift: +input.gift || 0 };
    var score = listingScore(input.checks || 0), z = lqz(M, cat, score);
    var floor = floorPrice(M, cat, costs, z);
    var R = C.rating;

    // candidate launch prices: ending in 9, from well below the band up to the band's top
    var lo = ends9(mk.p25 * 0.6), hi = ends9(mk.p75);
    var cands = [];
    for (var p = lo; p <= hi; p += 10) cands.push(p);
    var evals = cands.map(function (p) {
      return { p: p, sim: simulate(M, cat, flat(p, H), costs, z, scen(R, "expected")), u: unitEconomics(M, cat, p, costs, z) };
    });
    var viable = evals.filter(function (e) { return floor !== null && e.p >= floor; });
    var best = null;
    viable.forEach(function (e) { if (!best || e.sim.total > best.sim.total) best = e; });

    var out = {
      category: cat, costs: costs, listingScore: score, floor: floor, market: mk, ship: C.ship,
      rating: R, curve: evals.map(function (e) { return { p: e.p, total: e.sim.total, orders: e.sim.orders, perOrder: e.u.profit }; }),
      floorEcon: floor ? unitEconomics(M, cat, floor, costs, z) : null
    };

    if (!best || best.sim.total <= 0) {
      // no price buyers accept covers the costs: find the product cost that would make the median work
      var maxCost = null;
      for (var c = Math.floor(costs.product); c >= 0; c--) {
        var cc = { product: c, packaging: costs.packaging, gift: costs.gift };
        if (floorPrice(M, cat, cc, z) <= mk.median) { maxCost = c; break; }
      }
      out.status = "no_viable_price";
      out.maxProductCostAtMedian = maxCost;
      out.medianEcon = unitEconomics(M, cat, mk.median, costs, z);
      var bleedAt = function (p, rating) {
        var sim = simulate(M, cat, flat(p, LONG), costs, z, rating), e = unitEconomics(M, cat, p, costs, z), wk = null;
        var budget = +input.budget || 0;
        if (budget <= 0) wk = 0; else for (var i = 0; i < sim.weeks.length; i++) if (sim.weeks[i].cum <= -budget) { wk = sim.weeks[i].w; break; }
        return { price: p, lossPerPaid: -e.profit / e.kept, weeksBudget: wk, sim: sim };
      };
      var top = hi;
      out.bleed = { top: bleedAt(top, scen(R, "expected")), topGood: bleedAt(top, scen(R, "good")), median: bleedAt(ends9(mk.median), scen(R, "expected")),
                    budget: +input.budget || 0, horizon: LONG };
      var th = C.conv.rating_c + C.impr.rating_c, needLog = -Math.log(1 + priceEffect(C, floor / mk.median - 1)) + Math.log(1 + priceEffect(C, top / mk.median - 1));
      out.trust = { top: top, needPct: floor / top - 1, needRating: R.expected + needLog / th,
                    good: R.good, room: [R.good, 4.5].map(function (r) { return { rating: r, pct: trustRoom(C, r - R.expected) }; }) };
      return out;
    }

    // good range: neighbours within 90% of the best 12-week profit
    var idx = viable.indexOf(best), a = idx, b = idx;
    // start price: when a lower price earns almost the same (within 5%), take it - more orders, faster reviews
    var start = best;
    for (var s = idx - 1; s >= 0 && viable[s].sim.total >= 0.95 * best.sim.total; s--) start = viable[s];
    while (a - 1 >= 0 && viable[a - 1].sim.total >= 0.9 * best.sim.total) a--;
    while (b + 1 < viable.length && viable[b + 1].sim.total >= 0.9 * best.sim.total) b++;
    out.status = "ok";
    out.start = start.p;
    out.modelBest = best.p;
    out.range = [viable[a].p, viable[b].p];
    out.cappedAtBand = best.p === hi;
    out.startEcon = start.u;
    out.startSim = start.sim;
    out.position = positionOf(start.p, mk.median);
    out.thinMargin = start.u.profit < 0.08 * start.p;
    var lower = null;                                  // a cheaper price to compare against in the reasons
    viable.forEach(function (e) { if (e.p <= start.p - 30 && (!lower || e.p > lower.p)) lower = e; });
    out.compareLower = lower ? { p: lower.p, orders: lower.sim.orders, total: lower.sim.total } : null;

    var T = start.p;
    // launch at a loss (or just lower) to build reviews: default = 10% under the floor for 4 weeks
    var Ld = ends9(floor * 0.9);
    if (Ld >= start.p - 10) Ld = start.p - 20;
    var bMin = Math.max(49, ends9(floor * 0.7)), bMax = start.p - 10;
    out.launchBounds = { min: Math.min(bMin, Ld), max: bMax, step: 10 };
    out.launch = launchOption(M, input, out, Ld, 4);
    // the gentler alternative: a discount that stays a little above the floor, same 4 weeks
    var La = ends9(Math.max(floor * 1.1, start.p * 0.75));
    if (La < start.p - 10) out.launchAlt = launchOption(M, input, out, La, 4);

    // evidence from similar listings, re-costed with this seller's costs
    out.cohorts = cohortEvidence(M, cat, costs, R.expected);

    // what buyers complain about at the start price
    var rel = T / mk.median;
    var bucket = rel < 0.95 ? 0 : rel <= 1.05 ? 1 : rel <= 1.15 ? 2 : 3;
    out.complaintBucket = bucket;
    out.complaint = { at: C.complaints[bucket], above: C.complaints[3], base: C.complaints[1] };

    // triggers after launch
    var upPct = trustRoom(C, R.good - R.expected);
    out.triggers = {
      raiseAtRating: R.good, raisePct: upPct, raiseTo: ends9(T * (1 + upPct)),
      saleFloorCheck: { discount: 0.2, priceAfter: Math.round(T * 0.8), breaksFloor: T * 0.8 < floor },
      expectedReturnRate: start.u.ret, expectedRto: start.u.rto
    };

    if (input.offline) {
      out.offline = { price: +input.offline, econ: unitEconomics(M, cat, +input.offline, costs, z) };
    }
    return out;
  }

  var api = { ends9: ends9, detectCategory: detectCategory, listingScore: listingScore, unitEconomics: unitEconomics,
              scen: scen, priceEffect: priceEffect,
              floorPrice: floorPrice, simulate: simulate, recommend: recommend, cohortEvidence: cohortEvidence,
              launchOption: launchOption, trustRoom: trustRoom, LONG: 26 };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.PriceEngine = api;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
