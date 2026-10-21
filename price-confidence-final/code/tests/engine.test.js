// node --test tests/   (engine behaviour; Python runs this file too via tests/test_engine_node.py)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");
const E = require(path.join(__dirname, "..", "engine.js"));
const BASE = require(path.join(__dirname, "..", "model.json"));
const clone = (x) => JSON.parse(JSON.stringify(x));
const SOAP = { category: "soap", subcategory: "herbal", tier: "standard", product: 64, packaging: 12, gift: 0, checks: 2, budget: 3000, nSims: 200 };

// set a block's coefficient uncertainty to zero
function freeze(M, cat, model) {
  const b = M.categories[cat].param_blocks.find((x) => x.model === model);
  b.chol = b.chol.map((r) => r.map(() => 0));
}
// a world where price sensitivity is known: no listing-level slope spread, no coefficient uncertainty
function certain(M) {
  M = clone(M);
  for (const cat of Object.keys(M.categories)) {
    M.categories[cat].price_slope.sd = 0;
    M.categories[cat].param_blocks.forEach((b) => { b.chol = b.chol.map((r) => r.map(() => 0)); });
    M.categories[cat].product_effects.chol = M.categories[cat].product_effects.chol.map((r) => r.map((v) => v * 0.05));
  }
  return M;
}

test("model validates and outputs are marked synthetic", () => {
  assert.equal(E.validateModel(BASE), true);
  const r = E.recommend(BASE, SOAP);
  assert.equal(r.synthetic, true);
  assert.ok(["ok", "experiment", "abstain", "no_feasible_candidate"].includes(r.status));
});

test("same seed gives identical results; another seed differs", () => {
  const a = E.recommend(BASE, SOAP), b = E.recommend(BASE, SOAP), c = E.recommend(BASE, Object.assign({}, SOAP, { seed: 99 }));
  assert.deepEqual(a.curve.map((x) => x.mean), b.curve.map((x) => x.mean));
  assert.notDeepEqual(a.curve.map((x) => x.mean), c.curve.map((x) => x.mean));
});

test("common random numbers: the baseline price never beats itself", () => {
  const r = E.recommend(BASE, SOAP);
  const self = r.curve.find((c) => c.p === r.baseline.p);
  assert.ok(self, "baseline price is on the grid");
  assert.equal(self.probBeatBaseline, 0);
  assert.ok(Math.abs(self.mean - r.baseline.summary.mean) < 1e-6);
});

test("every candidate reports mean, median, P10, P90 and probabilities", () => {
  const r = E.recommend(BASE, SOAP);
  for (const c of r.curve) {
    assert.ok(c.p10 <= c.median + 1e-9 && c.median <= c.p90 + 1e-9);
    for (const k of ["probLoss", "probBudget", "probBeatBaseline"]) assert.ok(c[k] >= 0 && c[k] <= 1, k);
    assert.equal(c.eligible, c.fails.length === 0);
  }
});

test("recommended prices satisfy every constraint", () => {
  for (const policy of ["balanced", "conservative"]) {
    const r = E.recommend(BASE, Object.assign({}, SOAP, { policy }));
    assert.equal(r.status, "ok");
    const c = r.curve.find((x) => x.p === r.start);
    assert.ok(c.eligible);
    assert.ok(r.start >= r.floor && r.start <= r.grid.cap);
    assert.ok(c.probLoss <= r.constraints.maxLossProbability);
    assert.ok(c.probBudget <= r.constraints.maxBudgetBreachProbability);
  }
});

test("conservative maximises P10, balanced maximises the mean, among eligible prices", () => {
  const r = E.recommend(BASE, SOAP);
  const el = r.curve.filter((c) => c.eligible);
  assert.equal(r.policies.balanced.modelBest, el.reduce((a, b) => (b.mean > a.mean ? b : a)).p);
  assert.equal(r.policies.conservative.p, el.reduce((a, b) => (b.p10 > a.p10 ? b : a)).p);
});

test("break-even price: per-order contribution changes sign there; never called safe", () => {
  const V = E.view(BASE, SOAP);
  const f = E.breakEven(V);
  assert.ok(E.perOrder(V, f).contribution_per_order >= 0);
  assert.ok(E.perOrder(V, f - 1).contribution_per_order < 0);
  const r = E.recommend(BASE, SOAP);
  assert.ok(!/safe/i.test(JSON.stringify(Object.keys(r))));
});

test("nothing passes: explicit status, reasons and trade-offs, limits unchanged", () => {
  const r = E.recommend(BASE, Object.assign({}, SOAP, { product: 400 }));
  assert.equal(r.status, "no_feasible_candidate");
  assert.equal(r.message, "No feasible candidate under current constraints");
  assert.ok(r.reasons.length > 0);
  assert.equal(r.start, undefined);
  assert.equal(r.recommended, null);
  assert.equal(r.constraints.maxLossProbability, BASE.assumptions.policy.max_loss_probability);
  assert.equal(r.constraints.maxBudgetBreachProbability, BASE.assumptions.policy.max_budget_breach_probability);
  assert.equal(r.constraints.budget, SOAP.budget);
  assert.ok(r.tradeoffs.some((t) => t.code === "lower_unit_cost"));
});

test("a strict loss limit is reported as the binding reason, not relaxed", () => {
  const M = clone(BASE);
  M.assumptions.policy.max_loss_probability = 0;
  const r = E.recommend(M, Object.assign({}, SOAP, { product: 100, nSims: 200 }));
  if (r.status === "no_feasible_candidate") {
    assert.ok(r.reasons.some((x) => x.code === "loss_probability" || x.code === "below_break_even"));
    assert.equal(r.constraints.maxLossProbability, 0);
  } else {
    assert.equal(r.pick.probLoss, 0);
  }
});

test("higher price is optimal for one product and lower for another", () => {
  // shipped model: premium listings respond less to price than basic ones
  const prem = E.recommend(BASE, Object.assign({}, SOAP, { tier: "premium" }));
  const basic = E.recommend(BASE, Object.assign({}, SOAP, { tier: "basic" }));
  assert.ok(prem.policies.balanced.modelBest > basic.policies.balanced.modelBest, `${prem.policies.balanced.modelBest} vs ${basic.policies.balanced.modelBest}`);
  // constructed world with known responses: an insensitive product above the median, a commodity one below it
  const M = certain(BASE);
  const cv = M.categories.soap.param_blocks.find((b) => b.model === "conv");
  const set = (name, v) => { cv.mean[cv.names.indexOf(name)] = v; };
  set("rel", -2.0); set("rel2", 0); set("rel_x_premium", 1.6); set("rel_x_basic", -4.0);
  M.categories.soap.price_support = { basic: [-0.5, 0.5], standard: [-0.5, 0.5], premium: [-0.5, 0.5], all: [-0.5, 0.5] };
  const hi = E.recommend(M, Object.assign({}, SOAP, { tier: "premium", product: 50 }));
  const lo = E.recommend(M, Object.assign({}, SOAP, { tier: "basic", product: 50 }));
  const med = hi.market.median;
  assert.ok(hi.policies.balanced.modelBest > med, `insensitive best ${hi.policies.balanced.modelBest} vs median ${med}`);
  assert.ok(lo.policies.balanced.modelBest < med, `commodity best ${lo.policies.balanced.modelBest} vs median ${med}`);
});

test("no confident recommendation when price response is very uncertain", () => {
  const M = clone(BASE);
  M.categories.soap.price_slope.sd = 3.0;
  const r = E.recommend(M, Object.assign({}, SOAP, { policy: "learning" }));
  assert.equal(r.decision.confident, false);
  assert.ok(["experiment", "abstain"].includes(r.status));
  assert.equal(r.recommended, null);
  assert.ok(["experiment", "abstain"].includes(r.policies.learning.action));
  if (r.status === "experiment") {
    const [a, b, c] = r.experiment.arms;
    assert.ok(a < b && b < c);
    assert.ok(a >= r.floor && c <= r.grid.cap);
    assert.ok(r.experiment.slopeSdAfter < r.experiment.slopeSdBefore);
  }
  const k = E.recommend(certain(BASE), Object.assign({}, SOAP, { policy: "learning" }));
  assert.equal(k.decision.confident, true);
  assert.equal(k.status, "ok");
  assert.equal(k.recommended, k.start);
});

test("boundary report flags an optimum above the cap", () => {
  const M = certain(BASE);
  const cv = M.categories.soap.param_blocks.find((b) => b.model === "conv");
  cv.mean[cv.names.indexOf("rel")] = -0.3;
  cv.mean[cv.names.indexOf("rel2")] = 0;
  M.categories.soap.price_support.standard = [-0.6, 0.8];
  const r = E.recommend(M, SOAP);
  assert.equal(r.status, "ok");
  assert.equal(r.boundary.atUpperBound, true);
  assert.equal(r.boundary.upperLimitedBy, "cap");
  assert.equal(r.boundary.optimumBeyondCap, true);
  assert.ok(r.boundary.bestBeyondCap > r.grid.cap);
});

test("strategy comparison uses common conditions and enforces the budget stop", () => {
  const r = E.recommend(BASE, SOAP);
  const s = r.strategies;
  assert.ok(s.results.every((x) => x.n === s.n));
  assert.equal(s.results[0].key, "flat_recommended");
  assert.equal(s.results[0].probBeatRecommended, 0);
  const low = s.results.find((x) => x.key === "launch_at_loss");
  if (low) assert.equal(low.budgetEnforced, true);
});

test("forecast returns ordered weekly quantiles for a price plan", () => {
  const plan = [179, 179, 169, 169, 189, 189].map((p) => ({ price: p, discount: 0, festival: 0, campaign: 0, delivery: 4, median: 180, demand: 1 }));
  const f = E.forecast(BASE, SOAP, plan, { n: 150 });
  assert.equal(f.weekly.length, plan.length);
  f.weekly.forEach((w) => assert.ok(w.ordersP10 <= w.ordersP50 && w.ordersP50 <= w.ordersP90));
});

test("category detection uses the model's keywords", () => {
  assert.equal(E.detectCategory(BASE, "Cotton round-neck T-shirt"), "tshirt");
  assert.equal(E.detectCategory(BASE, "Neem sabun 100 g"), "soap");
  assert.equal(E.detectCategory(BASE, "Kundan jhumka earrings"), "earrings");
  assert.equal(E.detectCategory(BASE, "Double bedsheet with pillow covers"), "bedsheet");
  assert.equal(E.detectCategory(BASE, "Laptop stand"), null);
});

test("fields the app reads are present", () => {
  const r = E.recommend(BASE, Object.assign({}, SOAP, { offline: 150 }));
  for (const k of ["start", "range", "floor", "floorRange", "floorEcon", "curve", "startSim", "startFan", "startEcon", "decision",
                   "policies", "boundary", "experiment", "launch", "strategies", "cohorts", "complaint", "triggers", "offline", "market"]) {
    assert.ok(r[k] !== undefined, k);
  }
});
