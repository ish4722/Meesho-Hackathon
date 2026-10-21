/*
 * Batch runner used by evaluate.py: runs engine.js on many listings so evaluation uses exactly the code the app runs.
 *   node eval_runner.js tasks.json results.json
 * tasks.json = {model: "path/to/model.json", tasks: [{id, kind: "forecast" | "recommend", input, weeks?, n?}]}
 */
"use strict";
var fs = require("fs");
var path = require("path");
var E = require(path.join(__dirname, "engine.js"));

var spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var M = JSON.parse(fs.readFileSync(spec.model, "utf8"));
E.validateModel(M);
var out = [];
spec.tasks.forEach(function (t) {
  try {
    if (t.kind === "forecast") {
      var f = E.forecast(M, t.input, t.weeks, { n: t.n || 300, seed: t.seed, keepPaths: true });
      // per-path cumulative orders after 4 weeks and after all weeks (for medians and intervals of totals)
      var paths = f.paths, n4 = Math.min(4, t.weeks.length);
      out.push({ id: t.id, weekly: f.weekly.map(function (w) { return [w.orders, w.ordersP10, w.ordersP90, w.profit, w.cum, w.ordersP50]; }),
                 orderTotals: f.orderTotals, totals: f.totals,
                 cum4: paths.map(function (p) { var s = 0; for (var i = 0; i < n4; i++) s += p.weeks[i].orders; return s; }),
                 cumN: paths.map(function (p) { return p.orders; }) });
    } else {
      var r = E.recommend(M, t.input);
      var slim = function (c) { return c ? { p: c.p, mean: c.mean, p10: c.p10, probLoss: c.probLoss, probBudget: c.probBudget } : null; };
      out.push({
        id: t.id, status: r.status, floor: r.floor, floorRange: r.floorRange, grid: r.grid, decision: r.decision,
        balanced: r.policies ? r.policies.balanced.p : null, conservative: r.policies ? r.policies.conservative.p : null,
        learning: r.policies ? r.policies.learning : null, experiment: r.experiment ? { arms: r.experiment.arms, plan: r.experiment.plan, affordable: r.experiment.affordable } : null,
        point: r.pointPolicy ? r.pointPolicy.p : null, boundary: r.boundary || null, reasons: r.reasons || null,
        pick: slim(r.pick), curve: r.curve.map(function (c) { return [c.p, c.mean, c.p10, c.probLoss, c.probBudget, c.eligible ? 1 : 0]; })
      });
    }
  } catch (err) {
    out.push({ id: t.id, error: String(err && err.message || err) });
  }
});
fs.writeFileSync(process.argv[3], JSON.stringify(out));
