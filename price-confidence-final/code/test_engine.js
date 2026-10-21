const E = require('./engine.js');
const M = require('./model.json');
const cases = [
  {name:'soap S6-like', category:'soap', product:64, packaging:12, gift:15, checks:2, budget:3000, offline:134},
  {name:'soap cheap', category:'soap', product:45, packaging:8, gift:0, checks:3, budget:2000},
  {name:'soap costly', category:'soap', product:140, packaging:15, gift:0, checks:2, budget:2000},
  {name:'tshirt', category:'tshirt', product:120, packaging:10, gift:0, checks:2, budget:5000, offline:250},
  {name:'tshirt cheap good listing', category:'tshirt', product:95, packaging:8, gift:0, checks:4, budget:8000},
];
for (const c of cases) {
  const r = E.recommend(M, c);
  console.log('\n==', c.name, '| floor', r.floor, '| status', r.status, '| market', r.market.p25, r.market.median, r.market.p75);
  if (r.status !== 'ok') { console.log(' maxProductCostAtMedian', r.maxProductCostAtMedian); continue; }
  console.log(' start', r.start, 'range', r.range, 'capped', r.cappedAtBand, 'pos', r.position,
    '| perOrder', r.startEcon.profit.toFixed(1), 'ret', r.startEcon.ret.toFixed(3), 'kept', r.startEcon.kept.toFixed(3));
  console.log(' 12w flat expected: orders', r.startSim.orders.toFixed(0), 'profit', r.startSim.total.toFixed(0), 'reviews', r.startSim.reviews.toFixed(0));
  console.log(' curve', r.curve.map(x=>x.p+':'+x.total.toFixed(0)).join(' '));
  const iv = r.invest;
  
  console.log(' cohorts', r.cohorts.map(g=>`${g.pos}/${g.good?'good':'weak'} n${g.n} s${g.survival===null?'-':(g.survival*100).toFixed(0)} p${g.profit===null?'-':g.profit.toFixed(0)}`).join(' | '));
  console.log(' triggers', JSON.stringify(r.triggers), 'complaint', JSON.stringify(r.complaint.at));
  if (r.offline) console.log(' offline', r.offline.price, 'profit/order', r.offline.econ.profit.toFixed(1));
}
