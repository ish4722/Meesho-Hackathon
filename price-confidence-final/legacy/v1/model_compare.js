const E = require('./engine.js'); const truth = require('./truth.json');
const plans = { flat209: [209,26], L129x4: [129,4], L149x4: [149,4], L169x4: [169,4], L189x4: [189,4] };
for (const f of process.argv.slice(2)) {
  const M = require('./' + f), C = M.categories.soap, R = C.rating, costs = { product: 64, packaging: 12, gift: 15 };
  const z = (E.listingScore(2) - C.lq.mean) / C.lq.sd;
  console.log('\n=== model', f);
  for (const sc of ['good', 'expected', 'weak']) {
    const S = E.scen(R, sc);
    const run = (L, k) => E.simulate(M, 'soap', Array(k).fill(L).concat(Array(26 - k).fill(209)), costs, z, S);
    const base = run(209, 26);
    const row = [];
    for (const [name, [L, k]] of Object.entries(plans)) {
      const s = run(L, k), t = truth[sc][name];
      row.push(`${name}: model ${Math.round(s.total - base.total)} / truth ${Math.round(t.diff)}  rev4 ${s.weeks[3].reviews.toFixed(0)}/${t.rev4.toFixed(0)}`);
    }
    console.log(`${sc} (model ★${S.rating.toFixed(2)} vs truth ★${truth[sc].flat209.rating.toFixed(2)}) flat profit model ${Math.round(base.total)} / truth ${Math.round(truth[sc].flat209.profit)}`);
    row.slice(1).forEach(r => console.log('   ' + r));
  }
}
