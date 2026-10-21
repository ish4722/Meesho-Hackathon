"""Ground truth: run Seller 6's soap through the generator's own buyer mechanics (the 'real world' of this demo)."""
import numpy as np
C = dict(base_impr=9000, ctr0=0.045, atc0=0.09, ord0=0.36, a=0.9, b=1.1, c=1.0, ret0=0.08, ret_q=-0.5, ret_price=0.8,
         rto_cod=0.14, rto_prepaid=0.02, resale=0.0)
SHIP, PACK, GIFT, PROD, GST, MM = 30, 12, 15, 64, 0.05, 186.0
LZ = (65 - 62) / 12          # 2 checklist boxes -> score 65

def run(plan, q, seed):
    rng = np.random.default_rng(seed)
    cum_rev, star_sum, ema, cum = 0, 0.0, None, 0.0
    revs_wk4 = None
    conv_ref = C["ctr0"] * C["atc0"] * C["ord0"]
    for w, sell in enumerate(plan, start=1):
        rating = star_sum / cum_rev if cum_rev else None
        fair = MM * np.exp(0.08 * q + 0.04 * LZ); pr = sell / fair
        if cum_rev == 0: star_eff, trust = 0.85, 0.0
        else:
            star_eff = np.exp(0.35 * (rating - 4.0)) * (0.9 + 0.1 * min(1, cum_rev / 30))
            trust = min(1.0, np.log1p(cum_rev) / np.log1p(60)) * np.clip((rating - 3.0) / 1.2, 0, 1.2)
        rank = 0.55 + 0.45 * trust + (0.3 * (ema / conv_ref - 1) if ema is not None else 0)
        rank = float(np.clip(rank, 0.3, 1.6)); boost = 1.8 if w <= 2 else (1.3 if w <= 4 else 1.0)
        impr = int(rng.poisson(C["base_impr"] * rank * boost * rng.lognormal(0, 0.15)))
        ctr = np.clip(C["ctr0"] * np.exp(-C["a"] * (pr - 1)) * np.exp(0.25 * LZ) * star_eff, 0.002, 0.25)
        views = int(rng.binomial(impr, ctr))
        atc = int(rng.binomial(views, np.clip(C["atc0"] * np.exp(-C["b"] * (pr - 1)) * np.exp(0.2 * q) * (0.85 + 0.15 * trust), 0.005, 0.6)))
        orders = int(rng.binomial(atc, np.clip(C["ord0"] * np.exp(-C["c"] * (pr - 1)), 0.02, 0.9)))
        cod = int(rng.binomial(orders, 0.66))
        rto = int(rng.binomial(cod, C["rto_cod"])) + int(rng.binomial(orders - cod, C["rto_prepaid"]))
        delivered = orders - rto
        returns = int(rng.binomial(delivered, np.clip(C["ret0"] * np.exp(C["ret_q"] * q) * (1 + C["ret_price"] * max(0, pr - 1)), 0.005, 0.6)))
        kept = delivered - returns
        n_k, n_r = int(rng.binomial(kept, 0.14)), int(rng.binomial(returns, 0.18))
        for is_ret in [False] * n_k + [True] * n_r:
            mu = (2.6 + 0.3 * q) if is_ret else (3.95 + 0.45 * q - 0.5 * max(0, pr - 1.05))
            star_sum += int(np.clip(round(rng.normal(mu, 0.8)), 1, 5)); cum_rev += 1
        if w == 4: revs_wk4 = cum_rev
        cum += kept * sell / (1 + GST) - orders * (SHIP + PACK + GIFT) - returns * SHIP - (kept + returns) * PROD
        cv = orders / impr if impr else 0.0
        ema = cv if ema is None else 0.6 * ema + 0.4 * cv
    return cum, revs_wk4, (star_sum / cum_rev if cum_rev else 0)

N = 400
plans = {"start 209": [209] * 26, "129 x4 then 209 (loss)": [129] * 4 + [209] * 22, "169 x4 then 209": [169] * 4 + [209] * 22}
for q, label in [(1.0, "good product (q=+1)"), (0.0, "typical (q=0)"), (-1.0, "weak (q=-1)")]:
    res = {k: np.array([run(p, q, s) for s in range(N)]) for k, p in plans.items()}
    base = res["start 209"]
    print(f"\n{label}: final rating ~{base[:,2].mean():.2f}")
    for k, v in res.items():
        d = v[:, 0] - base[:, 0]
        print(f"  {k:24s} 26-wk profit {v[:,0].mean():8.0f} | reviews wk4 {v[:,1].mean():5.1f} | vs start-209: {d.mean():+7.0f} (beats it in {np.mean(d>0)*100:3.0f}% of runs)")
