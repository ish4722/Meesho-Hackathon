"""Generator: reproducibility, provenance, funnel consistency, scenarios."""
import hashlib
import json
import os

import numpy as np
import pandas as pd

import data_quality
import generate_dataset
import schema
import synth_market as sm
from conftest import SMALL_N, SMALL_SEED


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_same_seed_gives_identical_files(small_dir, tmp_path):
    generate_dataset.run(SMALL_SEED, "default", SMALL_N, str(tmp_path))
    for f in ["products.csv", "weekly_panel.csv", "reviews.csv", "market_weekly.csv"]:
        assert sha(os.path.join(small_dir, f)) == sha(os.path.join(tmp_path, f)), f


def test_different_seed_differs(small_dir, tmp_path):
    generate_dataset.run(SMALL_SEED + 1, "default", SMALL_N, str(tmp_path))
    assert sha(os.path.join(small_dir, "weekly_panel.csv")) != sha(os.path.join(tmp_path, "weekly_panel.csv"))


def test_manifest_records_provenance_and_versions(small_dir):
    man = json.load(open(os.path.join(small_dir, "dataset_manifest.json")))
    assert man["provenance"].startswith("SYNTHETIC")
    assert "Not Meesho data" in man["provenance"]
    assert man["generator_version"] == sm.GENERATOR_VERSION
    assert man["schema_version"] == schema.SCHEMA_VERSION
    assert man["seed"] == SMALL_SEED
    for k in ["python", "numpy", "pandas"]:
        assert man["environment"][k]
    for f, h in man["files"].items():
        assert sha(os.path.join(small_dir, f)) == h
    assert man["config_sha256"] == hashlib.sha256(json.dumps(man["config"], sort_keys=True, default=str).encode()).hexdigest()


def test_data_quality_has_no_hard_failures(small_dir):
    checks, summary = data_quality.run_checks(data_quality.load(small_dir))
    fails = [c for c in checks if c["status"] == "FAIL"]
    assert not fails, fails


def test_funnel_identities(small_tables):
    _, W, _, _, _ = small_tables
    assert (W.clicks <= W.impressions).all()
    assert (W.add_to_cart <= W.clicks).all()
    assert (W.orders_placed <= W.add_to_cart).all()
    assert (W.orders_placed == W.cancelled_orders + W.dispatched_orders).all()
    assert (W.dispatched_orders == W.rto_orders + W.delivered_orders).all()
    assert (W.delivered_orders == W.customer_returns + W.kept_orders).all()
    assert (W.returns_resold <= W.customer_returns).all()
    reasons = W[[f"return_{r}" for r in schema.RETURN_REASONS]].sum(axis=1)
    assert (reasons == W.customer_returns).all()


def test_heterogeneous_archetypes_and_categories(small_tables):
    P = small_tables[0]
    assert set(P.category) == set(sm.CATEGORIES)
    assert P.hidden_archetype.nunique() >= 5
    assert set(P.positioning_tier) == {"basic", "standard", "premium"}


def test_price_changes_have_reasons(small_tables):
    W = small_tables[1].sort_values(["product_id", "week_num"])
    changed = W.groupby("product_id").selling_price.diff().fillna(0) != 0
    assert (W.loc[changed & (W.week_num > 1), "price_change_reason"] != "none").all()


def test_low_price_variation_scenario_warns(tmp_path):
    generate_dataset.run(SMALL_SEED, "low_price_variation", 60, str(tmp_path))
    checks, _ = data_quality.run_checks(data_quality.load(str(tmp_path)))
    pv = [c for c in checks if c["check"].startswith("price_variation")]
    assert pv and any(c["status"] == "WARN" for c in pv)


def test_homogeneous_scenario_has_one_archetype(tmp_path):
    generate_dataset.run(SMALL_SEED, "homogeneous", 20, str(tmp_path))
    P = pd.read_csv(os.path.join(tmp_path, "products.csv"))
    assert set(P.hidden_archetype) == {"moderate"}


def test_truth_optimal_price_differs_by_archetype():
    """In the generator itself, a price-insensitive product is best sold dearer than a commodity product."""
    A = generate_dataset.econ.load_assumptions()
    market = sm.Market(np.random.default_rng(5))
    rng = np.random.default_rng(9)
    base = sm.draw_product("soap", 0, rng, sm.LAUNCH_FIRST)
    launch = sm.date(2026, 3, 2)
    med = market.comp_median("soap", base["obs"]["subcategory"], launch)
    prices = [sm.ends_in_9(med * f) for f in np.arange(0.7, 1.45, 0.05)]

    def best(arch, beta, kappa_hi, sig):
        p = json.loads(json.dumps(base, default=str))
        p["obs"]["launch_week"] = launch
        p["latent"].update(archetype=arch, resp_beta=beta, resp_kappa_hi=kappa_hi, resp_kappa_lo=0.0, resp_sig=sig,
                           fair_mult=1.0, quality=0.5)
        p["private"].update(unit=med * 0.35, packaging=8.0, gift=0.0)
        means = []
        for price in prices:
            tot = [sm.simulate_plan(p, [price] * 12, market, launch, A, seed=s)["cum"][-1] for s in range(25)]
            means.append(np.mean(tot))
        return prices[int(np.argmax(means))]

    commodity = best("commodity", 4.5, 3.0, 0.0)
    insensitive = best("insensitive", 0.4, 0.5, 0.0)
    assert insensitive > commodity, (insensitive, commodity)
