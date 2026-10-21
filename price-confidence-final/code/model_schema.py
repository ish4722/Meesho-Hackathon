"""
Price Confidence - model.json contract.

train_models.py calls validate() before writing; tests call it on the shipped model; engine.js has a
matching validateModel() so the offline app refuses a model file it cannot use instead of showing
numbers computed from missing fields.
"""
import math

import schema

MODEL_SCHEMA_VERSION = "2.0"

TOP = ["model_schema_version", "meta", "assumptions", "categories"]
META = ["synthetic", "provenance", "listings", "listing_weeks", "data_to", "feature_whitelist", "schema_version"]
CATEGORY = ["label", "plural", "keywords", "attr_name", "ship", "levels", "attr_freq", "subcats", "constants", "rating",
            "conv", "impr", "returns", "rto", "cancel", "resale", "review_rate", "stars", "product_effects",
            "price_slope", "param_blocks", "complaints", "cohorts", "price_support", "in_stock_mean", "state_ranges"]
LEVELS = ["tier", "sub", "attr"]
MARKET = ["median", "p25", "p75", "median_prev4", "median_last4", "demand_prev4", "demand_last4", "as_of"]
BLOCKS = {"conv": "conversion", "impr": "visibility", "ret": "returns", "rto": "rto", "cancel": "cancel", "stars": "stars"}
ECON = ["gst_rate_included_in_price", "platform_commission_rate", "payment_fee_rate", "forward_shipping_inr",
        "return_pickup_fee_multiplier", "rto_return_fee_inr", "cancellation_cost_inr", "rto_units_restockable",
        "resale_refurb_cost_inr", "ads_cost_per_1000_impressions_inr"]
POLICY = ["horizon_weeks", "n_simulations", "n_parameter_draws_for_decision_uncertainty", "seed", "candidate_step_inr",
          "candidate_low_vs_p25", "candidate_high_vs_p75", "max_loss_probability", "max_budget_breach_probability",
          "learning_max_optimal_price_width_vs_median", "experiment_weeks", "stop_loss_share_of_budget",
          "long_horizon_weeks", "explore_high_vs_p75", "good_range_share_of_best_mean", "prefer_lower_within_share_of_best_mean"]


class ModelSchemaError(ValueError):
    pass


def _need(d, keys, where):
    if not isinstance(d, dict):
        raise ModelSchemaError(f"{where}: expected an object")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ModelSchemaError(f"{where}: missing {missing}")


def _finite(x, where):
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            _finite(v, f"{where}[{i}]")
    elif isinstance(x, dict):
        for k, v in x.items():
            _finite(v, f"{where}.{k}")
    elif isinstance(x, float) and not math.isfinite(x):
        raise ModelSchemaError(f"{where}: not finite")


def validate(m):
    """Raise ModelSchemaError if `m` is not a usable v2 model. Returns m."""
    _need(m, TOP, "model")
    if m["model_schema_version"] != MODEL_SCHEMA_VERSION:
        raise ModelSchemaError(f"model_schema_version {m['model_schema_version']!r} != {MODEL_SCHEMA_VERSION!r}")
    _need(m["meta"], META, "meta")
    if m["meta"]["synthetic"] is not True:
        raise ModelSchemaError("meta.synthetic must be true: no real marketplace data is used")
    _need(m["assumptions"], ["economics", "policy"], "assumptions")
    _need(m["assumptions"]["economics"], ECON, "assumptions.economics")
    _need(m["assumptions"]["policy"], POLICY, "assumptions.policy")
    if not m["categories"]:
        raise ModelSchemaError("categories: empty")
    for cat, c in m["categories"].items():
        w = f"categories.{cat}"
        _need(c, CATEGORY, w)
        _need(c["levels"], LEVELS, w + ".levels")
        for name, col in (("tier", "tier"), ("sub", "sub"), ("attr", "attr")):
            if not c["levels"][name]:
                raise ModelSchemaError(f"{w}.levels.{name}: empty")
        if set(c["subcats"]) != set(c["levels"]["sub"]):
            raise ModelSchemaError(f"{w}.subcats keys differ from levels.sub")
        for s, sc in c["subcats"].items():
            _need(sc, ["market", "label"], f"{w}.subcats.{s}")
            _need(sc["market"], MARKET, f"{w}.subcats.{s}.market")
            mk = sc["market"]
            if not (0 < mk["p25"] <= mk["median"] <= mk["p75"]):
                raise ModelSchemaError(f"{w}.subcats.{s}.market: need 0 < p25 <= median <= p75")
        for part, model in (("conv", "conversion"), ("impr", "visibility"), ("returns", "returns")):
            _need(c[part], ["coef", "second", "fe_sd"], f"{w}.{part}")
            want = set(schema.MODEL_FEATURES[model])
            if set(c[part]["coef"]) != want:
                raise ModelSchemaError(f"{w}.{part}.coef: features {sorted(c[part]['coef'])} != whitelist {sorted(want)}")
            if "const" not in c[part]["second"]:
                raise ModelSchemaError(f"{w}.{part}.second: no const")
        if c["returns"]["coef"]["rel_pos"] < 0:
            raise ModelSchemaError(f"{w}.returns.coef.rel_pos must be >= 0")
        if c["stars"]["coef"]["rel_pos"] > 0:
            raise ModelSchemaError(f"{w}.stars.coef.rel_pos must be <= 0")
        for part in ("rto", "cancel"):
            if "const" not in c[part]["coef"]:
                raise ModelSchemaError(f"{w}.{part}.coef: no const")
        pe = c["product_effects"]
        _need(pe, ["names", "chol"], f"{w}.product_effects")
        n = len(pe["names"])
        if len(pe["chol"]) != n or any(len(r) != n for r in pe["chol"]):
            raise ModelSchemaError(f"{w}.product_effects.chol: not {n}x{n}")
        seen = set()
        for b in c["param_blocks"]:
            _need(b, ["model", "names", "mean", "chol"], f"{w}.param_blocks")
            k = len(b["names"])
            if len(b["mean"]) != k or len(b["chol"]) != k or any(len(r) != k for r in b["chol"]):
                raise ModelSchemaError(f"{w}.param_blocks.{b['model']}: names/mean/chol sizes differ")
            for i in range(k):
                if any(abs(b["chol"][i][j]) > 1e-12 for j in range(i + 1, k)):
                    raise ModelSchemaError(f"{w}.param_blocks.{b['model']}: chol not lower-triangular")
            seen.add(b["model"])
        missing = set(BLOCKS) - seen
        if missing:
            raise ModelSchemaError(f"{w}.param_blocks: missing {sorted(missing)}")
        for t in c["levels"]["tier"] + ["all"]:
            lo_hi = c["price_support"].get(t)
            if not lo_hi or not (lo_hi[0] < lo_hi[1]):
                raise ModelSchemaError(f"{w}.price_support.{t}: need [lo < hi]")
        if not (0 < c["in_stock_mean"] <= 1):
            raise ModelSchemaError(f"{w}.in_stock_mean outside (0, 1]")
        if not (0 <= c["resale"] <= 1):
            raise ModelSchemaError(f"{w}.resale outside [0, 1]")
        if c["price_slope"]["sd"] < 0:
            raise ModelSchemaError(f"{w}.price_slope.sd < 0")
        _finite({k: c[k] for k in ("conv", "impr", "returns", "rto", "cancel", "stars", "product_effects",
                                   "price_slope", "param_blocks", "constants")}, w)
    return m
