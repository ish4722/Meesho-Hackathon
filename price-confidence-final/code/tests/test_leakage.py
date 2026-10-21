"""Feature whitelist and temporal safeguards: hidden or future columns must never reach a model."""
import numpy as np
import pandas as pd
import pytest

import features as F
import schema
import train_models


def test_every_model_feature_respects_the_temporal_rule():
    for model, cols in schema.MODEL_FEATURES.items():
        assert schema.assert_features_allowed(model, cols)


@pytest.mark.parametrize("col", ["hidden_true_quality", "hidden_archetype", "final_rating", "total_orders",
                                 "seller_unit_cost_inr", "seller_contribution_inr", "cumulative_rating"])
def test_hidden_future_and_private_columns_are_rejected(col):
    with pytest.raises(schema.LeakageError):
        schema.assert_features_allowed("conversion", schema.MODEL_FEATURES["conversion"] + [col])


def test_same_week_outcome_is_rejected(monkeypatch):
    monkeypatch.setitem(schema.FEATURES, "orders_now", dict(sources=[("weekly_panel", "orders_placed", 0)], doc="leak"))
    monkeypatch.setitem(schema.MODEL_FEATURES, "visibility", schema.MODEL_FEATURES["visibility"] + ["orders_now"])
    with pytest.raises(schema.LeakageError, match="same week"):
        schema.assert_features_allowed("visibility", schema.MODEL_FEATURES["visibility"])


def test_same_week_market_is_rejected(monkeypatch):
    monkeypatch.setitem(schema.FEATURES, "median_now", dict(sources=[("market_weekly", "platform_median_price", 0)], doc="leak"))
    monkeypatch.setitem(schema.MODEL_FEATURES, "conversion", ["median_now"])
    with pytest.raises(schema.LeakageError, match="same week"):
        schema.assert_features_allowed("conversion", ["median_now"])


def test_design_refuses_injected_hidden_column(small_tables, monkeypatch):
    P, W, Mk, _, _ = small_tables
    k = F.fit_constants(W, P)
    fr = F.build_frame(W[W.category == "soap"], P[P.category == "soap"], Mk[Mk.category == "soap"], k)
    fr = fr.merge(P[["product_id", "hidden_true_quality"]], on="product_id")
    monkeypatch.setitem(schema.MODEL_FEATURES, "conversion", schema.MODEL_FEATURES["conversion"] + ["hidden_true_quality"])
    with pytest.raises(schema.LeakageError):
        F.design(fr, "conversion")


def _frame(P, W, Mk, k):
    return F.build_frame(W, P, Mk, k).set_index(["product_id", "week_num"]).sort_index()


def test_future_outcomes_do_not_change_earlier_features(small_tables):
    P, W, Mk, _, _ = small_tables
    k = F.fit_constants(W, P)
    pid = W.groupby("product_id").week_num.max().sort_values().index[-1]
    t = 6
    W2 = W.copy()
    fut = (W2.product_id == pid) & (W2.week_num >= t)
    for c in ["impressions", "orders_placed", "cumulative_reviews", "kept_orders", "new_reviews"]:
        W2.loc[fut, c] = W2.loc[fut, c] * 3 + 7
    W2.loc[fut, "cumulative_rating"] = 1.0
    feats = sorted(set(sum(schema.MODEL_FEATURES.values(), [])))
    a, b = _frame(P, W, Mk, k), _frame(P, W2, Mk, k)
    for c in feats:
        np.testing.assert_allclose(a.loc[(pid, slice(1, t)), c].values, b.loc[(pid, slice(1, t)), c].values, err_msg=c)
    # and the perturbation is visible after week t, so the test is not vacuous
    assert not np.allclose(a.loc[(pid, slice(t + 1, None)), "logrev"].values, b.loc[(pid, slice(t + 1, None)), "logrev"].values)


def test_this_weeks_market_does_not_change_this_weeks_features(small_tables):
    P, W, Mk, _, _ = small_tables
    k = F.fit_constants(W, P)
    row = W.iloc[len(W) // 2]
    Mk2 = Mk.copy()
    hit = (Mk2.subcategory == row.subcategory) & (Mk2.week_start == row.week_start)
    Mk2.loc[hit, ["platform_median_price", "category_demand_index"]] *= 2
    feats = sorted(set(sum(schema.MODEL_FEATURES.values(), [])))
    a, b = _frame(P, W, Mk, k), _frame(P, W, Mk2, k)
    key = (row.product_id, row.week_num)
    for c in feats:
        assert a.loc[key, c] == pytest.approx(b.loc[key, c]), c


def test_constants_come_from_training_rows_only(small_tables):
    P, W, Mk, R, man = small_tables
    soap = sorted(P[P.category == "soap"].product_id)
    ids = soap[: len(soap) * 2 // 3]
    base = train_models.fit(P, W, Mk, R, man, train_ids=ids, cutoff="2026-03-02", categories=["soap"])
    W2 = W.copy()
    held = ~W2.product_id.isin(ids)
    W2.loc[held, "orders_placed"] = W2.loc[held, "orders_placed"] * 5
    W2.loc[held, "cumulative_rating"] = 1.5
    P2 = P.copy()
    P2.loc[~P2.product_id.isin(ids), "final_rating"] = 1.0
    late = W2.week_start >= "2026-03-02"
    W2.loc[late, "orders_placed"] = 0
    other = train_models.fit(P2, W2, Mk, R, man, train_ids=ids, cutoff="2026-03-02", categories=["soap"])
    assert base["categories"]["soap"]["constants"] == other["categories"]["soap"]["constants"]
    assert base["categories"]["soap"]["conv"]["coef"] == pytest.approx(other["categories"]["soap"]["conv"]["coef"])


def test_v1_leak_would_have_been_caught():
    """v1 built the rating prior from final_rating and used this week's market median; both are rejected now."""
    assert schema.role("products", "final_rating") == "future_derived"
    with pytest.raises(schema.LeakageError):
        schema.assert_features_allowed("conversion", ["final_rating"])
