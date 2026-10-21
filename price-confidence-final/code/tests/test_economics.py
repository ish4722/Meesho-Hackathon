"""Seller economics: definitions, edge cases and Python / JavaScript parity."""
import json
import math
import os
import subprocess

import pytest

import economics as econ

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = econ.load_assumptions()
COSTS = dict(unit=60.0, packaging=10.0, gift=5.0)
RATES = dict(p_cancel=0.05, p_rto=0.12, p_return=0.08, resale_share=0.6)


def test_hand_computed_order():
    r = econ.expected_per_order("soap", 210.0, COSTS, A, **RATES)
    disp = 0.95
    rto = disp * 0.12
    deliv = disp - rto
    ret = deliv * 0.08
    kept = deliv - ret
    resold = ret * 0.6
    revenue = kept * 210 / 1.05
    cost = (kept + ret - resold) * 60 + disp * (10 + 5 + 30) + ret * 30
    assert r["contribution_per_order"] == pytest.approx(revenue - cost)
    assert r["contribution_per_kept"] == pytest.approx((revenue - cost) / kept)


def test_realized_matches_expected_in_aggregate():
    n = 10000
    o = dict(orders=n, cancelled=500, dispatched=9500, rto=1140, returns=667, resold=400, kept=9500 - 1140 - 667)
    r = econ.realized_contribution("soap", 210.0, COSTS, A, **o)
    e = econ.expected_per_order("soap", 210.0, COSTS, A, p_cancel=0.05, p_rto=1140 / 9500, p_return=667 / (9500 - 1140),
                                resale_share=400 / 667)
    assert r["contribution"] / n == pytest.approx(e["contribution_per_order"], rel=1e-9)


def test_zero_costs_and_full_resale():
    r = econ.expected_per_order("soap", 100.0, dict(unit=0, packaging=0, gift=0), A, p_cancel=0, p_rto=0, p_return=0.5, resale_share=1.0)
    # only shipping (forward on every parcel, reverse on returns) is lost
    assert r["contribution_per_order"] == pytest.approx(0.5 * 100 / 1.05 - 30 - 0.5 * 30)


def test_nothing_kept_gives_minus_infinity_per_kept():
    r = econ.expected_per_order("soap", 100.0, COSTS, A, p_cancel=0, p_rto=1.0, p_return=0, resale_share=0)
    assert r["kept"] == 0
    assert r["contribution_per_kept"] == -math.inf


def test_break_even_is_lowest_non_negative_price():
    rates = lambda p: RATES
    be = econ.break_even_price("soap", COSTS, A, rates, 2000)
    assert econ.expected_per_order("soap", be, COSTS, A, **RATES)["contribution_per_order"] >= 0
    assert econ.expected_per_order("soap", be - 1, COSTS, A, **RATES)["contribution_per_order"] < 0
    assert econ.break_even_price("soap", dict(unit=5000, packaging=0, gift=0), A, rates, 300) is None


def test_commission_and_rto_fee_raise_break_even():
    rates = lambda p: RATES
    be0 = econ.break_even_price("soap", COSTS, A, rates, 2000)
    A2 = json.loads(json.dumps(A))
    A2["economics"].update(platform_commission_rate=0.1, rto_return_fee_inr=20, cancellation_cost_inr=5,
                           rto_units_restockable=False, resale_refurb_cost_inr=8)
    assert econ.break_even_price("soap", COSTS, A2, rates, 2000) > be0


def test_cash_exposure_counts_launch_cost_and_upfront_stock():
    weekly = [-400, -300, 200, 300]
    assert econ.cash_exposure(weekly) == -700
    assert econ.cash_exposure(weekly, launch_cost=500) == -1200
    # 20 units bought up front (-1000); the 2 units used each week were already paid for (+100 back per week)
    low = econ.cash_exposure(weekly, upfront_units=20, unit_cost=50, units_used=[2, 2, 2, 2])
    assert low == pytest.approx(-1000 - 400 + 100 - 300 + 100)


CASES = [
    ("soap", 189.0, COSTS, RATES),
    ("tshirt", 349.0, dict(unit=120, packaging=10, gift=0), dict(p_cancel=0.04, p_rto=0.15, p_return=0.18, resale_share=0.5)),
    ("bedsheet", 599.0, dict(unit=0, packaging=0, gift=0), dict(p_cancel=0.0, p_rto=0.0, p_return=0.0, resale_share=1.0)),
    ("earrings", 99.0, dict(unit=30, packaging=5, gift=10), dict(p_cancel=0.2, p_rto=0.3, p_return=0.4, resale_share=0.0)),
]


@pytest.mark.parametrize("variant", ["default", "all_fees"])
def test_engine_js_matches_python(variant):
    A2 = json.loads(json.dumps(A))
    if variant == "all_fees":
        A2["economics"].update(platform_commission_rate=0.12, payment_fee_rate=0.02, rto_return_fee_inr=25,
                               cancellation_cost_inr=4, rto_units_restockable=False, resale_refurb_cost_inr=6,
                               return_pickup_fee_multiplier=1.3)
    script = """
      const E = require(process.argv[1]); const A = JSON.parse(process.argv[2]); const cases = JSON.parse(process.argv[3]);
      const out = cases.map(([cat, price, c, r]) => {
        const V = {A: A, cat: cat, costs: {unit: c.unit, packaging: c.packaging, gift: c.gift}};
        const e = E.perOrderAt(V, price, r);
        const o = {kept: 1000 * e.kept, dispatched: 1000 * e.dispatched, returns: 1000 * e.returns, rto: 1000 * e.rtoParcels,
                   cancelled: 1000 * r.p_cancel, resold: 1000 * e.resold};
        return [e.contribution_per_order, E.weekMoney(V, price, o).contribution / 1000];
      });
      console.log(JSON.stringify(out));
    """
    res = subprocess.run(["node", "-e", script, os.path.join(CODE, "engine.js"), json.dumps(A2), json.dumps(CASES)],
                         capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    for (cat, price, c, r), (per_order, weekly) in zip(CASES, js):
        py = econ.expected_per_order(cat, price, c, A2, **r)["contribution_per_order"]
        assert per_order == pytest.approx(py, abs=1e-9)
        assert weekly == pytest.approx(py, abs=1e-9)
