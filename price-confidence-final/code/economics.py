"""
Price Confidence - seller economics (shared accounting).

One place defines how a week's outcomes turn into money, so the generator's hidden truth,
the evaluation and the browser engine (engine.js, which mirrors these formulas and is
checked against them in tests/test_economics.py) all count the same things.

Definitions (all in rupees, all under the assumptions in assumptions.json):

  net price           listing price / (1 + GST) x (1 - commission - payment fee)
  contribution        revenue from kept orders minus every variable cost those orders caused:
                        forward shipping, packaging and gift on every DISPATCHED parcel
                        reverse pickup on customer returns, RTO handling fee on RTO parcels
                        cancellation cost on orders cancelled before dispatch
                        product cost of every unit not recovered (kept + returned - resold,
                          plus RTO units if they cannot be restocked), refurbishment of resold units
                        advertising spend
  contribution per placed order    expected contribution of one order at the moment it is placed
  contribution per retained sale   the same, divided by the share of orders that are kept
  cumulative contribution          running sum of weekly contribution
  cash-flow exposure               lowest point of the running cash position, which also counts
                                   one-off launch costs and stock bought up front

Contribution is not accounting profit: it leaves out fixed overheads, the seller's time,
financing and taxes on profit, which the data does not cover.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_assumptions(path=None):
    with open(path or os.path.join(HERE, "assumptions.json"), encoding="utf-8") as f:
        return json.load(f)


def net_price(price, A):
    e = A["economics"]
    return price / (1.0 + e["gst_rate_included_in_price"]) * (1.0 - e["platform_commission_rate"] - e["payment_fee_rate"])


def realized_contribution(cat, sell, costs, A, *, orders, cancelled, dispatched, rto, returns, resold, kept,
                          ad_spend=0.0):
    """Contribution of one week's realised outcomes. `costs` = dict(unit, packaging, gift)."""
    e = A["economics"]
    ship = e["forward_shipping_inr"][cat]
    revenue = kept * net_price(sell, A)
    parcel = dispatched * (ship + costs["packaging"] + costs["gift"])
    reverse = returns * ship * e["return_pickup_fee_multiplier"] + rto * e["rto_return_fee_inr"]
    cancel = cancelled * e["cancellation_cost_inr"]
    units_lost = kept + returns - resold + (0 if e["rto_units_restockable"] else rto)
    product = units_lost * costs["unit"] + resold * e["resale_refurb_cost_inr"]
    cost = parcel + reverse + cancel + product + ad_spend
    return {"revenue": revenue, "cost": cost, "contribution": revenue - cost}


def expected_per_order(cat, price, costs, A, *, p_cancel, p_rto, p_return, resale_share):
    """Expected money from ONE placed order at `price`, given event probabilities.

    p_rto is a share of dispatched orders, p_return a share of delivered orders.
    Returns per-placed-order quantities plus the contribution per retained (kept) sale.
    """
    e = A["economics"]
    ship = e["forward_shipping_inr"][cat]
    dispatched = 1.0 - p_cancel
    rto = dispatched * p_rto
    delivered = dispatched - rto
    returns = delivered * p_return
    kept = delivered - returns
    resold = returns * resale_share
    revenue = kept * net_price(price, A)
    parts = {
        "product": (kept + returns - resold + (0 if e["rto_units_restockable"] else rto)) * costs["unit"],
        "packaging": dispatched * costs["packaging"],
        "forward": dispatched * ship,
        "reverse": returns * ship * e["return_pickup_fee_multiplier"],
        "rto_fee": rto * e["rto_return_fee_inr"],
        "gift": dispatched * costs["gift"],
        "cancel": p_cancel * e["cancellation_cost_inr"],
        "refurb": resold * e["resale_refurb_cost_inr"],
    }
    cost = sum(parts.values())
    contribution = revenue - cost
    return {
        "price": price, "dispatched": dispatched, "rto": rto, "delivered": delivered, "returns": returns,
        "kept": kept, "resold": resold, "revenue": revenue, "parts": parts, "cost": cost,
        "contribution_per_order": contribution,
        "contribution_per_kept": contribution / kept if kept > 0 else float("-inf"),
    }


def break_even_price(cat, costs, A, rates_at, max_price):
    """Lowest whole-rupee price with expected contribution per placed order >= 0.

    `rates_at(price)` returns dict(p_cancel, p_rto, p_return, resale_share) at that price, because
    return and RTO rates can depend on price. Returns None if no price up to max_price breaks even.
    This is a break-even price under the stated assumptions, not a guarantee of no loss.
    """
    for p in range(1, int(max_price) + 1):
        if expected_per_order(cat, p, costs, A, **rates_at(p))["contribution_per_order"] >= 0:
            return p
    return None


def cash_exposure(weekly_contribution, *, launch_cost=0.0, upfront_units=0, unit_cost=0.0, units_used=None):
    """Lowest running cash position (<= 0) over the weeks.

    Contribution books product cost when a unit is used. If the seller buys `upfront_units` before
    launch, that cash leaves at week 0 and the product cost of the first units used is not paid again.
    `units_used` (same length as weekly_contribution) is needed when upfront_units > 0.
    """
    cash = -float(launch_cost) - upfront_units * unit_cost
    low = min(0.0, cash)
    stock = upfront_units
    for i, c in enumerate(weekly_contribution):
        refund = 0.0
        if stock > 0 and units_used is not None:
            used_from_stock = min(stock, units_used[i])
            stock -= used_from_stock
            refund = used_from_stock * unit_cost      # already paid for up front
        cash += c + refund
        low = min(low, cash)
    return low
