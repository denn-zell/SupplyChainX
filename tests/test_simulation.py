"""Tests of the simulation model (simulation.py)."""

import dataclasses
import math

import numpy as np
import pytest

from constants import DEFAULT_SCENARIO, DEFAULT_SEED, DEFAULT_WEEKS, GRID_POINTS
from simulation import (
    Scenario,
    draw_demand,
    effective_mean_demand,
    fill_rate_percent,
    run_monte_carlo,
    run_policy_grid,
    run_simulation,
    simulate_batch,
)


def random_scenario(rng: np.random.Generator) -> Scenario:
    """A valid scenario with random parameters, used to compare implementations on many cases."""
    return Scenario(
        reorder_point=int(rng.integers(-50, 250)),
        order_quantity=int(rng.integers(1, 300)),
        mean_demand=float(rng.uniform(0, 50)),
        std_demand=float(rng.uniform(0, 35)),
        lead_time=int(rng.integers(1, 9)),
        holding_cost=float(rng.uniform(0, 4)),
        backorder_cost=float(rng.uniform(0, 25)),
        order_cost=float(rng.uniform(0, 150)),
        initial_inventory=int(rng.integers(0, 400)),
    )


def reference_run(scenario: Scenario, demand: np.ndarray) -> dict:
    """Independent re-implementation of one run, written in a deliberately different style.

    Stock on hand and backlog are kept separately and deliveries are kept in a calendar.
    """
    on_hand, backlog = scenario.initial_inventory, 0
    calendar: dict[int, int] = {}  # week -> quantity that arrives
    result = {"cost": 0.0, "fulfilled": 0, "demand": 0, "orders": 0, "backlog_weeks": 0, "net_inventory": []}
    for week, week_demand in enumerate(demand, start=1):
        incoming = calendar.pop(week, 0)
        cleared = min(incoming, backlog)  # arrivals clear the backlog first
        backlog -= cleared
        on_hand += incoming - cleared
        served = min(int(week_demand), on_hand)
        on_hand -= served
        backlog += int(week_demand) - served
        position = on_hand - backlog + sum(calendar.values())
        placed = position <= scenario.reorder_point
        if placed:
            arrival = week + scenario.lead_time
            calendar[arrival] = calendar.get(arrival, 0) + scenario.order_quantity
        result["cost"] += (on_hand * scenario.holding_cost + backlog * scenario.backorder_cost
                           + (scenario.order_cost if placed else 0.0))
        result["fulfilled"] += served
        result["demand"] += int(week_demand)
        result["orders"] += placed
        result["backlog_weeks"] += backlog > 0
        result["net_inventory"].append(on_hand - backlog)
    return result


# Known results of the default scenario
def test_default_scenario_single_run():
    log = run_simulation(DEFAULT_SCENARIO, DEFAULT_WEEKS, DEFAULT_SEED)
    assert log["Total Cost"].sum() == pytest.approx(4027.00)
    assert fill_rate_percent(log) == pytest.approx(100.0)
    assert int(log["Order Placed"].sum()) == 9
    assert (log["Backlog"] == 0).all()


def test_default_scenario_monte_carlo():
    result = run_monte_carlo(DEFAULT_SCENARIO, runs=200, horizon=DEFAULT_WEEKS, seed=DEFAULT_SEED)
    assert result.total_cost.mean() == pytest.approx(3773.15, abs=0.005)
    assert result.total_cost[0] == pytest.approx(4027.00)  # run 1 is the single run
    assert result.fill_rate.mean() == pytest.approx(0.99963, abs=1e-5)


def test_default_scenario_policy_grid():
    s_values = tuple(int(v) for v in np.unique(np.linspace(0, 150, GRID_POINTS).round()))
    q_values = tuple(int(v) for v in np.unique(np.linspace(20, 300, GRID_POINTS).round()))
    cost, fill = run_policy_grid(DEFAULT_SCENARIO, 100, DEFAULT_WEEKS, DEFAULT_SEED, s_values, q_values)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    assert (s_values[i], q_values[j]) == (32, 60)
    assert cost[i, j] == pytest.approx(2697.55, abs=0.005)
    assert fill[i, j] == pytest.approx(0.9661, abs=1e-4)


# Consistency of the implementations
@pytest.mark.parametrize("case", range(25))
def test_batch_matches_single_run_and_reference(case):
    """The fast batch simulation, the weekly log, and an independent re-implementation agree."""
    rng = np.random.default_rng(case)
    scenario = random_scenario(rng)
    horizon, seed, runs = int(rng.integers(1, 80)), int(rng.integers(0, 10**6)), 5
    demand = draw_demand(seed, runs, horizon, scenario.mean_demand, scenario.std_demand)
    batch = simulate_batch(demand, scenario, return_path=True)

    for run in range(runs):
        ref = reference_run(scenario, demand[run])
        assert batch.total_cost[run] == pytest.approx(ref["cost"], rel=1e-9, abs=1e-6)
        assert batch.orders[run] == ref["orders"]
        assert batch.backlog_weeks[run] == ref["backlog_weeks"]
        assert list(batch.path[run]) == ref["net_inventory"]
        expected_fill = ref["fulfilled"] / ref["demand"] if ref["demand"] else 1.0
        assert batch.fill_rate[run] == pytest.approx(expected_fill)

    log = run_simulation(scenario, horizon, seed)  # the weekly log is run 0 of the batch
    assert log["Total Cost"].sum() == pytest.approx(batch.total_cost[0], rel=1e-9, abs=1e-6)
    assert list(log["Net Inventory"]) == list(batch.path[0])
    assert fill_rate_percent(log) == pytest.approx(batch.fill_rate[0] * 100)


def test_policy_grid_cells_equal_single_policies():
    s_values, q_values = (0, 40, 90), (10, 60, 150)
    cost, fill = run_policy_grid(DEFAULT_SCENARIO, 20, 30, 7, s_values, q_values)
    demand = draw_demand(7, 20, 30, DEFAULT_SCENARIO.mean_demand, DEFAULT_SCENARIO.std_demand)
    for i, s in enumerate(s_values):
        for j, q in enumerate(q_values):
            scenario = dataclasses.replace(DEFAULT_SCENARIO, reorder_point=s, order_quantity=q)
            single = simulate_batch(demand, scenario)
            assert cost[i, j] == pytest.approx(single.total_cost.mean())
            assert fill[i, j] == pytest.approx(single.fill_rate.mean())


# Properties of the model
def test_same_seed_same_result_other_seed_other_result():
    first = run_simulation(DEFAULT_SCENARIO, 30, seed=1)
    assert first.equals(run_simulation(DEFAULT_SCENARIO, 30, seed=1))
    assert not first.equals(run_simulation(DEFAULT_SCENARIO, 30, seed=2))


def test_costs_do_not_change_the_dynamics():
    cheap = dataclasses.replace(DEFAULT_SCENARIO, holding_cost=1.0, backorder_cost=2.0, order_cost=10.0)
    expensive = dataclasses.replace(DEFAULT_SCENARIO, holding_cost=3.0, backorder_cost=8.0, order_cost=30.0)
    demand = draw_demand(5, 10, 40, cheap.mean_demand, cheap.std_demand)
    a, b = simulate_batch(demand, cheap, True), simulate_batch(demand, expensive, True)
    assert np.array_equal(a.path, b.path) and np.array_equal(a.orders, b.orders)
    on_hand = np.maximum(a.path, 0).sum(axis=1)
    backlog = np.maximum(-a.path, 0).sum(axis=1)
    assert np.allclose(b.total_cost, 3.0 * on_hand + 8.0 * backlog + 30.0 * b.orders)


def test_zero_demand_is_fully_served_without_backlog():
    scenario = dataclasses.replace(DEFAULT_SCENARIO, mean_demand=0.0, std_demand=0.0)
    log = run_simulation(scenario, 20, seed=3)
    assert fill_rate_percent(log) == 100.0
    assert (log["Backlog"] == 0).all() and (log["Demand"] == 0).all()

    batch = simulate_batch(draw_demand(3, 5, 20, 0.0, 0.0), scenario)  # the batch simulation follows the same rule
    assert (batch.fill_rate == 1.0).all() and (batch.backlog_weeks == 0).all()


def test_demand_is_whole_and_never_negative():
    demand = draw_demand(9, 50, 60, mean_demand=3.0, std_demand=10.0)
    assert demand.min() >= 0 and demand.dtype == np.int64


def test_fulfilled_demand_never_exceeds_demand():
    log = run_simulation(DEFAULT_SCENARIO, 52, seed=11)
    assert ((log["Fulfilled"] >= 0) & (log["Fulfilled"] <= log["Demand"])).all()


# Plausibility against theory
def test_best_order_quantity_is_close_to_the_eoq():
    """With low demand variability, the cheapest Q of the scan lies near the EOQ formula."""
    eoq = math.sqrt(2 * DEFAULT_SCENARIO.mean_demand * DEFAULT_SCENARIO.order_cost / DEFAULT_SCENARIO.holding_cost)
    s_values, q_values = tuple(range(0, 121, 10)), tuple(range(20, 121, 5))
    cost, _ = run_policy_grid(DEFAULT_SCENARIO, 300, DEFAULT_WEEKS, DEFAULT_SEED, s_values, q_values)
    _, j = np.unravel_index(np.argmin(cost), cost.shape)
    assert abs(q_values[j] - eoq) <= 10  # EOQ = 44.7


@pytest.mark.parametrize("mean, std", [(20, 5), (10, 8), (5, 5), (20, 30)])
def test_effective_mean_demand_matches_simulation(mean, std):
    simulated = draw_demand(1, 4000, 200, mean, std).mean()
    assert effective_mean_demand(mean, std) == pytest.approx(simulated, abs=0.1)


# Validation of inputs
@pytest.mark.parametrize("field, value", [
    ("order_quantity", 0),
    ("lead_time", 0),
    ("initial_inventory", -1),
    ("mean_demand", -1.0),
    ("std_demand", -0.5),
    ("holding_cost", -0.1),
    ("backorder_cost", -1.0),
    ("order_cost", -5.0),
    ("mean_demand", float("nan")),
    ("std_demand", float("inf")),
    ("order_cost", float("-inf")),
])
def test_invalid_scenarios_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(DEFAULT_SCENARIO, **{field: value})


def test_invalid_horizon_runs_and_order_quantities_are_rejected():
    with pytest.raises(ValueError):
        run_simulation(DEFAULT_SCENARIO, 0, 1)
    with pytest.raises(ValueError):
        draw_demand(1, 0, 10, 20.0, 5.0)
    with pytest.raises(ValueError):
        draw_demand(1, 10, 0, 20.0, 5.0)
    with pytest.raises(ValueError):
        simulate_batch(draw_demand(1, 4, 10, 20.0, 5.0), DEFAULT_SCENARIO, order_quantities=np.array([5, 0, 5, 5]))
