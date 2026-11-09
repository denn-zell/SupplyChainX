"""Simulation model for SupplyChainX.

A single-item, periodic-review (s, Q) inventory policy with backordering and random weekly demand.
This module contains only the model (NumPy and pandas, no Streamlit), so it can be tested on its own.

- Scenario: the parameters of one inventory scenario.
- run_simulation: one run with a detailed weekly log.
- fill_rate_percent: fill rate of such a weekly log.
- draw_demand: random demand histories for many runs (row 0 matches run_simulation).
- simulate_batch, BatchResult: many runs at once, aggregates only.
- run_monte_carlo, run_policy_grid: batch analyses built on simulate_batch.
- effective_mean_demand: mean demand after clipping negative draws at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Scenario
@dataclass(frozen=True)
class Scenario:
    """Parameters of one inventory scenario: (s, Q) policy, demand, lead time, costs, and initial stock.

    Attributes:
        reorder_point: Reorder point s.
        order_quantity: Order quantity Q (at most one order per week).
        mean_demand: Mean weekly demand.
        std_demand: Standard deviation of weekly demand. Demand is normally distributed,
            rounded to whole units, and clipped at 0.
        lead_time: Replenishment lead time in weeks.
        holding_cost: Holding cost per unit on hand per week.
        backorder_cost: Backorder penalty per backordered unit per week.
        order_cost: Fixed cost per order placed.
        initial_inventory: Initial inventory on hand.
    """

    reorder_point: int
    order_quantity: int
    mean_demand: float
    std_demand: float
    lead_time: int
    holding_cost: float
    backorder_cost: float
    order_cost: float
    initial_inventory: int

    def __post_init__(self) -> None:
        """Reject parameter values for which the model is meaningless or inconsistent."""
        float_fields = (self.mean_demand, self.std_demand, self.holding_cost, self.backorder_cost, self.order_cost)
        if not all(math.isfinite(value) for value in float_fields):
            raise ValueError("demand and cost parameters must be finite numbers")
        if self.order_quantity < 1:
            raise ValueError("order_quantity must be at least 1")
        if self.initial_inventory < 0:
            raise ValueError("initial_inventory must not be negative")
        if self.lead_time < 1:
            raise ValueError("lead_time must be at least 1 week (an order never arrives in the week it is placed)")
        if self.mean_demand < 0 or self.std_demand < 0:
            raise ValueError("mean_demand and std_demand must not be negative")
        if min(self.holding_cost, self.backorder_cost, self.order_cost) < 0:
            raise ValueError("costs must not be negative")


# Simulation Model
def run_simulation(scenario: Scenario, horizon: int, seed: int) -> pd.DataFrame:
    """Simulate a periodic-review (s, Q) policy with backorders and return the weekly log.

    Event order within each week:
      1. Deliveries scheduled for this week arrive.
      2. Random demand occurs and is served from stock on hand; unmet demand is backordered.
      3. If the inventory position (net inventory plus open orders) is at or below s,
         one order of Q units is placed. It arrives `lead_time` weeks later.
      4. Holding, backorder, and ordering costs are charged on the end-of-week state.

    Args:
        scenario: Policy, demand, lead time, cost, and initial stock parameters.
        horizon: Number of weeks to simulate.
        seed: Seed for the random demand draws.

    Returns:
        A DataFrame with one row per week: net inventory and inventory position, demand,
        fulfilled demand, backlog, arrivals, order flag, and the cost components.

    Raises:
        ValueError: If horizon is less than 1.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1 week")

    # Seeded local generator: the same seed yields the same standard-normal sequence, so the selected
    # scenario and the default baseline are compared on identical random demand (common random numbers).
    # RandomState is used instead of default_rng because its stream is frozen across NumPy versions.
    rng = np.random.RandomState(seed)
    net_inv = scenario.initial_inventory  # Stock on hand minus backlog (negative = backlog)
    pend_orders: list[dict[str, int]] = []  # Open orders as {'qty': ..., 'delivery_week': ...}
    log = []

    for week in range(1, horizon + 1):
        # 1. Receive deliveries due this week, before demand occurs
        arrived = sum(o['qty'] for o in pend_orders if o['delivery_week'] == week)
        pend_orders = [o for o in pend_orders if o['delivery_week'] > week]
        net_inv += arrived

        # 2. Draw demand (whole units, never negative)
        demand = max(0, int(round(rng.normal(scenario.mean_demand, scenario.std_demand))))

        # Serve it from stock on hand. Net inventory already reflects earlier backlog, so new arrivals
        # clear the backlog first. 'Fulfilled' counts only demand served immediately from stock; it is
        # the basis of the fill rate.
        fulfilled = min(demand, max(0, net_inv))
        net_inv -= demand
        backlog = max(0, -net_inv)

        # 3. Reorder decision based on the inventory position (net inventory plus open orders)
        position = net_inv + sum(o['qty'] for o in pend_orders)

        ordered = position <= scenario.reorder_point
        if ordered:
            pend_orders.append({'qty': scenario.order_quantity, 'delivery_week': week + scenario.lead_time})

        # 4. Costs on the end-of-week state (holding applies to stock on hand only)
        holding = max(0, net_inv) * scenario.holding_cost
        backorder = backlog * scenario.backorder_cost
        ordering = scenario.order_cost if ordered else 0.0

        log.append({
            'Week': week, 'Net Inventory': net_inv, 'Inventory Position': position,
            'Demand': demand, 'Fulfilled': fulfilled, 'Backlog': backlog, 'Arrived Qty': arrived,
            'Order Placed': ordered, 'Holding Cost': holding,
            'Backorder Cost': backorder, 'Ordering Cost': ordering,
            'Total Cost': holding + backorder + ordering
        })
    return pd.DataFrame(log)


def fill_rate_percent(log: pd.DataFrame) -> float:
    """Fill rate of a weekly log in percent: demand served immediately from stock / total demand."""
    total_demand = log['Demand'].sum()
    # A run without any demand counts as fully served
    return log['Fulfilled'].sum() / total_demand * 100 if total_demand > 0 else 100.0


# Batch Simulation (Monte Carlo and policy scan)
def draw_demand(seed: int, runs: int, horizon: int, mean_demand: float, std_demand: float) -> np.ndarray:
    """Draw a (runs x horizon) matrix of weekly demand: normal, rounded to whole units, clipped at 0.

    The generator and the sampling order match run_simulation: for the same seed, mean, standard
    deviation, and horizon, row 0 is exactly the demand history of the single run.

    Args:
        seed: Seed for the random draws.
        runs: Number of independent demand histories (rows).
        horizon: Number of weeks per history (columns).
        mean_demand: Mean weekly demand before clipping.
        std_demand: Standard deviation of the weekly demand before clipping.

    Returns:
        Integer array of shape (runs, horizon).

    Raises:
        ValueError: If runs or horizon is less than 1.
    """
    if runs < 1 or horizon < 1:
        raise ValueError("runs and horizon must be at least 1")
    z = np.random.RandomState(seed).standard_normal((runs, horizon))
    return np.maximum(0, np.round(mean_demand + std_demand * z)).astype(np.int64)


@dataclass(frozen=True)
class BatchResult:
    """Per-run results of simulate_batch, one value per simulated run.

    Attributes:
        total_cost: Total cost over the horizon.
        fill_rate: Fill rate (0-1): demand served immediately from stock / total demand.
        backlog_weeks: Number of weeks with backlog.
        orders: Number of orders placed.
        path: End-of-week net inventory, shape (n_runs, horizon); None unless requested.
    """

    total_cost: np.ndarray
    fill_rate: np.ndarray
    backlog_weeks: np.ndarray
    orders: np.ndarray
    path: np.ndarray | None = None


def simulate_batch(
    demand: np.ndarray,
    scenario: Scenario,
    return_path: bool = False,
    reorder_points: np.ndarray | None = None,
    order_quantities: np.ndarray | None = None,
) -> BatchResult:
    """Simulate many independent runs of the (s, Q) model at once (vectorized over runs).

    Implements the same weekly logic as run_simulation but keeps only aggregates per run,
    which is much faster than building a weekly log for every run.

    Args:
        demand: Integer array (n_runs, horizon) of weekly demand, one row per run.
        scenario: Policy, lead time, cost, and initial stock parameters. Its demand parameters are
            not used here, because the demand is given explicitly.
        return_path: If True, also return the end-of-week net inventory of every run.
        reorder_points: Optional array with one reorder point per run, replacing scenario.reorder_point.
        order_quantities: Optional array with one order quantity per run, replacing
            scenario.order_quantity. Together with reorder_points this evaluates different policies
            in parallel.

    Returns:
        A BatchResult with one value per run. Its path is only filled if return_path is True.

    Raises:
        ValueError: If any order quantity is less than 1.
    """
    n, horizon = demand.shape
    lead_time = scenario.lead_time
    s_arr = np.broadcast_to(scenario.reorder_point if reorder_points is None else reorder_points, (n,))
    q_arr = np.broadcast_to(scenario.order_quantity if order_quantities is None else order_quantities, (n,))
    if np.any(q_arr < 1):
        raise ValueError("order quantities must be at least 1")

    net_inv = np.full(n, scenario.initial_inventory, dtype=np.int64)  # Stock on hand minus backlog
    open_qty = np.zeros(n, dtype=np.int64)  # Total quantity on order but not yet delivered
    # Column k holds the quantity that arrives in week k (orders arrive `lead_time` weeks after placement)
    arrivals = np.zeros((n, horizon + lead_time + 1), dtype=np.int64)

    # Aggregates over the horizon (on_hand_sum and backlog_sum are in unit-weeks)
    fulfilled = np.zeros(n, dtype=np.int64)
    on_hand_sum = np.zeros(n, dtype=np.int64)
    backlog_sum = np.zeros(n, dtype=np.int64)
    backlog_weeks = np.zeros(n, dtype=np.int64)
    orders = np.zeros(n, dtype=np.int64)
    path = np.zeros((n, horizon), dtype=np.int64) if return_path else None

    for week in range(1, horizon + 1):
        # 1. Deliveries due this week
        arrived = arrivals[:, week]
        net_inv = net_inv + arrived
        open_qty = open_qty - arrived

        # 2. Demand, served from stock on hand; the remainder becomes backlog
        week_demand = demand[:, week - 1]
        fulfilled += np.minimum(week_demand, np.maximum(net_inv, 0))
        net_inv = net_inv - week_demand
        backlog = np.maximum(-net_inv, 0)
        on_hand_sum += np.maximum(net_inv, 0)
        backlog_sum += backlog
        backlog_weeks += backlog > 0

        # 3. Reorder decision on the inventory position (net inventory plus open orders)
        ordered = (net_inv + open_qty) <= s_arr
        placed = np.flatnonzero(ordered)
        arrivals[placed, week + lead_time] += q_arr[placed]
        open_qty = open_qty + np.where(ordered, q_arr, 0)
        orders += ordered

        if path is not None:
            path[:, week - 1] = net_inv

    total_demand = demand.sum(axis=1)
    # Fill rate is 1 for a run without any demand
    fill_rate = np.divide(fulfilled, total_demand, out=np.ones(n), where=total_demand > 0)
    total_cost = (scenario.holding_cost * on_hand_sum + scenario.backorder_cost * backlog_sum
                  + scenario.order_cost * orders)
    return BatchResult(total_cost=total_cost, fill_rate=fill_rate, backlog_weeks=backlog_weeks,
                       orders=orders, path=path)


def run_monte_carlo(scenario: Scenario, runs: int, horizon: int, seed: int) -> BatchResult:
    """Run `runs` independent simulations of one scenario and return the per-run results.

    The parameters are the same as for run_simulation, except that `runs` independent demand
    histories are drawn at once. The returned BatchResult includes the net inventory paths.
    """
    demand = draw_demand(seed, runs, horizon, scenario.mean_demand, scenario.std_demand)
    return simulate_batch(demand, scenario, return_path=True)


def run_policy_grid(
    scenario: Scenario,
    runs: int,
    horizon: int,
    seed: int,
    s_values: tuple[int, ...],
    q_values: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate every (s, Q) combination of a grid, each on the same `runs` demand histories.

    Using identical random demand for all grid cells (common random numbers) means that every policy
    faces the same conditions. This reduces the noise in the cost differences between similar policies.

    Args:
        scenario: Lead time, cost, demand, and initial stock parameters. Its reorder point and order
            quantity are replaced by the grid values.
        runs: Number of demand histories per grid cell.
        horizon: Number of weeks per run.
        seed: Seed for the random demand draws.
        s_values: Reorder points of the grid rows.
        q_values: Order quantities of the grid columns.

    Returns:
        Mean total cost and mean fill rate per grid cell, each of shape (len(s_values), len(q_values)).
    """
    demand = draw_demand(seed, runs, horizon, scenario.mean_demand, scenario.std_demand)
    s_grid, q_grid = np.meshgrid(s_values, q_values, indexing="ij")
    cells = s_grid.size

    # All cells and runs are simulated in one batch: cell k occupies rows k*runs ... (k+1)*runs - 1
    result = simulate_batch(
        np.tile(demand, (cells, 1)),
        scenario,
        reorder_points=np.repeat(s_grid.ravel(), runs),
        order_quantities=np.repeat(q_grid.ravel(), runs),
    )
    mean_cost = result.total_cost.reshape(cells, runs).mean(axis=1).reshape(s_grid.shape)
    mean_fill = result.fill_rate.reshape(cells, runs).mean(axis=1).reshape(s_grid.shape)
    return mean_cost, mean_fill


# Demand Statistics
def effective_mean_demand(mean_demand: float, std_demand: float) -> float:
    """Expected weekly demand after clipping a normal distribution at 0, i.e. E[max(0, X)].

    For X ~ N(mean, std²) and z = mean / std:  E[max(0, X)] = mean·Φ(z) + std·φ(z).
    Rounding to whole units is ignored, so the value is approximate. Requires std_demand > 0.
    """
    z = mean_demand / std_demand
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return mean_demand * cdf + std_demand * pdf
