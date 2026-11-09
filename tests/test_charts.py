"""Tests of the figure builders (charts.py)."""

import numpy as np
import pytest

from charts import (
    cost_breakdown_chart,
    histogram_chart,
    inventory_band_chart,
    inventory_chart,
    policy_heatmap,
    split_at_zero,
    with_alpha,
)
from constants import DEFAULT_SCENARIO, DEFAULT_SEED, DEFAULT_WEEKS
from simulation import run_monte_carlo, run_policy_grid, run_simulation


def test_split_at_zero_inserts_the_zero_crossings():
    x, y = split_at_zero(np.array([0, 1, 2]), np.array([-1, 1, -1]))
    assert list(x) == [0, 0.5, 1, 1.5, 2]
    assert list(y) == [-1, 0, 1, 0, -1]


def test_split_at_zero_leaves_a_curve_without_crossings_unchanged():
    x, y = split_at_zero(np.array([0, 1, 2]), np.array([3, 0, 5]))
    assert list(x) == [0, 1, 2] and list(y) == [3, 0, 5]


def test_with_alpha_converts_hex_colors():
    assert with_alpha("#2563EB", 0.15) == "rgba(37, 99, 235, 0.15)"


def test_all_charts_are_built_for_the_default_scenario():
    log = run_simulation(DEFAULT_SCENARIO, DEFAULT_WEEKS, DEFAULT_SEED)
    mc = run_monte_carlo(DEFAULT_SCENARIO, 50, DEFAULT_WEEKS, DEFAULT_SEED)
    cost, fill = run_policy_grid(DEFAULT_SCENARIO, 20, DEFAULT_WEEKS, DEFAULT_SEED, (0, 50, 100), (20, 100, 200))

    inventory = inventory_chart(log, DEFAULT_SCENARIO.reorder_point, DEFAULT_SCENARIO.order_quantity)
    assert [t.name for t in inventory.data if t.showlegend is not False] == [
        "Net Inventory", "Inventory Position", "Reorder Point (s)", "Order Placed"]

    pie = cost_breakdown_chart(log)
    assert list(pie.data[0].labels) == ["Holding Cost", "Backorder Cost", "Ordering Cost"]
    assert sum(pie.data[0].values) == pytest.approx(log["Total Cost"].sum())

    reference_lines = [(float(mc.total_cost.mean()), "Mean", "#DC2626", "dash"),
                       (float(mc.total_cost[0]), "Single run", "#64748B", "solid")]
    histogram = histogram_chart(mc.total_cost, "#2563EB", reference_lines, "Total cost ($)")
    assert len(histogram.data[0].x) == 50 and len(histogram.layout.shapes) == 2
    # The lines are shapes and the percentages are fixed, so these legends must not hide anything on click
    assert histogram.layout.legend.itemclick is False and pie.layout.legend.itemclick is False

    band = inventory_band_chart(mc.path, DEFAULT_SCENARIO.reorder_point)
    shown = sorted((t for t in band.data if t.showlegend is not False), key=lambda t: t.legendrank)
    assert [t.name for t in shown] == ["Median", "5th–95th Percentile", "Reorder Point (s)"]  # order in the legend
    assert band.layout.legend.traceorder == "normal"

    heatmap = policy_heatmap((0, 50, 100), (20, 100, 200), cost, fill, best_policy=(50, 100), selected_policy=(50, 100))
    assert [t.name for t in heatmap.data[1:]] == ["Best Policy", "Selected Policy"]
    assert len(policy_heatmap((0, 50, 100), (20, 100, 200), cost, fill, best_policy=(50, 100)).data) == 2
