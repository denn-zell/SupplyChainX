"""Smoke tests of the Streamlit app (app.py), run headless with Streamlit's AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app(**changes) -> AppTest:
    """Start the app and set widgets by label prefix, e.g. run_app(**{"Demand Std Dev": 0.0})."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    for label, value in changes.items():
        widget = next(w for w in list(at.number_input) + list(at.slider) if w.label.startswith(label))
        widget.set_value(value)
    return at.run() if changes else at


def test_app_starts_with_the_default_results():
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "📊 Inventory Trajectory", "💰 Cost Breakdown", "🎲 Monte Carlo", "🎯 Policy Optimization", "📋 Weekly Log & Export"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total Supply Chain Cost"] == "$4,027.00"
    assert metrics["Fill Rate"] == "100.00%"
    assert metrics["Mean Total Cost"] == "$3,773.15"
    assert metrics["Best Policy (s, Q)"] == "(32, 60)"
    assert len(at.get("plotly_chart")) == 6


@pytest.mark.parametrize("changes", [
    {"Demand Std Dev": 0.0},
    {"Demand Std Dev": 30.0},
    {"Holding Cost": 0.0, "Backorder Penalty": 0.0, "Fixed Order Cost": 0.0},
    {"Reorder Point (s)": -50, "Order Quantity (Q)": 1},
    {"Reorder Point Range": (0, 0), "Order Quantity Range": (5, 5)},
    {"Replenishment Lead Time": 8, "Simulation Horizon": 104},
    {"Initial Inventory": 0},
])
def test_app_runs_without_errors_for_extreme_inputs(changes):
    assert not run_app(**changes).exception


def test_app_explains_instead_of_drawing_meaningless_charts():
    zero_costs = run_app(**{"Holding Cost": 0.0, "Backorder Penalty": 0.0, "Fixed Order Cost": 0.0})
    assert any("nothing to break down" in i.value for i in zero_costs.info)
    assert any("no best policy" in i.value for i in zero_costs.info)

    narrow = run_app(**{"Reorder Point Range": (0, 0)})
    assert any("at least two different values" in i.value for i in narrow.info)
    assert len(narrow.get("plotly_chart")) == 5


def test_app_warns_when_demand_is_clipped_strongly():
    assert not run_app().sidebar.warning
    assert run_app(**{"Demand Std Dev": 30.0}).sidebar.warning
