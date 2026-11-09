"""SupplyChainX: interactive stochastic inventory simulator.

Simulates a single-item, periodic-review (s, Q) inventory policy with
backordering and random weekly demand. The selected parameters are compared
against a fixed default scenario, and the app visualizes inventory dynamics,
the cost structure, and the underlying weekly log. A Monte Carlo analysis shows
how much the results vary across random demand histories, and a policy scan
searches a grid of reorder points and order quantities for low-cost policies.

Project layout:
    app.py         Streamlit user interface (this file)
    simulation.py  Simulation model (no Streamlit)
    charts.py      Plotly figure builders (no Streamlit)
    constants.py   Defaults, analysis settings, colors, chart layout, stylesheet, texts

Run with:
    streamlit run app.py
"""

from typing import Literal

import numpy as np
import streamlit as st

from charts import (
    cost_breakdown_chart,
    histogram_chart,
    inventory_band_chart,
    inventory_chart,
    policy_heatmap,
)
from constants import (
    AUTHOR,
    CACHE_MAX_ENTRIES,
    CLIPPING_WARNING_Z,
    COLORS,
    CUSTOM_CSS,
    DEFAULT_SCENARIO,
    DEFAULT_SEED,
    DEFAULT_WEEKS,
    GRID_POINTS,
    MAX_INPUT,
    MODEL_EXPLANATION,
    REPO_URL,
    TEXT_COLORS,
)
from simulation import (
    Scenario,
    effective_mean_demand,
    fill_rate_percent,
    run_monte_carlo,
    run_policy_grid,
    run_simulation,
)

# Page Configuration
st.set_page_config(
    page_title="SupplyChainX | Inventory Simulator",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Cached Batch Simulations:
# Streamlit re-runs this script on every widget change. Caching avoids recomputing the batch
# simulations for parameter sets that have already been evaluated.
cached_monte_carlo = st.cache_data(show_spinner=False, max_entries=CACHE_MAX_ENTRIES)(run_monte_carlo)
cached_policy_grid = st.cache_data(show_spinner=False, max_entries=CACHE_MAX_ENTRIES)(run_policy_grid)


def format_cost_delta(delta: float, reference: str = "Default") -> str:
    """Format a cost difference as a signed currency string for st.metric ('-' marks a decrease)."""
    sign = "-" if round(delta, 2) < 0 else "+"  # a difference that rounds to zero gets no minus sign
    return f"{sign}${abs(delta):,.2f} vs {reference}"


def metric_delta_color(
    delta: float, higher_is_better: bool, tolerance: float = 0.005
) -> Literal["off", "normal", "inverse"]:
    """Color mode for st.metric: neutral (gray) if there is no difference, otherwise by whether the change is good."""
    if abs(delta) < tolerance:
        return "off"
    return "normal" if higher_is_better else "inverse"


def format_fill_rate_delta(delta: float) -> str:
    """Format a fill rate difference in percentage points, without a minus sign on zero."""
    return f"{round(delta, 2) + 0.0:+.2f}% vs Default"


# App Header
st.title("📦 SupplyChainX")
st.markdown("##### Interactive Stochastic Inventory Simulator — (s, Q) Policy")

st.markdown("---")

# Sidebar Controls (Interactive Parameters)
st.sidebar.header("⚙️ Control Center")

with st.sidebar.expander("🗓️ Horizon & Stock", expanded=True):
    simulation_weeks = st.slider(
        "Simulation Horizon (Weeks)", min_value=10, max_value=104, value=DEFAULT_WEEKS, step=1,
        help="Number of weeks that are simulated.",
    )
    initial_inventory = st.number_input(
        "Initial Inventory", min_value=0, max_value=MAX_INPUT, value=DEFAULT_SCENARIO.initial_inventory, step=10,
        help="Units on hand at the start of the first week.",
    )

with st.sidebar.expander("🔁 Policy Parameters (s, Q)", expanded=True):
    s = st.number_input(
        "Reorder Point (s)", min_value=-50, max_value=MAX_INPUT, value=DEFAULT_SCENARIO.reorder_point, step=5,
        help="An order is placed when the inventory position (net inventory plus open orders) "
             "is at or below this level.",
    )
    Q = st.number_input(
        "Order Quantity (Q)", min_value=1, max_value=MAX_INPUT, value=DEFAULT_SCENARIO.order_quantity, step=10,
        help="Units per order. At most one order is placed per week.",
    )

with st.sidebar.expander("📈 Demand & Lead Time", expanded=True):
    mean_demand = st.number_input(
        r"Mean Weekly Demand ($\mu$)", min_value=1.0, max_value=float(MAX_INPUT),
        value=DEFAULT_SCENARIO.mean_demand, step=1.0,
        help="Average demand per week.",
    )
    std_demand = st.number_input(
        r"Demand Std Dev ($\sigma$)", min_value=0.0, max_value=float(MAX_INPUT),
        value=DEFAULT_SCENARIO.std_demand, step=0.5,
        help="Standard deviation of the weekly demand: how strongly the demand varies from week to week.",
    )

    # Negative demand draws are clipped at 0, which raises the effective mean demand.
    # Warn when this affects more than ~1% of the draws.
    if std_demand > 0 and mean_demand / std_demand < CLIPPING_WARNING_Z:
        eff_mean = effective_mean_demand(mean_demand, std_demand)
        st.warning(
            f"σ is large relative to μ: negative demand is set to 0, so the effective mean demand "
            f"is ≈ {eff_mean:.1f} instead of {mean_demand:.1f}."
        )

    lead_time = st.slider(
        r"Replenishment Lead Time ($L$ Weeks)", min_value=1, max_value=8, value=DEFAULT_SCENARIO.lead_time,
        help="Weeks between placing an order and its arrival. An order never arrives in the week it is placed.",
    )
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=999_999_999, value=DEFAULT_SEED, step=1,
        help="The same seed reproduces the same random demand histories.",
    )

with st.sidebar.expander("💵 Cost Structure ($)", expanded=True):
    holding_cost = st.number_input(
        r"Holding Cost ($/Unit/Week)", min_value=0.0, max_value=float(MAX_INPUT),
        value=DEFAULT_SCENARIO.holding_cost, step=0.1,
        help="Cost per unit on hand at the end of a week.",
    )
    backorder_cost = st.number_input(
        r"Backorder Penalty ($/Unit/Week)", min_value=0.0, max_value=float(MAX_INPUT),
        value=DEFAULT_SCENARIO.backorder_cost, step=0.5,
        help="Cost per backordered unit at the end of a week.",
    )
    order_cost = st.number_input(
        r"Fixed Order Cost ($/Order)", min_value=0.0, max_value=float(MAX_INPUT),
        value=DEFAULT_SCENARIO.order_cost, step=5.0,
        help="Cost per order placed, independent of the order quantity.",
    )

# Run Simulations
scenario = Scenario(
    reorder_point=s,
    order_quantity=Q,
    mean_demand=mean_demand,
    std_demand=std_demand,
    lead_time=lead_time,
    holding_cost=holding_cost,
    backorder_cost=backorder_cost,
    order_cost=order_cost,
    initial_inventory=initial_inventory,
)

# Selected scenario and default baseline (same horizon, same random draws)
df = run_simulation(scenario, simulation_weeks, seed)
df_base = run_simulation(DEFAULT_SCENARIO, simulation_weeks, seed)

# KPI Calculations & Deltas
fill_rate = fill_rate_percent(df)
total_cost = df['Total Cost'].sum()

base_fill_rate = fill_rate_percent(df_base)
base_cost = df_base['Total Cost'].sum()

cost_delta = total_cost - base_cost
fill_rate_delta = fill_rate - base_fill_rate

# KPI Metric Cards
m1, m2, m3, m4 = st.columns(4)

m1.metric("Total Supply Chain Cost", f"${total_cost:,.2f}",
          delta=format_cost_delta(cost_delta), delta_color=metric_delta_color(cost_delta, higher_is_better=False),
          help="Holding, backorder, and ordering costs of the single simulation run, summed over all weeks. "
               "The change refers to the default scenario.")
m2.metric("Fill Rate", f"{fill_rate:.2f}%", delta=format_fill_rate_delta(fill_rate_delta),
          delta_color=metric_delta_color(fill_rate_delta, higher_is_better=True),
          help="Share of the total demand that was served immediately from stock in the single simulation run. "
               "The change refers to the default scenario.")
m3.metric("Backlog Unit-Weeks", f"{df['Backlog'].sum():,}",
          help="Backlog summed over all weeks: one unit on backorder for one week counts as one unit-week.")
m4.metric("Total Orders Placed", f"{df['Order Placed'].sum():,}")

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("ℹ️ How this model works"):
    st.markdown(MODEL_EXPLANATION)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Inventory Trajectory",
    "💰 Cost Breakdown",
    "🎲 Monte Carlo",
    "🎯 Policy Optimization",
    "📋 Weekly Log & Export",
])

with tab1:
    st.subheader("Net Inventory over Time (Stock and Backlog)")
    st.markdown(
        "Visualizing simulated stock fluctuations, backorder depths, and replenishment triggers "
        "against the reorder point $s$."
    )

    st.plotly_chart(inventory_chart(df, s, Q), width="stretch")

with tab2:
    st.subheader("Cost Breakdown by Category")
    st.markdown(
        "Detailed breakdown of operational expenditures across holding, backorder penalties, "
        "and fixed ordering costs."
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        if total_cost > 0:
            st.plotly_chart(cost_breakdown_chart(df), width="stretch")
        else:
            st.info("No costs occurred in this scenario, so there is nothing to break down.")

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 💡 Cost Driver Insights")

        # Category names double as column names of the weekly log
        cost_categories = [
            ("Holding Cost", TEXT_COLORS["holding"]),
            ("Backorder Cost", TEXT_COLORS["backlog"]),
            ("Ordering Cost", TEXT_COLORS["order"]),
        ]
        for label, color in cost_categories:
            # Guard against division by zero when no costs occurred
            share = df[label].sum() / total_cost * 100 if total_cost > 0 else 0.0
            st.markdown(
                f"<span style='color: {color}; font-weight: 600;'>{label}:</span> {share:.1f}% of total.",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "Adjust the reorder point and the order quantity in the sidebar to see how they shift "
            "the balance between holding, backorder, and ordering costs."
        )

with tab3:
    st.subheader("Monte Carlo Analysis")
    st.markdown(
        "A single simulation is only one possible outcome. This tab repeats the simulation with many "
        "independent random demand histories to show how much the results vary. "
        "Run 1 uses the same demand as the single run shown in the other tabs."
    )

    mc_runs = st.slider(
        "Number of Simulation Runs", min_value=50, max_value=1000, value=200, step=50,
        help="More runs give a more precise mean but take longer to compute.",
    )

    # Selected policy and default baseline, both on the same random demand draws
    mc = cached_monte_carlo(scenario, mc_runs, simulation_weeks, seed)
    mc_base = cached_monte_carlo(DEFAULT_SCENARIO, mc_runs, simulation_weeks, seed)

    mc_cost = mc.total_cost
    mc_fill = mc.fill_rate * 100
    mean_cost = mc_cost.mean()
    mean_fill = mc_fill.mean()
    cost_p5, cost_p95 = np.percentile(mc_cost, [5, 95])
    cost_gap = mean_cost - mc_base.total_cost.mean()
    fill_gap = mean_fill - mc_base.fill_rate.mean() * 100
    backorder_share = (mc.backlog_weeks > 0).mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mean Total Cost", f"${mean_cost:,.2f}", delta=format_cost_delta(cost_gap),
              delta_color=metric_delta_color(cost_gap, higher_is_better=False),
              help="Average total cost over all runs. The change refers to the default scenario.")
    # Only one currency sign per value: Streamlit would render a pair of $ signs as math
    k2.metric("Cost Range, P5–P95 ($)", f"{cost_p5:,.0f} – {cost_p95:,.0f}",
              help="5th and 95th percentile of the total cost across all runs.")
    k3.metric("Mean Fill Rate", f"{mean_fill:.2f}%", delta=format_fill_rate_delta(fill_gap),
              delta_color=metric_delta_color(fill_gap, higher_is_better=True),
              help="Average fill rate over all runs. The change refers to the default scenario.")
    k4.metric("Runs with Backorders", f"{backorder_share:.0f}%",
              help="Share of runs with at least one week of backorders.")

    # Precision of the estimated mean: 95% confidence half-width (normal approximation)
    ci_half = 1.96 * mc_cost.std(ddof=1) / np.sqrt(mc_runs)
    st.caption(
        f"With {mc_runs} runs, the mean total cost is accurate to about ±${ci_half:,.0f} (95% confidence)."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_cost, col_fill = st.columns(2)
    with col_cost:
        st.markdown("##### Total Cost Distribution")
        fig_cost = histogram_chart(mc_cost, COLORS["stock"], [
            (mean_cost, f"Mean: ${mean_cost:,.0f}", COLORS["mean"], "dash"),
            (mc_cost[0], f"Single run: ${mc_cost[0]:,.0f}", COLORS["position"], "solid"),
        ], "Total cost ($)", x_tickformat=",")
        st.plotly_chart(fig_cost, width="stretch")

    with col_fill:
        st.markdown("##### Fill Rate Distribution")
        fig_fill = histogram_chart(mc_fill, COLORS["order"], [
            (mean_fill, f"Mean: {mean_fill:.2f}%", COLORS["mean"], "dash"),
            (mc_fill[0], f"Single run: {mc_fill[0]:.2f}%", COLORS["position"], "solid"),
        ], "Fill rate (%)")
        st.plotly_chart(fig_fill, width="stretch")

    st.markdown("##### Net Inventory Across All Runs")
    assert mc.path is not None  # run_monte_carlo always records the net inventory paths
    st.plotly_chart(inventory_band_chart(mc.path, s), width="stretch")

with tab4:
    st.subheader("Policy Optimization: Reorder Point and Order Quantity")
    st.markdown(
        "Scans a grid of reorder points (s) and order quantities (Q) using the demand, lead time, and "
        "cost settings from the sidebar. Every grid cell is evaluated on the same random demand histories, "
        "so all policies face identical conditions. Lower cost is better. Near the optimum the cost surface "
        "is flat, so neighboring policies often differ only slightly."
    )

    g1, g2, g3 = st.columns(3)
    s_lo, s_hi = g1.slider(
        "Reorder Point Range (s)", min_value=-50, max_value=400, value=(0, 150), step=5,
        help="Range of reorder points covered by the scan.",
    )
    q_lo, q_hi = g2.slider(
        "Order Quantity Range (Q)", min_value=5, max_value=500, value=(20, 300), step=5,
        help="Range of order quantities covered by the scan.",
    )
    grid_runs = g3.slider(
        "Runs per Policy", min_value=50, max_value=300, value=100, step=50,
        help="Number of random demand histories on which every policy is evaluated.",
    )

    # Evenly spaced grid points per axis, rounded to whole units
    s_values = tuple(int(v) for v in np.unique(np.linspace(s_lo, s_hi, GRID_POINTS).round()))
    q_values = tuple(int(v) for v in np.unique(np.linspace(q_lo, q_hi, GRID_POINTS).round()))

    grid_cost, grid_fill = cached_policy_grid(scenario, grid_runs, simulation_weeks, seed, s_values, q_values)
    # The selected policy on the same demand histories, as reference for the best grid policy
    sel_cost, _ = cached_policy_grid(scenario, grid_runs, simulation_weeks, seed, (s,), (Q,))

    if len(s_values) < 2 or len(q_values) < 2:
        st.info("Select a range with at least two different values for both the reorder point and the order "
                "quantity to compare policies.")
    elif np.ptp(grid_cost) == 0:
        st.info("All scanned policies have the same cost, so there is no best policy to show.")
    else:
        best_i, best_j = np.unravel_index(np.argmin(grid_cost), grid_cost.shape)
        best_s, best_q = s_values[best_i], q_values[best_j]
        best_cost = grid_cost[best_i, best_j]
        cost_saving = best_cost - sel_cost[0, 0]

        o1, o2, o3 = st.columns(3)
        o1.metric("Lowest Mean Cost in Grid", f"${best_cost:,.2f}",
                  delta=format_cost_delta(cost_saving, "Selected"),
                  delta_color=metric_delta_color(cost_saving, higher_is_better=False),
                  help="Lowest mean total cost of all scanned policies. "
                       "The change is relative to the policy selected in the sidebar.")
        o2.metric("Best Policy (s, Q)", f"({best_s}, {best_q})",
                  help="Reorder point and order quantity with the lowest mean total cost.")
        o3.metric("Mean Fill Rate at Best Policy", f"{grid_fill[best_i, best_j] * 100:.2f}%",
                  help="Mean fill rate of the policy with the lowest cost.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Mark the selected policy only if it lies inside the scanned ranges
        selected_in_range = s_lo <= s <= s_hi and q_lo <= Q <= q_hi
        fig_heat = policy_heatmap(
            s_values, q_values, grid_cost, grid_fill,
            best_policy=(best_s, best_q),
            selected_policy=(s, Q) if selected_in_range else None,
        )
        st.plotly_chart(fig_heat, width="stretch")

        # A minimum on the border of the grid suggests that a better policy may lie outside the scanned range
        on_edge = best_i in (0, len(s_values) - 1) or best_j in (0, len(q_values) - 1)
        if on_edge:
            st.info("The best policy lies on the edge of the scanned range. A better policy may exist outside it, "
                    "so consider widening the ranges.")
        st.caption(
            f"Grid: {len(s_values)} × {len(q_values)} policies, {grid_runs} runs each. "
            "The gold star marks the lowest-cost policy found; the white diamond marks the policy selected "
            "in the sidebar. The color scale is capped at the 75th percentile of the scanned costs, "
            "so darker cells may be considerably more expensive."
        )
        if not selected_in_range:
            st.caption("The policy selected in the sidebar lies outside the scanned ranges and is therefore not shown.")

with tab5:
    st.subheader("Weekly Simulation Log & Data Export")
    st.markdown("Complete period-by-period breakdown of the stochastic simulation.")

    # Cost columns as currency; the CSV export below keeps the plain numbers
    cost_columns = ("Holding Cost", "Backorder Cost", "Ordering Cost", "Total Cost")
    st.dataframe(
        df, width="stretch", height=400, hide_index=True,
        column_config={column: st.column_config.NumberColumn(format="$%.2f") for column in cost_columns},
    )

    st.markdown("<br>", unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete Simulation Dataset (CSV)",
        data=csv,
        file_name="SupplyChainX_Simulation_Log.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
footer = f"Built by {AUTHOR}"
if REPO_URL:
    footer += f" · [Source code on GitHub]({REPO_URL})"
st.caption(footer)
