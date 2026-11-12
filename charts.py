"""Plotly figure builders for SupplyChainX.

The chart functions take plain data (DataFrames, arrays, numbers) and return a Plotly figure.
The module contains no Streamlit code, so the figures can be reused and tested independently.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import CHART_LAYOUT, COLORS, LEGEND, TEXT_COLORS

# A labeled vertical reference line: (x position, label, color, dash style)
ReferenceLine = tuple[float, str, str, str]


# Helpers
def split_at_zero(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert an interpolated point at every zero crossing of a piecewise-linear curve.

    Splitting the curve exactly at y = 0 allows the positive and the negative part to be
    drawn in different colors without gaps or overlaps.

    Args:
        x: X positions of the samples.
        y: Y values of the samples.

    Returns:
        The extended x and y values as float arrays.
    """
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        # Sign change between two samples: linearly interpolate where the curve crosses zero
        if (y[i - 1] < 0 < y[i]) or (y[i - 1] > 0 > y[i]):
            x_cross = x[i - 1] + (0 - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1])
            xs.append(x_cross)
            ys.append(0.0)
        xs.append(x[i])
        ys.append(y[i])
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert a '#RRGGBB' color to an 'rgba(r, g, b, alpha)' string (used for area fills)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def bold_label(text: str, color: str) -> str:
    """Wrap text in bold, colored HTML for a Plotly tooltip."""
    return f'<span style="color:{color}"><b>{text}</b></span>'


def reorder_point_trace(x: pd.Series | np.ndarray, reorder_point: int) -> go.Scatter:
    """Dashed horizontal line at the reorder point s over the given x positions."""
    return go.Scatter(
        x=x, y=[reorder_point] * len(x), mode='lines',
        name='Reorder Point (s)', line=dict(color=COLORS["reorder"], width=2, dash='dash'),
        hovertemplate=bold_label('Reorder Point (s): %{y}', TEXT_COLORS["reorder"]) + '<extra></extra>'
    )


def add_reference_lines(fig: go.Figure, lines: list[ReferenceLine]) -> None:
    """Add labeled vertical reference lines to a histogram.

    The lines are drawn as shapes. Their labels appear in the legend, via dummy traces without data.
    Labels next to the lines would be cut off at the edge of narrow charts; the legend wraps instead.

    Args:
        fig: The histogram figure.
        lines: List of (x, label, color, dash) tuples.
    """
    for x, label, color, dash in lines:
        fig.add_vline(x=x, line_color=color, line_dash=dash, line_width=2)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', name=label,
            line=dict(color=color, dash=dash, width=2), hoverinfo='skip'
        ))


# Single Run Charts
def inventory_chart(df: pd.DataFrame, reorder_point: int, order_quantity: int) -> go.Figure:
    """Weekly inventory trajectory of the single run.

    Net inventory is drawn as a blue part (stock) and a red part (backlog), together with the
    inventory position, the reorder point, and markers for the weeks in which an order was placed.

    Args:
        df: Weekly log returned by simulation.run_simulation.
        reorder_point: Reorder point s (horizontal line).
        order_quantity: Order quantity Q (shown in the tooltip of weeks with an order).

    Returns:
        The Plotly figure.
    """
    # Plot helpers derived from the log:
    #   order_marker_y: y-position of the order markers (NaN = no order that week)
    #   marker_color:   blue for stock, red for backlog
    #   *_hover:        pre-built HTML snippets passed to Plotly via customdata, which allows
    #                   per-point tooltip colors and the order quantity in the tooltip
    order_marker_y = np.where(df['Order Placed'], df['Inventory Position'], np.nan)
    marker_color = np.where(df['Net Inventory'] >= 0, COLORS["stock"], COLORS["backlog"])
    order_hover = [
        '  ' + bold_label(f'▲ Order placed: {order_quantity}', TEXT_COLORS["order"]) if placed else ''
        for placed in df['Order Placed']
    ]
    net_hover = [
        bold_label(f'Net Inventory: {v}', TEXT_COLORS["stock"] if v >= 0 else TEXT_COLORS["backlog"])
        for v in df['Net Inventory']
    ]

    fig = go.Figure()

    # The curve is split at its zero crossings so that the blue (stock) and the red (backlog) part end
    # exactly at y = 0 and never overlap. All net inventory traces share the legend group 'net', so a
    # single legend click toggles them together.
    fill_x, fill_y = split_at_zero(df['Week'].to_numpy(), df['Net Inventory'].to_numpy())

    # Area fills between the curve and zero (no legend entry, no tooltip)
    fig.add_trace(go.Scatter(
        x=fill_x, y=np.clip(fill_y, 0, None),
        mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor=with_alpha(COLORS["stock"], 0.08),
        legendgroup='net',
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fill_x, y=np.clip(fill_y, None, 0),
        mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor=with_alpha(COLORS["backlog"], 0.10),
        legendgroup='net',
        hoverinfo='skip',
        showlegend=False
    ))

    # Net inventory line: blue above zero (carries the legend entry), red below zero.
    # The zero-crossing points belong to both parts, so the line stays continuous.
    fig.add_trace(go.Scatter(
        x=fill_x, y=np.where(fill_y >= 0, fill_y, np.nan),
        mode='lines',
        name='Net Inventory',
        legendgroup='net',
        line=dict(color=COLORS["stock"], width=2.5),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=fill_x, y=np.where(fill_y <= 0, fill_y, np.nan),
        mode='lines',
        legendgroup='net',
        line=dict(color=COLORS["backlog"], width=2.5),
        hoverinfo='skip',
        showlegend=False
    ))

    # Net inventory markers; they carry the tooltip because the line and fill traces are hover-less
    fig.add_trace(go.Scatter(
        x=df['Week'], y=df['Net Inventory'],
        mode='markers',
        name='Net Inventory',
        legendgroup='net',
        showlegend=False,
        marker=dict(size=6, color=marker_color),
        customdata=net_hover,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    # Inventory position (net inventory plus open orders): the quantity that is compared with s.
    # Its tooltip also reports an order placed that week. The order markers have no tooltip of their own
    # because under hovermode='x' they would also show up for neighboring weeks.
    fig.add_trace(go.Scatter(
        x=df['Week'], y=df['Inventory Position'],
        mode='lines',
        name='Inventory Position',
        line=dict(color=COLORS["position"], width=1.5, dash='dot'),
        customdata=order_hover,
        hovertemplate=bold_label('Inventory Position: %{y}', TEXT_COLORS["position"]) + '%{customdata}<extra></extra>'
    ))

    fig.add_trace(reorder_point_trace(df['Week'], reorder_point))

    # Order markers: placed on the inventory position line in the week the order is triggered
    fig.add_trace(go.Scatter(
        x=df['Week'], y=order_marker_y,
        mode='markers',
        name='Order Placed',
        marker=dict(color=COLORS["order"], size=11, symbol='triangle-up'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        xaxis=dict(
            title="Simulation week",
            showspikes=True,
            spikemode="toaxis",
            spikesnap="cursor",
            spikecolor=COLORS["spike"],
            spikethickness=1,
            spikedash="dot"
        ),
        yaxis_title="Units (positive = stock, negative = backlog)",
        hovermode="x",  # tooltips for all traces at the hovered week
    )
    return fig


def cost_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of the total cost by category (holding, backorder, ordering) of the single run."""
    # Total cost per category over the simulation horizon; the hues match the cost labels in the
    # insights panel of the app (which use darker shades for readable text)
    amounts = [df['Holding Cost'].sum(), df['Backorder Cost'].sum(), df['Ordering Cost'].sum()]
    total = sum(amounts)
    fig = go.Figure(data=[go.Pie(
        labels=['Holding Cost', 'Backorder Cost', 'Ordering Cost'],
        values=amounts,
        hole=.5,
        sort=False,  # fixed order, so that legend, colors, and the insights panel always match
        hovertemplate='<b>%{label}</b><br>$%{value:,.2f}<br>%{percent:.1%}<extra></extra>',
        # Percent labels, but none for categories without cost (plotly would show a floating "0%")
        text=[f"{amount / total:.1%}" if amount > 0 else "" for amount in amounts],
        textinfo="text",
        marker=dict(colors=[COLORS["holding"], COLORS["backlog"], COLORS["order"]])
    )])
    # Same layout as the other charts, with the legend below the donut instead of above the plot. The legend
    # entries are not clickable: the percent labels are fixed and would be wrong after hiding a slice.
    legend_below = dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5,
                        itemclick=False, itemdoubleclick=False)
    fig.update_layout({**CHART_LAYOUT, "legend": legend_below})
    return fig


# Monte Carlo Charts
def histogram_chart(
    values: np.ndarray, bar_color: str, lines: list[ReferenceLine], x_title: str, x_tickformat: str | None = None
) -> go.Figure:
    """Histogram of per-run results with labeled vertical reference lines (e.g. mean and single run).

    Args:
        values: One value per Monte Carlo run.
        bar_color: Fill color of the bars.
        lines: List of (x, label, color, dash) tuples, see add_reference_lines.
        x_title: X axis title.
        x_tickformat: Optional d3 number format of the x axis ticks, e.g. "," for thousands separators.

    Returns:
        The Plotly figure.
    """
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=30, marker_color=bar_color, showlegend=False,
        hovertemplate='%{x}<br>Number of runs: %{y}<extra></extra>'
    ))
    add_reference_lines(fig, lines)
    # The lines are shapes, so a click on a legend entry could not hide them: the entries are not clickable
    legend = {**LEGEND, "itemclick": False, "itemdoubleclick": False}
    fig.update_layout({**CHART_LAYOUT, "legend": legend}, xaxis_title=x_title, yaxis_title="Number of runs")
    fig.update_xaxes(tickformat=x_tickformat)
    return fig


def inventory_band_chart(path: np.ndarray, reorder_point: int) -> go.Figure:
    """Median and 5th-95th percentile band of the net inventory across all Monte Carlo runs.

    Args:
        path: Net inventory per run and week, shape (n_runs, horizon).
        reorder_point: Reorder point s (horizontal line).

    Returns:
        The Plotly figure.
    """
    weeks = np.arange(1, path.shape[1] + 1)
    p5, p50, p95 = np.percentile(path, [5, 50, 95], axis=0)

    # Uncertainty band between the 5th and 95th percentile (the lower trace only anchors the fill)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=p5, mode='lines', line=dict(width=0),
        hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=p95, mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor=with_alpha(COLORS["stock"], 0.15),
        name='5th–95th Percentile', legendrank=2, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=p50, mode='lines',
        name='Median', legendrank=1, line=dict(color=COLORS["stock"], width=2.5),
        customdata=np.column_stack([p5, p95]),
        hovertemplate=(
            bold_label('Median: %{y:.0f}', TEXT_COLORS["stock"])
            + ' (5th–95th: %{customdata[0]:.0f} to %{customdata[1]:.0f})<extra></extra>'
        )
    ))
    reorder_line = reorder_point_trace(weeks, reorder_point)
    reorder_line.legendrank = 3
    fig.add_trace(reorder_line)
    fig.add_hline(y=0, line_color=COLORS["border"], line_width=1)
    # A filled band makes Plotly reverse the legend by default; the legend order is fixed explicitly instead
    legend = {**LEGEND, "traceorder": "normal"}
    fig.update_layout(
        {**CHART_LAYOUT, "legend": legend},
        xaxis_title="Simulation week",
        yaxis_title="Net inventory (units)",
        hovermode="x",
    )
    return fig


# Policy Scan Chart
def policy_heatmap(
    s_values: Sequence[int],
    q_values: Sequence[int],
    grid_cost: np.ndarray,
    grid_fill: np.ndarray,
    best_policy: tuple[int, int],
    selected_policy: tuple[int, int] | None = None,
) -> go.Figure:
    """Heatmap of the mean total cost over a grid of reorder points (rows) and order quantities (columns).

    Args:
        s_values: Reorder points of the grid rows.
        q_values: Order quantities of the grid columns.
        grid_cost: Mean total cost per cell, shape (len(s_values), len(q_values)).
        grid_fill: Mean fill rate (0-1) per cell, same shape; shown in the tooltip.
        best_policy: (s, Q) of the lowest-cost cell, marked with a star.
        selected_policy: (s, Q) selected in the sidebar, marked with a diamond; None if outside the grid.

    Returns:
        The Plotly figure.
    """
    best_s, best_q = best_policy
    fig = go.Figure(go.Heatmap(
        x=q_values, y=s_values, z=grid_cost,
        customdata=grid_fill * 100,
        colorscale=[[0, COLORS["heat_low"]], [1, COLORS["heat_high"]]],
        # Cap the color scale at the 75th percentile: a few extreme corners (e.g. tiny Q with low s)
        # would otherwise compress all other cells into one pale color. Tooltips still show true values.
        zmin=grid_cost.min(), zmax=np.percentile(grid_cost, 75),
        colorbar=dict(title="Mean cost ($)", tickformat=","),
        hovertemplate=(
            "s = %{y}<br>Q = %{x}<br>Mean cost: $%{z:,.2f}<br>"
            "Mean fill rate: %{customdata:.2f}%<extra></extra>"
        )
    ))
    fig.add_trace(go.Scatter(
        x=[best_q], y=[best_s], mode='markers', name='Best Policy',
        marker=dict(symbol='star', size=18, color=COLORS["best_policy"],
                    line=dict(color=COLORS["marker_outline"], width=1.5)),
        hoverinfo='skip'
    ))
    if selected_policy is not None:
        selected_s, selected_q = selected_policy
        fig.add_trace(go.Scatter(
            x=[selected_q], y=[selected_s], mode='markers', name='Selected Policy',
            marker=dict(symbol='diamond', size=12, color=COLORS["selected_policy"],
                        line=dict(color=COLORS["marker_outline"], width=1.5)),
            hoverinfo='skip'
        ))
    fig.update_layout(**CHART_LAYOUT, xaxis_title="Order quantity (Q)", yaxis_title="Reorder point (s)")
    return fig
