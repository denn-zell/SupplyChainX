"""Shared constants for SupplyChainX: defaults, analysis settings, colors, chart layout, stylesheet, and texts."""

from simulation import Scenario

# Defaults:
# Single source of truth for the sidebar presets and for the baseline run behind the "vs Default" deltas.
DEFAULT_SCENARIO = Scenario(
    reorder_point=50,
    order_quantity=100,
    mean_demand=20.0,
    std_demand=5.0,
    lead_time=2,
    holding_cost=1.0,
    backorder_cost=10.0,
    order_cost=50.0,
    initial_inventory=150,
)
DEFAULT_WEEKS = 52
DEFAULT_SEED = 42

# Analysis Settings
GRID_POINTS = 15           # Grid points per axis in the policy scan
CACHE_MAX_ENTRIES = 50     # Results kept per cached batch analysis (limits memory use in long sessions)
MAX_INPUT = 1_000_000      # Upper limit of the numeric sidebar inputs (keeps all sums far from integer overflow)
CLIPPING_WARNING_Z = 2.33  # Warn about clipped demand if mean / std is below this (more than ~1% negative draws)

# About
AUTHOR = "Dennis Zeller"
REPO_URL = "https://github.com/denn-zell/SupplyChainX"  # shown in the footer

# Color Palette
# Colors for charts, tooltips, and labels. The stylesheet below keeps its own neutral UI colors.
COLORS = {
    "stock": "#2563EB",            # positive net inventory
    "backlog": "#EF4444",          # negative net inventory, backorder cost
    "order": "#10B981",            # order markers, ordering cost
    "holding": "#3B82F6",          # holding cost
    "position": "#64748B",         # inventory position line
    "reorder": "#D97706",          # reorder point line
    "border": "#E2E8F0",           # tooltip border, zero line in the band chart
    "spike": "#CBD5E1",            # hover spike line
    "heat_low": "#DBEAFE",         # policy scan color scale: low cost
    "heat_high": "#1E3A8A",        # policy scan color scale: high cost
    "best_policy": "#FBBF24",      # policy scan: best policy marker (gold)
    "selected_policy": "#FFFFFF",  # policy scan: selected policy marker
    "marker_outline": "#000000",   # policy scan: outline of the best and the selected policy marker
    "mean": "#DC2626",             # mean line and label in the Monte Carlo histograms
}

# Darker shades of the series colors for text (labels and tooltips). The series colors above are too light
# for small text on a white background.
TEXT_COLORS = {
    "stock": "#2563EB",
    "backlog": "#DC2626",
    "order": "#047857",
    "holding": "#1D4ED8",
    "position": "#64748B",
    "reorder": "#B45309",
}

# Font of all charts; matches the font of the page text (see the stylesheet below)
FONT_FAMILY = "Inter, sans-serif"

# Legend above the plot, aligned to the left (the toolbar of the chart sits on the right)
LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)

# Shared Plotly layout for all charts
CHART_LAYOUT = dict(
    font=dict(family=FONT_FAMILY),
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=20, b=20),
    legend=LEGEND,
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=COLORS["border"],
        font_size=12,
        font_family=FONT_FAMILY,
        namelength=-1
    ),
)

# Custom CSS (page, sidebar, cards, tabs):
# Streamlit changes its internal markup between versions. The selectors below therefore rely on
# stable attributes and cover both the current and the older tab markup.
CUSTOM_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Page background: light gray, so that white cards and the white sidebar stand out.
       The header and the containers around the main area are transparent to show it. */
    body .stApp {
        background-color: #F8FAFC;
    }
    [data-testid="stHeader"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent;
    }

    /* Inter for the main text elements. Streamlit sets the font on many components itself, so the
       elements are listed explicitly instead of using a wildcard, which would also change icon fonts. */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMetricValue"],
    [role="tab"],
    input {
        font-family: 'Inter', sans-serif;
    }

    /* Main content area spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Readable line length: on wide screens, text lines of full width are hard to read */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        max-width: 90ch;
    }

    /* Sidebar surface */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Hide the browser's number input spinners */
    input[type="number"]::-webkit-outer-spin-button,
    input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type="number"] {
        -moz-appearance: textfield;
    }

    /* KPI cards: uniform minimum height so cards align in a row */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Charts as white cards, matching the KPI cards. The card is the container around the chart, so the
       chart itself keeps the size that Streamlit measures for it. */
    [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px;
        box-sizing: border-box;
    }

    /* KPI card label and value typography */
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-weight: 500;
        font-size: 0.875rem;
    }
    /* Streamlit cuts long KPI labels and change values ("+$250.00 vs Default") off with "...". On narrow
       screens they wrap instead; the minimum height of the cards keeps all cards in a row equally tall. */
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] [data-testid="stMarkdownContainer"] {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* Tabs styled as a segmented control: light gray bar, the active tab as a white pill */
    .stTabs [role="tablist"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
        flex-wrap: wrap;
        overflow: visible;
    }
    /* Remove the default baseline under the tab bar and the colored underline of the active tab */
    .stTabs [role="tablist"]::after,
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"],
    .stTabs .react-aria-SelectionIndicator {
        display: none !important;
    }
    .stTabs [role="tab"] {
        height: auto;
        background-color: transparent;
        border-radius: 8px;
        color: #475569;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [role="tab"]:hover {
        color: #0F172A;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
    }
    </style>
"""

# Model Explanation:
# Shown in the expander "How this model works" (markdown)
MODEL_EXPLANATION = """
**The model.** A single item is managed with a periodic-review (s, Q) policy. Each week:

1. Deliveries that are due arrive.
2. Random demand occurs and is served from stock on hand. Demand that cannot be served is backordered
   and delivered later from incoming stock.
3. If the inventory position is at or below the reorder point s, one order of Q units is placed. It arrives
   after the lead time. At most one order is placed per week.
4. Costs are charged on the end-of-week state: holding cost per unit on hand, backorder penalty per unit
   in backlog, and a fixed cost per order.

*Net inventory* is stock on hand minus backlog. The *inventory position* is net inventory plus the quantity
that is on order but has not arrived yet.

**Demand** is normally distributed (mean μ, standard deviation σ), rounded to whole units, and negative values
are set to 0. If σ is large relative to μ, this raises the effective mean demand, and the sidebar shows a warning.

**Fill rate** is the share of the total demand that was served immediately from stock in the week it occurred.
It is not the cycle service level.

**Comparison with the default.** The changes shown below the key figures compare your scenario with the default
scenario (the sidebar values that the app starts with) on the same random draws.

**Monte Carlo.** Everything except the Monte Carlo and Policy Optimization tabs shows a single simulation run,
which is only one possible outcome. The Monte Carlo tab repeats the simulation with many independent demand
histories. Run 1 is identical to the single run, and the same seed always reproduces the same results.

**Policy optimization.** The policy scan simulates a grid of reorder points and order quantities. All policies
see the same demand histories, and the star marks the lowest mean cost. Only cost is minimized, so the best
policy can have a lower fill rate than a more expensive one. Near the optimum the cost surface is flat, so
neighboring policies often differ only slightly.
"""
