"""
Nassau Candy Distributor
Streamlit Profitability Dashboard — app.py

Run: streamlit run app.py
"""

from pathlib import Path
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

from data_processing import load_and_clean_data

warnings.filterwarnings("ignore")


def configure_page() -> None:
    """Configure the Streamlit page layout and metadata."""
    st.set_page_config(
        page_title="Nassau Candy Dashboard",
        page_icon="🍫",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    """Load custom CSS to style the dashboard."""
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --primary: #EC4899;      
        --secondary: #A855F7;    
        --accent: #14B8A6;       
        --success: #10B981;
        --warning: #FBBF24;
        --bg-main: #FFF5F7;      
        --bg-card: #FFFFFF;
        --text-dark: #334155;
        --glass: rgba(255, 255, 255, 0.85);
        --glass-border: rgba(236, 72, 153, 0.15); 
        --shadow: 0 10px 30px -10px rgba(236, 72, 153, 0.15); 
        --candy-radius: 20px;
    }

    .stApp {
        background: var(--bg-main);
        color: var(--text-dark);
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', 'Manrope', serif;
        letter-spacing: 0.2px;
        color: var(--text-dark);
    }

    h1 {
        font-size: 2.4rem;
        margin-bottom: 0.1rem;
    }
    h2 { font-size: 1.7rem; }
    h3 { font-size: 1.2rem; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out forwards;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .glass-panel {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: var(--candy-radius);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
    }

    .glass-panel:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 15px 35px -5px rgba(236, 72, 153, 0.25); /* Stronger pink candy glow */
    }

    .hero {
        padding: 22px 26px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
    }

    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.1rem;
        margin: 0 0 6px 0;
        color: var(--primary);
    }

    .hero-subtitle {
        margin: 0;
        font-size: 1rem;
        color: var(--text-dark);
        opacity: 0.8;
    }

    .hero-kpi {
        display: grid;
        grid-template-columns: repeat(2, minmax(140px, 1fr));
        gap: 10px;
    }

    .hero-kpi .kpi-box {
        background: var(--bg-card);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 10px 12px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .hero-kpi .kpi-box:hover {
        transform: translateY(-3px);
    }
    .hero-kpi .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-dark);
        opacity: 0.7;
    }
    .hero-kpi .kpi-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--primary);
    }

    div[data-testid="stMetric"] {
        background: var(--glass);
        border-radius: var(--candy-radius);
        padding: 14px;
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 15px 25px -5px rgba(236, 72, 153, 0.2);
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--glass);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        color: var(--text-dark);
        padding: 8px 16px;
        margin-right: 8px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #FDF2F8; /* Soft pink hover */
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        color: var(--bg-card);
        border-color: var(--primary);
        box-shadow: 0 4px 10px rgba(236, 72, 153, 0.3);
    }

    .stDataFrame {
        background: var(--glass);
        border-radius: var(--candy-radius);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow);
        padding: 6px;
    }

    /* Streamlit Sidebar Tags and Inputs override */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: var(--accent) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }

    .section-header { font-size: 1.1rem; font-weight: 600; margin: 10px 0 5px; color: var(--text-dark); }

    /* Override markdown text colors */
    .stMarkdown p, .stMarkdown li, .stMarkdown label, .stText, .stInfo, .stSuccess, .stWarning, .stError {
        color: var(--text-dark) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the sales dataset utilizing the shared processing utility."""
    try:
        return load_and_clean_data()
    except FileNotFoundError:
        st.error("❌ Could not find the dataset CSV file. Please ensure 'Nassau Candy Distributor.csv' is in the project folder.")
        st.stop()


@st.cache_data
def get_product_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dataset at product level and add clustering labels."""
    product_summary = data.groupby("Product Name").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Total_Units=("Units", "sum"),
        Total_Cost=("Cost", "sum"),
        Avg_Margin_Pct=("Gross_Margin_Pct", "mean"),
        Avg_PPU=("Profit_Per_Unit", "mean"),
        Orders=("Sales", "count"),
    ).reset_index()

    product_summary["Profit_Contribution_Pct"] = (
        product_summary["Total_Profit"] / product_summary["Total_Profit"].sum() * 100
    )
    product_summary["Revenue_Contribution_Pct"] = (
        product_summary["Total_Sales"] / product_summary["Total_Sales"].sum() * 100
    )

    if product_summary.empty:
        # Keep expected columns even when filters return no rows.
        product_summary["Cluster"] = []
        product_summary["Cluster_Label"] = []
        return product_summary

    # Cluster on a small set of profitability signals.
    features = ["Total_Profit", "Avg_Margin_Pct", "Avg_PPU", "Total_Units"]
    scaled_features = StandardScaler().fit_transform(product_summary[features])
    n_samples = len(product_summary)
    n_clusters = min(3, n_samples)

    if n_clusters == 1:
        # With a single product, clustering would be unstable; assign one cluster.
        product_summary["Cluster"] = 0
    else:
        product_summary["Cluster"] = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        ).fit_predict(scaled_features)

    # Calculate average profit for each cluster to identify them consistently
    cluster_stats = product_summary.groupby("Cluster")["Total_Profit"].mean().sort_values(ascending=False)
    
    # Map clusters based on their average profit ranking
    label_map = {}
    if len(cluster_stats) >= 1:
        label_map[cluster_stats.index[0]] = "⭐ Star"
    if len(cluster_stats) >= 2:
        label_map[cluster_stats.index[1]] = "⚠ Average"
    if len(cluster_stats) >= 3:
        label_map[cluster_stats.index[2]] = "❌ Dead Weight"
    
    product_summary["Cluster_Label"] = product_summary["Cluster"].map(label_map)

    return product_summary.sort_values("Total_Profit", ascending=False).reset_index(drop=True)


def build_sidebar_filters(data: pd.DataFrame) -> tuple[list[str], list, int, str]:
    """Render sidebar inputs and return the selected filter values."""
    st.sidebar.title("🍫 Nassau Candy")
    st.sidebar.markdown("---")

    divisions = st.sidebar.multiselect(
        "🏢 Division:",
        options=data["Division"].unique().tolist(),
        default=data["Division"].unique().tolist(),
    )

    min_date, max_date = data["Order Date"].min(), data["Order Date"].max()
    date_range = st.sidebar.date_input(
        "📅 Date Range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date,
    )

    margin_threshold = st.sidebar.slider("⚠ Margin Risk Threshold (%):", 0, 100, 40)
    search = st.sidebar.text_input("🔎 Product Search:", "")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cluster Legend:**")
    st.sidebar.markdown("🟢  Star — High profit & margin")
    st.sidebar.markdown("🟡  Average — Medium performance")
    st.sidebar.markdown("🔴  Dead Weight — Low profit & margin")

    return divisions, date_range, margin_threshold, search


def apply_filters(
    data: pd.DataFrame,
    divisions: list[str],
    date_range: list,
    search_term: str,
) -> pd.DataFrame:
    """Apply sidebar filters to the dataset."""
    # Start with division filter, then narrow by date and search keyword.
    filtered = data[data["Division"].isin(divisions)].copy()

    if len(date_range) == 2:
        filtered = filtered[
            (filtered["Order Date"] >= pd.to_datetime(date_range[0]))
            & (filtered["Order Date"] <= pd.to_datetime(date_range[1]))
        ]

    if search_term:
        filtered = filtered[filtered["Product Name"].str.contains(search_term, case=False)]

    return filtered


def render_header(
    filtered: pd.DataFrame,
    product_summary: pd.DataFrame,
    divisions: list[str],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> None:
    """Render the hero header block and summary caption."""
    # Hero block is HTML/CSS so it can host KPI tiles in a single panel.
    st.markdown(
        f"""
        <div class="glass-panel hero">
            <div>
                <div class="hero-title">Nassau Candy Profitability Dashboard</div>
                <p class="hero-subtitle">Executive overview of revenue, margin health, and product performance.</p>
            </div>
            <div class="hero-kpi">
                <div class="kpi-box">
                    <div class="kpi-label">Orders</div>
                    <div class="kpi-value">{len(filtered):,}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Products</div>
                    <div class="kpi-value">{product_summary['Product Name'].nunique():,}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Divisions</div>
                    <div class="kpi-value">{len(divisions):,}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Window</div>
                    <div class="kpi-value">{min_date:%b %Y}–{max_date:%b %Y}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Showing {len(filtered):,} orders | "
        f"{product_summary['Product Name'].nunique()} products | "
        f"{', '.join(divisions)}"
    )
    st.markdown("---")


def render_kpis(filtered: pd.DataFrame) -> None:
    """Render the KPI cards at the top of the dashboard."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Total Revenue", f"${filtered['Sales'].sum():,.0f}")
    k2.metric("📈 Total Profit", f"${filtered['Gross Profit'].sum():,.0f}")
    k3.metric("📊 Avg Margin", f"{filtered['Gross_Margin_Pct'].mean():.1f}%")
    k4.metric("📦 Total Units", f"{filtered['Units'].sum():,.0f}")
    k5.metric("🛒 Total Orders", f"{len(filtered):,}")
    st.markdown("---")


def render_tab_product_profitability(
    product_summary: pd.DataFrame,
    margin_threshold: int,
) -> None:
    """Render the product profitability tab content."""
    st.subheader("📊 Product Profitability Overview")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            product_summary,
            x="Total_Profit",
            y="Product Name",
            orientation="h",
            color="Avg_Margin_Pct",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            title="Total Profit by Product",
            labels={"Total_Profit": "Profit ($)", "Avg_Margin_Pct": "Margin %"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            product_summary.sort_values("Avg_Margin_Pct"),
            x="Avg_Margin_Pct",
            y="Product Name",
            orientation="h",
            color="Avg_Margin_Pct",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            title="Avg Gross Margin % by Product",
            labels={"Avg_Margin_Pct": "Margin %"},
        )
        fig2.add_vline(
            x=margin_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold {margin_threshold}%",
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # Profit contribution donut
    fig3 = px.pie(
        product_summary,
        values="Profit_Contribution_Pct",
        names="Product Name",
        title="Profit Contribution % by Product",
        hole=0.4,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Leaderboard
    st.subheader("🏆 Product Leaderboard")
    display_table = product_summary[
        [
            "Product Name",
            "Cluster_Label",
            "Total_Sales",
            "Total_Profit",
            "Avg_Margin_Pct",
            "Avg_PPU",
            "Profit_Contribution_Pct",
        ]
    ].copy()
    display_table.columns = [
        "Product",
        "Segment",
        "Revenue",
        "Profit",
        "Margin %",
        "Profit/Unit",
        "Contribution %",
    ]
    display_table["Revenue"] = display_table["Revenue"].map("${:,.0f}".format)
    display_table["Profit"] = display_table["Profit"].map("${:,.0f}".format)
    display_table["Margin %"] = display_table["Margin %"].map("{:.1f}%".format)
    display_table["Profit/Unit"] = display_table["Profit/Unit"].map("${:.2f}".format)
    display_table["Contribution %"] = display_table["Contribution %"].map("{:.1f}%".format)
    st.dataframe(display_table.reset_index(drop=True), use_container_width=True)

    # Risk flags
    risk = product_summary[product_summary["Avg_Margin_Pct"] < margin_threshold]
    if len(risk) > 0:
        st.warning(
            f"⚠ {len(risk)} products below {margin_threshold}% margin — review needed!"
        )
        st.dataframe(
            risk[
                ["Product Name", "Avg_Margin_Pct", "Total_Profit", "Cluster_Label"]
            ]
            .rename(
                columns={
                    "Avg_Margin_Pct": "Margin %",
                    "Total_Profit": "Profit",
                    "Cluster_Label": "Segment",
                }
            )
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        st.success(f"✅ All products are above the {margin_threshold}% margin threshold.")


def render_tab_division_performance(
    filtered: pd.DataFrame,
    margin_threshold: int,
) -> None:
    """Render the division performance tab content."""
    st.subheader("🏢 Division Performance Dashboard")

    division_summary = filtered.groupby("Division").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Avg_Margin=("Gross_Margin_Pct", "mean"),
        Total_Units=("Units", "sum"),
        Products=("Product Name", "nunique"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Revenue",
                x=division_summary["Division"],
                y=division_summary["Total_Sales"],
                marker_color="#06B6D4",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Profit",
                x=division_summary["Division"],
                y=division_summary["Total_Profit"],
                marker_color="#7C3AED",
            )
        )
        fig.update_layout(barmode="group", title="Revenue vs Profit by Division", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            division_summary,
            x="Division",
            y="Avg_Margin",
            color="Avg_Margin",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            title="Avg Margin % by Division",
        )
        fig2.add_hline(
            y=margin_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold {margin_threshold}%",
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Box plot — margin distribution
    fig3 = px.box(
        filtered,
        x="Division",
        y="Gross_Margin_Pct",
        color="Division",
        title="Margin Distribution by Division",
        labels={"Gross_Margin_Pct": "Gross Margin %"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Monthly trend by division
    monthly_div = (
        filtered.groupby(["Month", "Division"])["Gross Profit"].sum().reset_index()
    )
    fig4 = px.line(
        monthly_div,
        x="Month",
        y="Gross Profit",
        color="Division",
        title="Monthly Profit Trend by Division",
        labels={"Gross Profit": "Profit ($)"},
    )
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)


def render_tab_cost_vs_margin(
    filtered: pd.DataFrame,
    product_summary: pd.DataFrame,
) -> None:
    """Render the cost vs margin diagnostics tab."""
    st.subheader("💸 Cost vs Margin Diagnostics")

    fig = px.scatter(
        filtered,
        x="Cost",
        y="Sales",
        color="Gross_Margin_Pct",
        color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
        hover_data=["Product Name", "Division"],
        title="Cost vs Sales (colour = Margin %)",
        labels={"Gross_Margin_Pct": "Margin %"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        product_costs = filtered.groupby("Product Name").agg(
            Avg_Cost=("Cost", "mean"),
            Avg_Sales=("Sales", "mean"),
            Avg_Margin=("Gross_Margin_Pct", "mean"),
        ).reset_index()
        fig2 = px.scatter(
            product_costs,
            x="Avg_Cost",
            y="Avg_Sales",
            size="Avg_Margin",
            color="Avg_Margin",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            hover_data=["Product Name"],
            title="Avg Cost vs Avg Sales per Product (size = Margin)",
            labels={"Avg_Margin": "Margin %"},
        )
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        fig3 = px.bar(
            product_summary.sort_values("Avg_PPU", ascending=False),
            x="Product Name",
            y="Avg_PPU",
            color="Avg_PPU",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            title="Profit per Unit by Product",
            labels={"Avg_PPU": "Profit/Unit ($)"},
        )
        fig3.update_layout(xaxis_tickangle=-40, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    # Factory-level cost analysis
    if "Factory" in filtered.columns:
        st.subheader("🏭 Factory Cost Analysis")
        factory_sum = filtered.groupby("Factory").agg(
            Avg_Cost=("Cost", "mean"),
            Total_Cost=("Cost", "sum"),
            Avg_Margin=("Gross_Margin_Pct", "mean"),
        ).reset_index()
        fig4 = px.bar(
            factory_sum,
            x="Factory",
            y="Avg_Margin",
            color="Avg_Margin",
            color_continuous_scale=["#EC4899", "#F59E0B", "#10B981"],
            title="Avg Margin % by Factory",
        )
        st.plotly_chart(fig4, use_container_width=True)


def render_tab_pareto(product_summary: pd.DataFrame) -> None:
    """Render the Pareto analysis tab."""
    st.subheader("📉 Profit Concentration — Pareto Analysis")

    pareto_df = product_summary.sort_values("Total_Profit", ascending=False).copy()
    pareto_df["Cumulative_Profit"] = pareto_df["Total_Profit"].cumsum()
    pareto_df["Cumulative_Pct"] = (
        pareto_df["Cumulative_Profit"] / pareto_df["Total_Profit"].sum() * 100
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pareto_df["Product Name"],
            y=pareto_df["Total_Profit"],
            name="Profit",
            marker_color="#06B6D4",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_df["Product Name"],
            y=pareto_df["Cumulative_Pct"],
            name="Cumulative %",
            yaxis="y2",
            line=dict(color="red", width=2),
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="Pareto Chart — Profit Concentration",
        yaxis=dict(title="Total Profit ($)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]),
        shapes=[
            dict(
                type="line",
                x0=-0.5,
                x1=len(pareto_df) - 0.5,
                y0=80,
                y1=80,
                yref="y2",
                line=dict(color="orange", width=2, dash="dash"),
            )
        ],
        annotations=[
            dict(
                x=len(pareto_df) - 1,
                y=80,
                yref="y2",
                text="80% line",
                showarrow=False,
                font=dict(color="orange"),
            )
        ],
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    pareto_80 = pareto_df[pareto_df["Cumulative_Pct"] <= 80]
    pct_prod = len(pareto_80) / len(pareto_df) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("80% Profit Level", f"{len(pareto_80)} products")
    m2.metric("Total Products", f"{len(pareto_df)}")
    m3.metric("% of Portfolio", f"{pct_prod:.0f}%")

    st.info(
        f"📌 Only **{len(pareto_80)} products** ({pct_prod:.0f}% of the portfolio) drive "
        f"**80% of total profit**. The remaining {len(pareto_df)-len(pareto_80)} "
        "products need close review."
    )

    st.subheader("Top Contributors")
    show_p = pareto_80[
        ["Product Name", "Total_Profit", "Profit_Contribution_Pct", "Cumulative_Pct"]
    ].copy()
    show_p.columns = ["Product", "Profit", "Contribution %", "Cumulative %"]
    show_p["Profit"] = show_p["Profit"].map("${:,.0f}".format)
    show_p["Contribution %"] = show_p["Contribution %"].map("{:.1f}%".format)
    show_p["Cumulative %"] = show_p["Cumulative %"].map("{:.1f}%".format)
    st.dataframe(show_p.reset_index(drop=True), use_container_width=True)


def render_tab_clustering(
    product_summary: pd.DataFrame,
    margin_threshold: int,
) -> None:
    """Render the ML clustering tab content."""
    st.subheader("🤖 ML Product Segmentation — K-Means Clustering")
    st.info(
        "K-Means automatically segments products into three groups based on profit, margin, "
        "and profit per unit."
    )

    color_map = {
        " Star": "#10B981",       # Success (Mint Green)
        "Average": "#F59E0B",    # Warning (Orange)
        " Dead Weight": "#EC4899", # Secondary (Pink) instead of Red
    }
    fig = px.scatter(
        product_summary,
        x="Total_Profit",

        y="Avg_Margin_Pct",
        size="Total_Units",
        color="Cluster_Label",
        color_discrete_map=color_map,
        hover_data=["Product Name", "Avg_PPU"],
        title="Product Clusters: Profit vs Margin",
        labels={
            "Total_Profit": "Total Profit ($)",
            "Avg_Margin_Pct": "Avg Margin %",
            "Cluster_Label": "Segment",
        },
        text="Product Name",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.add_hline(
        y=margin_threshold,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Margin threshold {margin_threshold}%",
    )
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    # Cluster breakdown
    st.subheader("📋 Segment Breakdown")
    
    # Sort order for segments
    segment_order = ["⭐ Star", "⚠ Average", "❌ Dead Weight"]
    
    for label in segment_order:
        # Filter for this specific segment
        group = product_summary[product_summary["Cluster_Label"] == label].copy()
        
        # Define color based on label
        if label == "⭐ Star":
            color_func = st.success
            desc = "High profit & high margin"
        elif label == "⚠ Average":
            color_func = st.warning
            desc = "Medium performance"
        else:
            color_func = st.error
            desc = "Low performance"

        # Show the header with product count
        color_func(f"{label} — {len(group)} products ({desc})")

        if not group.empty:
            # Prepare display table
            display_group = group[["Product Name", "Total_Profit", "Avg_Margin_Pct", "Avg_PPU"]].copy()
            display_group.columns = ["Product", "Profit", "Margin %", "Profit/Unit"]
            
            # Format values
            display_group["Profit"] = display_group["Profit"].map("${:,.0f}".format)
            display_group["Margin %"] = display_group["Margin %"].map("{:.1f}%".format)
            display_group["Profit/Unit"] = display_group["Profit/Unit"].map("${:.2f}".format)
            
            st.dataframe(display_group.reset_index(drop=True), use_container_width=True)
        else:
            st.info(f"No products found in the {label} segment for current filters.")

    # Recommendations
    st.subheader("📌 Recommendations")
    st.markdown("""
| Action | Products |
|--------|----------|
| ✅ **Invest & Promote** | ⭐ Star segment products |
| ⚠ **Reprice / Renegotiate Cost** | ⚠ Average + low-margin products |
| ❌ **Review for Discontinuation** | ❌ Dead Weight products |
""")


def render_tab_logistics_efficiency(filtered: pd.DataFrame) -> None:
    """Render the logistics efficiency analysis tab."""
    st.subheader("🚚 Logistics & Route Efficiency")
    st.info("Analyzing lead times and route-level operational intelligence for nationwide delivery reliability.")

    # Calculate metrics by Route (Region-Ship Mode)
    route_efficiency = filtered.groupby(["Region", "Ship Mode"]).agg(
        Avg_Lead_Time=("Lead_Time", "mean"),
        Order_Count=("Order ID", "count"),
        Total_Profit=("Gross Profit", "sum"),
        Avg_Profit_Per_Order=("Gross Profit", "mean")
    ).reset_index()
    
    route_efficiency["Route"] = route_efficiency["Region"] + " - " + route_efficiency["Ship Mode"]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            route_efficiency.sort_values("Avg_Lead_Time"),
            x="Avg_Lead_Time",
            y="Route",
            orientation="h",
            color="Avg_Lead_Time",
            color_continuous_scale="Viridis",
            title="Avg Lead Time by Route (Days)",
            labels={"Avg_Lead_Time": "Avg Lead Time (Days)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            route_efficiency,
            x="Avg_Lead_Time",
            y="Avg_Profit_Per_Order",
            size="Order_Count",
            color="Region",
            title="Lead Time vs Profitability per Route",
            labels={
                "Avg_Lead_Time": "Avg Lead Time (Days)",
                "Avg_Profit_Per_Order": "Profit per Order ($)"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Route Performance Data")
    display_routes = route_efficiency.sort_values("Avg_Lead_Time", ascending=False).copy()
    display_routes["Avg_Lead_Time"] = display_routes["Avg_Lead_Time"].map("{:.1f} days".format)
    display_routes["Total_Profit"] = display_routes["Total_Profit"].map("${:,.0f}".format)
    display_routes["Avg_Profit_Per_Order"] = display_routes["Avg_Profit_Per_Order"].map("${:.2f}".format)
    st.dataframe(display_routes.drop(columns=["Route"]).reset_index(drop=True), use_container_width=True)


def main() -> None:
    """"Run the Streamlit dashboard."""
    configure_page()
    inject_css()

    # Data pipeline: load → filter → summarize → render tabs.
    data = load_data()

    divisions, date_range, margin_threshold, search_term = build_sidebar_filters(data)
    
    filtered = apply_filters(data, divisions, date_range, search_term)
    product_summary = get_product_summary(filtered)

    min_date, max_date = data["Order Date"].min(), data["Order Date"].max()
    render_header(filtered, product_summary, divisions, min_date, max_date)
    render_kpis(filtered)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Product Profitability",
            "🏢 Division Performance",
            "💸 Cost vs Margin",
            "📉 Pareto Analysis",
            "🤖 ML Clustering",
            "🚚 Logistics Efficiency"
        ]
    )

    with tab1:
        render_tab_product_profitability(product_summary, margin_threshold)

    with tab2:
        render_tab_division_performance(filtered, margin_threshold)

    with tab3:
        render_tab_cost_vs_margin(filtered, product_summary)

    with tab4:
        render_tab_pareto(product_summary)

    with tab5:
        render_tab_clustering(product_summary, margin_threshold)
        
    with tab6:
        render_tab_logistics_efficiency(filtered)

    st.markdown("---")
    st.caption("🍫 Nassau Candy Distributor | ML Internship Project | Built with Streamlit + Plotly")


if __name__ == "__main__":
    main()
