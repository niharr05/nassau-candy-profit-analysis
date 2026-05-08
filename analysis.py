"""
Nassau Candy Distributor
Product Line Profitability & Margin Performance Analysis
ML Internship Project — analysis.py

Run: python analysis.py
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from data_processing import load_and_clean_data

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def print_banner() -> None:
    """Print the report header banner."""
    print("=" * 60)
    print("  Nassau Candy — Profitability Analysis")
    print("=" * 60)


def load_and_preprocess() -> pd.DataFrame:
    """Load, clean, and enrich the data using the shared utility."""
    print("\n📂 STEP 1-3: Data Loading & Preprocessing...")
    
    # Use the shared data_processing logic
    df = load_and_clean_data("Nassau Candy Distributor.csv")
    
    print(f"  ✅ Complete! Clean rows: {len(df)}")
    print(f"  Divisions  : {df['Division'].unique().tolist()}")
    print(f"  Products   : {df['Product Name'].nunique()} unique")
    print(f"  Date range : {df['Order Date'].min().date()} → {df['Order Date'].max().date()}")

    total_sales = df["Sales"].sum()
    total_profit = df["Gross Profit"].sum()
    avg_margin = df["Gross_Margin_Pct"].mean()

    print("\n  ┌─────────────────────────────────┐")
    print(f"  │  Total Revenue : ${total_sales:>12,.2f}  │")
    print(f"  │  Total Profit  : ${total_profit:>12,.2f}  │")
    print(f"  │  Avg Margin    : {avg_margin:>11.1f}%  │")
    print("  └─────────────────────────────────┘")

    # Map Month back to just Month_Period for the existing analysis charts if needed
    df["Month"] = df["Month_Period"]
    return df


def summarize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate product-level metrics and print the rankings table."""
    print("\n🏆 STEP 4: Product-Level Profitability Analysis...")

    product_summary = df.groupby("Product Name").agg(
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
    product_summary = product_summary.sort_values("Total_Profit", ascending=False).reset_index(drop=True)

    print("\n  📋 Product Rankings (Best → Worst Profit):")
    print(f"  {'Product':<35} {'Sales':>10} {'Profit':>10} {'Margin':>8} {'Contribution':>13}")
    print("  " + "-" * 80)
    for _, row in product_summary.iterrows():
        print(
            f"  {row['Product Name']:<35} ${row['Total_Sales']:>9,.0f} "
            f"${row['Total_Profit']:>9,.0f} {row['Avg_Margin_Pct']:>7.1f}% "
            f"{row['Profit_Contribution_Pct']:>11.1f}%"
        )

    return product_summary


def summarize_divisions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate division-level metrics and print the summary table."""
    print("\n🏢 STEP 5: Division Performance Analysis...")

    division_summary = df.groupby("Division").agg(
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Total_Cost=("Cost", "sum"),
        Avg_Margin=("Gross_Margin_Pct", "mean"),
        Total_Units=("Units", "sum"),
        Products=("Product Name", "nunique"),
    ).reset_index()
    division_summary["Profit_Margin"] = (
        division_summary["Total_Profit"] / division_summary["Total_Sales"] * 100
    )
    division_summary = division_summary.sort_values("Avg_Margin", ascending=False)

    print(f"\n  {'Division':<12} {'Revenue':>12} {'Profit':>12} {'Avg Margin':>12} {'Products':>10}")
    print("  " + "-" * 60)
    for _, row in division_summary.iterrows():
        print(
            f"  {row['Division']:<12} ${row['Total_Sales']:>11,.0f} "
            f"${row['Total_Profit']:>11,.0f} {row['Avg_Margin']:>11.1f}% "
            f"{int(row['Products']):>10}"
        )

    return division_summary


def run_pareto_analysis(product_summary: pd.DataFrame) -> pd.DataFrame:
    """Run the 80/20 Pareto analysis and print the top contributors."""
    print("\n📉 STEP 6: Pareto (80/20) Analysis...")

    pareto_df = product_summary.sort_values("Total_Profit", ascending=False).copy()
    pareto_df["Cumulative_Pct"] = (
        pareto_df["Total_Profit"].cumsum() / pareto_df["Total_Profit"].sum() * 100
    )
    pareto_80 = pareto_df[pareto_df["Cumulative_Pct"] <= 80]
    pct_products = len(pareto_80) / len(pareto_df) * 100

    print(
        f"\n  ✅ {len(pareto_80)} products ({pct_products:.0f}% of portfolio) → 80% of total profit"
    )
    print(
        f"  ⚠  Remaining {len(pareto_df)-len(pareto_80)} products contribute only 20% profit"
    )
    print("\n  Top Contributors:")
    for _, row in pareto_80.iterrows():
        print(
            f"    → {row['Product Name']:<35} {row['Profit_Contribution_Pct']:.1f}% | "
            f"Cumulative: {row['Cumulative_Pct']:.1f}%"
        )

    return pareto_df


def run_clustering(product_summary: pd.DataFrame) -> pd.DataFrame:
    """Apply K-Means clustering and print the segment breakdown."""
    print("\n🤖 STEP 7: ML Clustering (K-Means)...")

    features = ["Total_Profit", "Avg_Margin_Pct", "Avg_PPU", "Total_Units"]
    X = product_summary[features].copy()
    scaler = StandardScaler()
    # Standardize scales so profit and unit counts contribute evenly to distance.
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    product_summary["Cluster"] = kmeans.fit_predict(X_scaled)

    # Label clusters by average profit so the names are stable across runs.
    cluster_means = product_summary.groupby("Cluster")["Total_Profit"].mean().sort_values(ascending=False)
    label_map = {
        cluster_means.index[0]: "⭐ Star",
        cluster_means.index[1]: "⚠ Average",
        cluster_means.index[2]: "❌ Dead Weight",
    }
    product_summary["Cluster_Label"] = product_summary["Cluster"].map(label_map)

    print("\n  Product Clusters:")
    for label in ["⭐ Star", "⚠ Average", "❌ Dead Weight"]:
        group = product_summary[product_summary["Cluster_Label"] == label]
        print(f"\n  {label}:")
        for _, row in group.iterrows():
            print(
                f"    • {row['Product Name']:<35} Profit: ${row['Total_Profit']:>8,.0f} | "
                f"Margin: {row['Avg_Margin_Pct']:.1f}%"
            )

    # Margin risk flag
    risk_products = product_summary[product_summary["Avg_Margin_Pct"] < 40]
    print(f"\n  ⚠  Margin Risk (<40%): {len(risk_products)} products")
    for _, row in risk_products.iterrows():
        print(f"    → {row['Product Name']}: {row['Avg_Margin_Pct']:.1f}%")

    return product_summary


def build_charts(product_summary: pd.DataFrame, division_summary: pd.DataFrame) -> None:
    """Generate and save the combined figure plus each chart as a separate image."""
    print("\n📈 STEP 8: Generating Charts...")

    def plot_total_profit_by_product(ax: plt.Axes) -> None:
        colors = [
            "#10B981" if m >= 50 else "#F59E0B" if m >= 35 else "#EC4899"
            for m in product_summary["Avg_Margin_Pct"]
        ]
        bars = ax.barh(product_summary["Product Name"], product_summary["Total_Profit"], color=colors)
        ax.set_xlabel("Total Profit ($)")
        ax.set_title("📊 Total Profit by Product", fontweight="bold")
        ax.invert_yaxis()
        for bar, val in zip(bars, product_summary["Total_Profit"]):
            ax.text(
                bar.get_width() + 100,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}",
                va="center",
                fontsize=8,
            )

    def plot_margin_by_product(ax: plt.Axes) -> None:
        sorted_margin = product_summary.sort_values("Avg_Margin_Pct")
        colors = [
            "#10B981" if m >= 50 else "#F59E0B" if m >= 35 else "#EC4899"
            for m in sorted_margin["Avg_Margin_Pct"]
        ]
        ax.barh(sorted_margin["Product Name"], sorted_margin["Avg_Margin_Pct"], color=colors)
        ax.axvline(40, color="red", linestyle="--", alpha=0.7, label="40% threshold")
        ax.set_xlabel("Avg Margin %")
        ax.set_title("📈 Margin % by Product", fontweight="bold")
        ax.legend(fontsize=8)

    def plot_division_revenue_profit(ax: plt.Axes) -> None:
        x = range(len(division_summary))
        width = 0.35
        ax.bar(
            [i - width / 2 for i in x],
            division_summary["Total_Sales"],
            width,
            label="Revenue",
            color="#06B6D4",
        )
        ax.bar(
            [i + width / 2 for i in x],
            division_summary["Total_Profit"],
            width,
            label="Profit",
            color="#7C3AED",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(division_summary["Division"])
        ax.set_title("🏢 Division: Revenue vs Profit", fontweight="bold")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    def plot_division_margin(ax: plt.Axes) -> None:
        colors = [
            "#10B981" if m >= 55 else "#F59E0B" if m >= 40 else "#EC4899"
            for m in division_summary["Avg_Margin"]
        ]
        ax.bar(division_summary["Division"], division_summary["Avg_Margin"], color=colors)
        ax.axhline(40, color="red", linestyle="--", alpha=0.7, label="40% threshold")
        ax.set_ylabel("Avg Margin %")
        ax.set_title("📊 Avg Margin by Division", fontweight="bold")
        ax.legend(fontsize=8)

    def plot_pareto(ax: plt.Axes) -> None:
        pareto_df = product_summary.sort_values("Total_Profit", ascending=False).copy()
        pareto_df["Cumulative_Pct"] = (
            pareto_df["Total_Profit"].cumsum() / pareto_df["Total_Profit"].sum() * 100
        )
        ax.bar(range(len(pareto_df)), pareto_df["Total_Profit"], color="#06B6D4", alpha=0.75)
        ax_r = ax.twinx()
        ax_r.plot(range(len(pareto_df)), pareto_df["Cumulative_Pct"], "r-o", markersize=4)
        ax_r.axhline(80, color="#F59E0B", linestyle="--", linewidth=1.5, label="80%")
        ax_r.set_ylabel("Cumulative %")
        ax_r.legend(fontsize=8)
        ax.set_xticks(range(len(pareto_df)))
        ax.set_xticklabels(
            [n[:10] for n in pareto_df["Product Name"]],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.set_title("📉 Pareto Analysis", fontweight="bold")

    def plot_clustering(ax: plt.Axes) -> None:
        cluster_colors = {
            "⭐ Star": "#10B981",
            "⚠ Average": "#F59E0B",
            "❌ Dead Weight": "#EC4899",
        }
        for label, group in product_summary.groupby("Cluster_Label"):
            ax.scatter(
                group["Total_Profit"],
                group["Avg_Margin_Pct"],
                label=label,
                color=cluster_colors.get(label, "gray"),
                s=180,
                zorder=5,
                edgecolors="white",
                linewidth=1.5,
            )
            for _, row in group.iterrows():
                ax.annotate(
                    row["Product Name"][:18],
                    (row["Total_Profit"], row["Avg_Margin_Pct"]),
                    fontsize=7.5,
                    xytext=(5, 5),
                    textcoords="offset points",
                )
        ax.set_xlabel("Total Profit ($)")
        ax.set_ylabel("Avg Margin %")
        ax.set_title("🤖 ML Clustering — Product Segments (K-Means)", fontweight="bold")
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.axhline(40, color="gray", linestyle=":", alpha=0.5)

    def save_single_chart(filename: str, plot_func: callable, size: tuple[int, int]) -> None:
        # Reuse plotting logic and keep chart exports consistent.
        fig, ax = plt.subplots(figsize=size)
        plot_func(ax)
        plt.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Save each chart separately
    save_single_chart(
        "nassau_chart_1_total_profit_by_product.png",
        plot_total_profit_by_product,
        (10, 8),
    )
    save_single_chart(
        "nassau_chart_2_margin_by_product.png",
        plot_margin_by_product,
        (8, 8),
    )
    save_single_chart(
        "nassau_chart_3_division_revenue_profit.png",
        plot_division_revenue_profit,
        (8, 6),
    )
    save_single_chart(
        "nassau_chart_4_division_margin.png",
        plot_division_margin,
        (8, 6),
    )
    save_single_chart(
        "nassau_chart_5_pareto.png",
        plot_pareto,
        (10, 6),
    )
    save_single_chart(
        "nassau_chart_6_clustering.png",
        plot_clustering,
        (10, 6),
    )

    # Combined 6-panel figure
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Nassau Candy — Product Profitability Analysis",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    plot_total_profit_by_product(fig.add_subplot(gs[0, :2]))
    plot_margin_by_product(fig.add_subplot(gs[0, 2]))
    plot_division_revenue_profit(fig.add_subplot(gs[1, 0]))
    plot_division_margin(fig.add_subplot(gs[1, 1]))
    plot_pareto(fig.add_subplot(gs[1, 2]))
    plot_clustering(fig.add_subplot(gs[2, :]))

    plt.tight_layout()
    plt.savefig("nassau_charts.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  ✅ Charts saved: nassau_charts.png")
    print("  ✅ Individual charts saved: nassau_chart_1_* through nassau_chart_6_*")


def print_footer() -> None:
    """Print the completion footer."""
    print("\n" + "=" * 60)
    print("  ✅ ANALYSIS COMPLETE!")
    print("  → nassau_charts.png saved")
    print("  → Now run 'streamlit run app.py' for the dashboard")
    print("=" * 60)


def main() -> None:
    """Run the end-to-end analysis script."""
    print_banner()

    # Step-by-step pipeline: load → enrich metrics → summarize → model → chart.
    df = load_and_preprocess()

    product_summary = summarize_products(df)
    division_summary = summarize_divisions(df)
    run_pareto_analysis(product_summary)
    product_summary = run_clustering(product_summary)

    build_charts(product_summary, division_summary)
    print_footer()


if __name__ == "__main__":
    main()