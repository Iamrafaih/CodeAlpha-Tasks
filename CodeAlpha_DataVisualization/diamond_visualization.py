
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. CONFIGURATION - one place to control the look of every chart

DATA_PATH = os.path.join("data", "diamonds.csv")
CHART_DIR = "charts"

INK = "#1a1a2e"          # main text colour
MUTED = "#6b7280"        # secondary text colour
ACCENT = "#c2410c"       # highlight colour for the key insight
GRID = "#e5e7eb"         # grid line colour
PALETTE = ["#0f766e", "#0891b2", "#6366f1", "#c026d3", "#e11d48"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 10,
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})

sns.set_palette(PALETTE)


def save(fig, filename):
    """Save a figure into the charts folder and report it."""
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path}")


def subtitle(fig, title, message):
    
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.text(0.012, 0.965, title, fontsize=16, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(0.012, 0.912, message, fontsize=10.5, color=MUTED,
             ha="left", va="top")


# 2. LOAD AND CLEAN THE DATA

def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Raw data:", df.shape[0], "rows,", df.shape[1], "columns")

    
    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    colour_order = ["J", "I", "H", "G", "F", "E", "D"]        # J worst, D best
    clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

    df["cut"] = pd.Categorical(df["cut"], cut_order, ordered=True)
    df["color"] = pd.Categorical(df["color"], colour_order, ordered=True)
    df["clarity"] = pd.Categorical(df["clarity"], clarity_order, ordered=True)

    
    bad_rows = (df[["x", "y", "z"]] == 0).any(axis=1).sum()
    df = df[(df[["x", "y", "z"]] != 0).all(axis=1)].copy()
    print(f"Removed {bad_rows} rows with impossible zero dimensions")

    
    df["price_per_carat"] = df["price"] / df["carat"]

    print("Clean data:", df.shape[0], "rows")
    print("Missing values:", int(df.isna().sum().sum()))
    return df

# 3. CHART 1 - How are diamond prices distributed?

def chart_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(df["price"], bins=60, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    axes[0].set_title("Price on a normal scale")
    axes[0].set_xlabel("Price (USD)")
    axes[0].set_ylabel("Number of diamonds")
    axes[0].xaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")

    median = df["price"].median()
    axes[0].axvline(median, color=ACCENT, linestyle="--", linewidth=2)
    axes[0].text(median * 1.15, axes[0].get_ylim()[1] * 0.8,
                 f"Median\n${median:,.0f}", color=ACCENT, fontweight="bold")

    axes[1].hist(np.log10(df["price"]), bins=60, color=PALETTE[2],
                 edgecolor="white", linewidth=0.5)
    axes[1].set_title("Same prices on a log scale")
    axes[1].set_xlabel("Price (log\u2081\u2080 USD)")
    axes[1].set_ylabel("Number of diamonds")

    subtitle(fig,
             "Most diamonds are cheap, a few are extremely expensive",
             f"Half of all stones sell below ${median:,.0f}, but the top price reaches "
             f"${df['price'].max():,.0f}. The log view reveals two hidden price clusters.")
    save(fig, "01_price_distribution.png")

# 4. CHART 2 - Does size drive price?

def chart_carat_vs_price(df):
    sample = df.sample(n=7000, random_state=42)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.scatterplot(data=sample, x="carat", y="price", hue="cut",
                    palette=PALETTE, alpha=0.55, s=16, edgecolor="none", ax=ax)

   
    grouped = df.groupby(pd.cut(df["carat"], bins=40), observed=True)["price"]
    trend = pd.DataFrame({"mean": grouped.mean(), "n": grouped.count()})
    trend = trend[trend["n"] >= 30]
    centres = [iv.mid for iv in trend.index]
    ax.plot(centres, trend["mean"].values, color=INK,
            linewidth=2.5, label="Average trend (n \u2265 30)")

    r = df["carat"].corr(df["price"])
    ax.set_title("Carat weight versus price")
    ax.set_xlabel("Carat (weight)")
    ax.set_ylabel("Price (USD)")
    ax.set_xlim(0, 3.2)
    ax.yaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")
    ax.legend(title="Cut quality", frameon=False, loc="upper left")

    ax.text(0.98, 0.04, f"Correlation = {r:.2f}", transform=ax.transAxes,
            ha="right", fontsize=12, fontweight="bold", color=ACCENT)

    subtitle(fig,
             "Size is by far the strongest price driver",
             "Price rises faster than weight - a 2 carat stone costs far more than "
             "two 1 carat stones. Notice the vertical stacks at round weights.")
    save(fig, "02_carat_vs_price.png")

# 5. CHART 3 - The counter-intuitive result

def chart_quality_paradox(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    
    by_cut = df.groupby("cut", observed=True)["price"].mean()
    bars = axes[0].bar(by_cut.index.astype(str), by_cut.values,
                       color=[ACCENT if c == "Ideal" else PALETTE[1] for c in by_cut.index],
                       edgecolor="white", linewidth=1.5)
    axes[0].set_title("Average price by cut quality")
    axes[0].set_ylabel("Average price (USD)")
    axes[0].yaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")
    for bar, value in zip(bars, by_cut.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 90,
                     f"${value:,.0f}", ha="center", fontsize=9.5,
                     fontweight="bold", color=INK)
    axes[0].text(0.5, 0.92, "Best cut looks cheapest?!", transform=axes[0].transAxes,
                 ha="center", color=ACCENT, fontweight="bold", fontsize=11)

    carat_by_cut = df.groupby("cut", observed=True)["carat"].mean()
    axes[1].bar(carat_by_cut.index.astype(str), carat_by_cut.values,
                color=[ACCENT if c == "Ideal" else PALETTE[0] for c in carat_by_cut.index],
                edgecolor="white", linewidth=1.5)
    axes[1].set_title("Average size by cut quality")
    axes[1].set_ylabel("Average carat")
    for i, value in enumerate(carat_by_cut.values):
        axes[1].text(i, value + 0.01, f"{value:.2f}", ha="center", fontsize=9.5,
                     fontweight="bold", color=INK)
    axes[1].text(0.5, 0.92, "...because it is the smallest",
                 transform=axes[1].transAxes, ha="center", color=ACCENT,
                 fontweight="bold", fontsize=11)

    subtitle(fig,
             "A trap in the data: better cut appears to mean lower price",
             "The effect is caused by size, not quality. Cutters spend their best work on "
             "small stones, so the top grade carries the lightest average weight.")
    save(fig, "03_quality_paradox.png")

# 6. CHART 4 - Fixing the comparison with price per carat

def chart_price_per_carat(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, column, title in zip(
        axes,
        ["cut", "color", "clarity"],
        ["Cut (worst to best)", "Colour (worst to best)", "Clarity (worst to best)"],
    ):
        means = df.groupby(column, observed=True)["price_per_carat"].mean()
        ax.plot(means.index.astype(str), means.values, marker="o", markersize=8,
                linewidth=2.5, color=PALETTE[2])
        ax.fill_between(range(len(means)), means.values, means.values.min() * 0.95,
                        alpha=0.12, color=PALETTE[2])
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Price per carat (USD)" if column == "cut" else "")
        ax.yaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")
        ax.tick_params(axis="x", rotation=45 if column == "clarity" else 0)

    subtitle(fig,
             "Once size is held constant, quality does pay",
             "Dividing price by carat removes the size effect. All three quality grades now "
             "reward the buyer in the direction you would expect.")
    save(fig, "04_price_per_carat_by_quality.png")

# 7. CHART 5 - Which numeric features move together?

def chart_correlation(df):
    numeric = df[["carat", "depth", "table", "price", "x", "y", "z"]]
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(8.5, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=2,
                linecolor="white", cbar_kws={"shrink": 0.7, "label": "Correlation"},
                annot_kws={"fontsize": 10, "fontweight": "bold"}, ax=ax)
    ax.set_title("Correlation between numeric features")

    subtitle(fig,
             "Weight and physical dimensions explain almost all of the price",
             "Carat and the x, y, z measurements all score above 0.86 against price. "
             "Depth and table percentages are close to zero - they barely matter.")
    save(fig, "05_correlation_heatmap.png")

# 8. CHART 6 - What does the market actually stock?

def chart_market_composition(df):
    pivot = pd.crosstab(df["cut"], df["clarity"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(pivot, annot=True, fmt=",d", cmap="YlGnBu", linewidths=2,
                linecolor="white", cbar_kws={"label": "Number of diamonds"},
                annot_kws={"fontsize": 9}, ax=ax)
    ax.set_title("Inventory count by cut and clarity")
    ax.set_xlabel("Clarity")
    ax.set_ylabel("Cut")

    top = pivot.stack().idxmax()
    subtitle(fig,
             "Supply is concentrated in the mid-range, not at the extremes",
             f"The most common combination is {top[0]} cut with {top[1]} clarity. "
             "Flawless stones are rare, which is exactly why they command a premium.")
    save(fig, "06_market_composition.png")

# 9. CHART 7 - Price spread inside each size band

def chart_price_bands(df):
    bands = pd.cut(df["carat"],
                   bins=[0, 0.5, 0.75, 1.0, 1.5, 2.0, 6],
                   labels=["<0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", "1.5-2.0", "2.0+"])
    data = df.assign(band=bands)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=data, x="band", y="price", hue="band", legend=False,
                palette="crest", fliersize=1, linewidth=1.2, ax=ax)
    ax.set_title("Price range within each size band")
    ax.set_xlabel("Carat band")
    ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(lambda v, p: f"${v:,.0f}")

    # Sample sizes sit in a reserved strip below the boxes.
    ax.set_ylim(-2600, df["price"].max() * 1.05)
    ax.set_yticks(np.arange(0, df["price"].max() + 2500, 2500))
    counts = data["band"].value_counts().sort_index()
    for i, n in enumerate(counts.values):
        ax.text(i, -1700, f"n = {n:,}", ha="center", fontsize=9, color=MUTED)

    subtitle(fig,
             "Bigger stones are not just pricier, they are far less predictable",
             "The box heights grow with size. Below half a carat almost every stone is "
             "cheap; above two carats the price depends heavily on quality.")
    save(fig, "07_price_bands.png")

# 10. CHART 8 - The single-page dashboard

def chart_dashboard(df):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.28,
                          height_ratios=[0.22, 1, 1])

    
    header = fig.add_subplot(gs[0, :])
    header.axis("off")
    kpis = [
        (f"{len(df):,}", "Diamonds analysed"),
        (f"${df['price'].median():,.0f}", "Median price"),
        (f"{df['carat'].mean():.2f}", "Average carat"),
        (f"{df['carat'].corr(df['price']):.2f}", "Carat-price correlation"),
    ]
    for i, (value, label) in enumerate(kpis):
        x = 0.02 + i * 0.25
        header.text(x, 0.62, value, fontsize=26, fontweight="bold", color=INK)
        header.text(x, 0.18, label, fontsize=10.5, color=MUTED)

    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(df["price"], bins=45, color=PALETTE[0], edgecolor="white", linewidth=0.4)
    ax1.set_title("Price distribution", fontsize=12)
    ax1.set_xlabel("Price (USD)")
    ax1.set_ylabel("Count")
    ax1.xaxis.set_major_formatter(lambda v, p: f"${v/1000:.0f}k")

    
    ax2 = fig.add_subplot(gs[1, 1])
    s = df.sample(4000, random_state=42)
    ax2.scatter(s["carat"], s["price"], s=6, alpha=0.35, color=PALETTE[2],
                edgecolor="none")
    ax2.set_title("Size drives price", fontsize=12)
    ax2.set_xlabel("Carat")
    ax2.set_ylabel("Price (USD)")
    ax2.set_xlim(0, 3)
    ax2.yaxis.set_major_formatter(lambda v, p: f"${v/1000:.0f}k")

    
    ax3 = fig.add_subplot(gs[1, 2])
    ppc = df.groupby("clarity", observed=True)["price_per_carat"].mean()
    ax3.plot(ppc.index.astype(str), ppc.values, marker="o", color=ACCENT, linewidth=2.5)
    ax3.set_title("Clarity pays, per carat", fontsize=12)
    ax3.set_ylabel("Price per carat")
    ax3.tick_params(axis="x", rotation=45)
    ax3.yaxis.set_major_formatter(lambda v, p: f"${v/1000:.1f}k")

    
    ax4 = fig.add_subplot(gs[2, 0])
    by_cut = df.groupby("cut", observed=True)["price"].mean()
    ax4.bar(by_cut.index.astype(str), by_cut.values, color=PALETTE[1],
            edgecolor="white", linewidth=1.2)
    ax4.set_title("Average price by cut", fontsize=12)
    ax4.set_ylabel("Price (USD)")
    ax4.tick_params(axis="x", rotation=20)
    ax4.yaxis.set_major_formatter(lambda v, p: f"${v/1000:.0f}k")

    
    ax5 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=df, x="color", y="price", hue="color", legend=False,
                palette="crest", fliersize=0.6, linewidth=1, ax=ax5)
    ax5.set_title("Price spread by colour", fontsize=12)
    ax5.set_xlabel("Colour (J worst - D best)")
    ax5.set_ylabel("Price (USD)")
    ax5.yaxis.set_major_formatter(lambda v, p: f"${v/1000:.0f}k")

    
    ax6 = fig.add_subplot(gs[2, 2])
    corr = df[["carat", "x", "y", "z", "table", "depth"]].corrwith(df["price"]).sort_values()
    colours = [ACCENT if v > 0.5 else MUTED for v in corr.values]
    ax6.barh(corr.index, corr.values, color=colours, edgecolor="white", linewidth=1.2)
    ax6.set_title("What correlates with price", fontsize=12)
    ax6.set_xlabel("Correlation with price")
    ax6.set_xlim(-0.2, 1)

    fig.suptitle("Diamond Price Analysis - Executive Dashboard",
                 fontsize=20, fontweight="bold", color=INK, x=0.012, ha="left", y=1.0)
    fig.text(0.012, 0.963,
             "Weight explains the price. Cut, colour and clarity only reveal their value "
             "once you compare stones of equal size.",
             fontsize=11.5, color=MUTED, ha="left")

    save(fig, "08_dashboard.png")

# 11. PRINTED SUMMARY - the findings in plain language

def print_insights(df):
    print("\n" + "=" * 68)
    print("KEY FINDINGS")
    print("=" * 68)

    print(f"1. Carat correlates {df['carat'].corr(df['price']):.2f} with price - "
          "the strongest single driver.")

    ideal = df[df["cut"] == "Ideal"]
    fair = df[df["cut"] == "Fair"]
    print(f"2. Ideal-cut stones average ${ideal['price'].mean():,.0f} versus "
          f"${fair['price'].mean():,.0f} for Fair cut, because Ideal stones are "
          f"smaller ({ideal['carat'].mean():.2f} vs {fair['carat'].mean():.2f} carat).")

    print(f"3. Per carat, Ideal cut earns ${ideal['price_per_carat'].mean():,.0f} "
          f"against ${fair['price_per_carat'].mean():,.0f} for Fair - "
          "the ranking flips once size is controlled for.")

    best = df[df["clarity"] == "IF"]["price_per_carat"].mean()
    worst = df[df["clarity"] == "I1"]["price_per_carat"].mean()
    print(f"4. Flawless clarity earns ${best:,.0f} per carat versus ${worst:,.0f} "
          f"for the lowest grade - a {best/worst:.1f}x premium.")

    print(f"5. Depth and table percentages correlate "
          f"{df['depth'].corr(df['price']):.2f} and {df['table'].corr(df['price']):.2f} "
          "with price - effectively irrelevant to buyers.")
    print("=" * 68)

# 12. MAIN

def main():
    print("CodeAlpha Task 3 - Data Visualization\n" + "-" * 40)
    df = load_data()

    print("\nBuilding charts...")
    chart_price_distribution(df)
    chart_carat_vs_price(df)
    chart_quality_paradox(df)
    chart_price_per_carat(df)
    chart_correlation(df)
    chart_market_composition(df)
    chart_price_bands(df)
    chart_dashboard(df)

    print_insights(df)
    print(f"\nDone. All charts are in the '{CHART_DIR}' folder.")


if __name__ == "__main__":
    main()
