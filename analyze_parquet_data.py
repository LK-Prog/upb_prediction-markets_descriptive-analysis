#!/.venv/bin/python
# TODO This was pretty much just thrown together and is in dire need of refactoring

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================
# SETTINGS
# ============================================================
DATA_PATH = Path("merged/data.parquet")
DURATION = "start_to_scheduled_end_days"

# unit of analysis:
# "market" = one observation per market (recommended for market durations)
# "event"  = one observation per event, selecting one market's duration per event.
# The latter is NOT necessarily a well-defined event-wide duration.
ANALYSIS_LEVEL = "market"

MIN_CATEGORY_EVENTS = 100      # Minimum observations for a category comparison
MIN_HEATMAP_EVENTS = 30        # Minimum observations in each platform/category cell
TOP_CATEGORIES = 10            # Number of major categories to display

sns.set_theme(style="whitegrid")


def print_section(number, heading):
    """Give each console output an obvious heading."""
    print(f"\n{number}. {heading}")
    print("=" * 70)


def duration_stats(frame, group_fields):
    """Compute comparable duration statistics for groups of markets.

    The median describes a typical market; p90 describes the longer tail.
    Percentage statistics are on a 0-to-100 scale.
    """
    return (
        frame.groupby(group_fields, dropna=False)[DURATION]
        .agg(
            markets="size",
            mean_days="mean",
            median_days="median",
            p90_days=lambda s: s.quantile(0.90),
            pct_within_1_day=lambda s: 100 * (s <= 1).mean(),
            pct_over_30_days=lambda s: 100 * (s > 30).mean(),
        )
        .sort_values("markets", ascending=False)
    )


def main():
    section_number = 0
    
    # ========================================================
    # 1. LOAD DATA AND CHOOSE THE OBSERVATION LEVEL
    # ========================================================
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Set DATA_PATH to your parquet file."
        )

    raw = pd.read_parquet(DATA_PATH)

    # Platform must be part of the key: ID namespaces are platform-specific.
    # Keeping the latest retrieved row avoids counting repeated snapshots.
    if ANALYSIS_LEVEL == "market":
        identifier = "market_id"
    elif ANALYSIS_LEVEL == "event":
        identifier = "event_id"
    else:
        raise ValueError('ANALYSIS_LEVEL must be "market" or "event"')

    # Missing identifiers cannot reliably be deduplicated; keep each such row.
    with_id = raw.loc[raw[identifier].notna()]
    without_id = raw.loc[raw[identifier].isna()]

    deduped = (
        with_id.sort_values("retrieved_at")
        .drop_duplicates(["platform", identifier], keep="last")
    )
    df = pd.concat([deduped, without_id], ignore_index=True).copy()

    df[DURATION] = pd.to_numeric(df[DURATION], errors="coerce")

    section_number = section_number + 1
    print_section(section_number, "DATA COVERAGE AND QUALITY")
    print(f"Records in source file:                 {len(raw):,}")
    print(f"Unique analysis level:                  {ANALYSIS_LEVEL}")
    print(f"Records after deduplication:            {len(df):,}")
    print(f"Missing market duration:                {df[DURATION].isna().sum():,}")
    print(f"Negative market duration:               {(df[DURATION] < 0).sum():,}")
    print(f"Exactly zero-day duration:              {(df[DURATION] == 0).sum():,}")
    print(f"Missing event category:                 {df['event_category'].isna().sum():,}")

    # Negative and missing values are excluded, not silently turned into zero.
    valid = df.loc[df[DURATION].notna() & (df[DURATION] >= 0)].copy()
    if valid.empty:
        raise ValueError("No non-negative, non-missing market durations found.")

    print(f"Valid durations used in analysis:       {len(valid):,}")
    print(f"Usable percentage after deduplication:  {100 * len(valid) / len(df):.1f}%")

    # ========================================================
    # 2. DESCRIPTIVE STATISTICS AND PERCENTILES
    # ========================================================
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    duration = valid[DURATION]
    section_number = section_number + 1
    print_section(section_number, "OVERALL DURATION STATISTICS (DAYS)")
    print(duration.describe(percentiles=percentiles).round(3))
    print(
        "Interpretation: Median = 50% of markets are shorter; "
        "p90 = 90% are shorter. A mean much larger than the median "
        "suggests a long right-hand tail."
    )
    
    duration_kalshi = valid[valid['platform'] == 'kalshi'][DURATION]
    section_number = section_number + 1
    print_section(section_number, "KALSHI DURATION STATISTICS (DAYS)")
    print(duration_kalshi.describe(percentiles=percentiles).round(3))
    print(
        "Interpretation: Median = 50% of markets are shorter; "
        "p90 = 90% are shorter. A mean much larger than the median "
        "suggests a long right-hand tail."
    )
    
    duration_polymarket = valid[valid['platform'] == 'polymarket'][DURATION]
    section_number = section_number + 1
    print_section(section_number, "POLYMARKET DURATION STATISTICS (DAYS)")
    print(duration_polymarket.describe(percentiles=percentiles).round(3))
    print(
        "Interpretation: Median = 50% of markets are shorter; "
        "p90 = 90% are shorter. A mean much larger than the median "
        "suggests a long right-hand tail."
    )

    # ========================================================
    # 3. COUNT MARKETS IN HUMAN-READABLE DURATION RANGES
    # ========================================================
    # A date interval is open on the left and closed on the right:
    # >1 day to 7 days means 1 < duration <= 7.
    group_labels = [
        "Exactly 0 days",
        ">0 to 1 hour",
        ">1 hour to 1 day",
        ">1 to 7 days",
        ">7 to 30 days",
        ">30 to 90 days",
        ">90 to 365 days",
        ">365 days",
    ]
    valid["duration_group"] = pd.cut(
        valid[DURATION],
        bins=[-np.inf, 0, 1 / 24, 1, 7, 30, 90, 365, np.inf],
        labels=group_labels,
    )
    counts = valid["duration_group"].value_counts(sort=False)
    duration_bands = pd.DataFrame({
        "markets": counts,
        "percentage": 100 * counts / len(valid),
    })

    section_number = section_number + 1
    print_section(section_number, "HOW COMMON IS EACH DURATION RANGE?")
    print(duration_bands.round(2).to_string())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(duration_bands.index.astype(str), duration_bands["percentage"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlim(0, max(2, duration_bands["percentage"].max() * 1.2))
    for index, share in enumerate(duration_bands["percentage"]):
        ax.text(share + 0.3, index, f"{share:.1f}%", va="center")
    ax.set(title="How Long Are Markets Scheduled to Last?", xlabel="Share of markets (%)", ylabel="Duration range")
    plt.tight_layout()
    plt.savefig("analysis/bar_market_durations_combined.svg")
    # plt.show()

    # ========================================================
    # 3. COMPARE DURATION RANGES BETWEEN PLATFORMS
    # ========================================================
    # Do Kalshi and Polymarket differ in how long their
    # markets are scheduled to last?
    # 
    # Each platform is normalized independently:
    # 100% = all valid markets on that platform.
    # ========================================================
    group_labels = [
        "Exactly 0 days",
        ">0 to 1 hour",
        ">1 hour to 1 day",
        ">1 to 7 days",
        ">7 to 30 days",
        ">30 to 90 days",
        ">90 to 365 days",
        ">365 days",
    ]

    # Assign each market to a duration range
    valid["duration_group"] = pd.cut(
        valid[DURATION],
        bins=[-np.inf, 0, 1 / 24, 1, 7, 30, 90, 365, np.inf],
        labels=group_labels,
    )

    # Count markets in each duration range, separately for each platform
    duration_by_platform = (
        valid.groupby(
            ["platform", "duration_group"],
            observed=False
        )
        .size()
        .reset_index(name="markets")
    )

    # Calculate percentages WITHIN each platform
    duration_by_platform["percentage"] = (
        100 * duration_by_platform["markets"]
        / duration_by_platform.groupby("platform")["markets"]
        .transform("sum")
    )

    # Print comparison table
    comparison_table = duration_by_platform.pivot(
        index="duration_group",
        columns="platform",
        values="percentage"
    )

    section_number = section_number + 1
    print_section(section_number, "DURATION DISTRIBUTION BY PLATFORM")
    print(comparison_table.round(2).to_string())


    # ========================================================
    # 5. VISUALIZE COMPARISON
    # ========================================================
    fig, ax = plt.subplots(figsize=(13, 7))

    sns.barplot(
        data=duration_by_platform,
        x="percentage",
        y="duration_group",
        hue="platform",
        order=group_labels,
        palette="Set2",
        ax=ax
    )

    # Display percentages next to the bars
    for container in ax.containers:
        labels = [
            f"{bar.get_width():.1f}%"
            if bar.get_width() > 0 else ""
            for bar in container
        ]

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9
        )

    # Formatting
    ax.set_xlim(
        0,
        min(100, duration_by_platform["percentage"].max() * 1.2 + 2)
    )

    ax.set_xlabel("Share of markets within each platform (%)")
    ax.set_ylabel("Scheduled duration")
    ax.set_title(
        "Distribution of Scheduled Market Durations\n"
        "Comparison Between Platforms"
    )

    ax.legend(title="Platform")
    plt.tight_layout()
    plt.savefig("analysis/bar_market_durations_by_platform.svg")
    # plt.show()

    # ========================================================
    # 4. LOGARITHMIC HISTOGRAM FOR RIGHT-SKEWED DATA
    # ========================================================
    positive = duration[duration > 0]
    section_number = section_number + 1
    print_section(section_number, "DETAILS OF POSITIVE DURATIONS")
    print(f"Positive durations: {len(positive):,}; zero durations are shown in Section 3.")
    if positive.nunique() > 1:
        edges = np.geomspace(positive.min(), positive.max(), 50)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(x=positive, bins=edges, color="steelblue", ax=ax)
        ax.set_xscale("log")
        ax.axvline(positive.median(), color="crimson", linestyle="--", label=f"Median (positive only): {positive.median():.2f} days")
        ax.set(title="Scheduled Duration Distribution (Positive Values)", xlabel="Days (logarithmic scale)", ylabel="Number of markets per logarithmic bin")
        ax.legend()
        plt.tight_layout()
        plt.savefig("analysis/hist_log_durations_combined.svg")
        # plt.show()
        print("Note: The bins have equal width on the log scale, but cover different numbers of days.")
    else:
        print("Not enough distinct positive values for a log histogram.")

    # ========================================================
    # 5. CHECK HOW MANY MARKETS BELONG TO EACH EVENT
    # ========================================================
    # Event/market distinction affects weight of event with multiple outcomes receives in analysis
    markets_per_event = (
        raw.dropna(subset=["event_id", "market_id"])
        .groupby(["platform", "event_id"])["market_id"]
        .nunique()
    )
    section_number = section_number + 1
    print_section(section_number, "EVENT-TO-MARKET RELATIONSHIP BY PLATFORM")
    if not markets_per_event.empty:
        event_structure = markets_per_event.groupby(level="platform").agg(
            events="size",
            average_markets_per_event="mean",
            median_markets_per_event="median",
            max_markets_per_event="max",
            pct_multi_market_events=lambda s: 100 * (s > 1).mean(),
        )
        print(event_structure.round(2).to_string())
    else:
        print("No non-missing event/market IDs found.")

    # ========================================================
    # 6. COMPARE OVERALL DURATIONS BETWEEN PLATFORMS
    # ========================================================
    section_number = section_number + 1
    print_section(section_number, "DURATION STATISTICS BY PLATFORM")
    platform_stats = duration_stats(valid, "platform")
    print(platform_stats.round(2).to_string())

    plot_platforms = platform_stats.sort_values("pct_within_1_day")
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_platforms) * 0.7)))
    ax.barh(plot_platforms.index.astype(str), plot_platforms["pct_within_1_day"], color="teal")
    ax.set(xlim=(0, 100), title="Share of Markets Lasting At Most One Day, by Platform", xlabel="Share within one day (%)", ylabel="Platform")
    plt.tight_layout()
    plt.savefig("analysis/bar_one_day_by_platform.svg")
    # plt.show()

    # ========================================================
    # 7. MISSING CATEGORY LABELS BY PLATFORM
    # ========================================================
    # If category coverage differs a lot between platforms, comparisons of known categories may be unrepresentative
    section_number = section_number + 1
    print_section(section_number, "EVENT-CATEGORY COMPLETENESS BY PLATFORM")
    coverage = valid.groupby("platform")["event_category"].agg(
        markets="size",
        categorized_markets=lambda s: s.notna().sum(),
        missing_category_pct=lambda s: 100 * s.isna().mean(),
    )
    print(coverage.round(2).to_string())

    # ========================================================
    # 8. COMPARE EVENT CATEGORIES ACROSS ALL PLATFORMS
    # ========================================================
    # Exclude missing category labels from this comparison
    # We retain them in the overall/platform comparisons
    section_number = section_number + 1
    print_section(section_number, "CATEGORIES COUNTS BY PLATFORM")
    category_counts = (
        df.groupby(["platform", "event_category"], dropna=False)
            .size()
            .rename("market_count")
        ).groupby(level="platform", group_keys=False).apply(lambda group: group.sort_values(ascending=False))

    print(category_counts)
    
    known = valid.dropna(subset=["event_category"]).copy()
    section_number = section_number + 1
    print_section(section_number, "DURATION STATISTICS BY KNOWN EVENT CATEGORY")
    if known.empty:
        print("No known event categories; skipping category charts.")
    else:
        category_stats = duration_stats(known, "event_category")
        eligible = category_stats.loc[category_stats["markets"] >= MIN_CATEGORY_EVENTS]
        print(eligible.round(2).to_string())
        print(f"Categories with fewer than {MIN_CATEGORY_EVENTS} markets are not listed above.")

        top = eligible.head(TOP_CATEGORIES).sort_values("pct_within_1_day")
        if not top.empty:
            fig, ax = plt.subplots(figsize=(11, max(5, len(top) * 0.55)))
            ax.barh(top.index.astype(str), top["pct_within_1_day"], color="seagreen")
            ax.set(xlim=(0, 100), title="Short-Duration Markets by Event Category", xlabel="Share lasting at most one day (%)", ylabel="Category")
            plt.tight_layout()
            plt.savefig("analysis/bar_duartions_by_category.svg")
            # plt.show()

        # ====================================================
        # 9. CONTROL FOR CATEGORY WHEN COMPARING PLATFORMS
        # ====================================================
        # This is a descriptive within-category comparison, not evidence of a causal platform effect
        most_common = known["event_category"].value_counts().head(TOP_CATEGORIES).index
        subset = known[known["event_category"].isin(most_common)]
        cross_stats = duration_stats(subset, ["platform", "event_category"])
        section_number = section_number + 1
        print_section(section_number, "PLATFORM × CATEGORY DURATION STATISTICS")
        print(cross_stats.round(2).to_string())

        # A heatmap compares the SAME category across platforms
        # Cells with fewer than 30 markets remain blank, not zero
        heatmap_stats = cross_stats.loc[cross_stats["markets"] >= MIN_HEATMAP_EVENTS]
        if not heatmap_stats.empty:
            matrix = heatmap_stats["pct_within_1_day"].unstack("platform")
            fig, ax = plt.subplots(figsize=(max(8, len(matrix.columns) * 1.8), max(5, len(matrix) * 0.55)))
            sns.heatmap(
                matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                vmin=0, vmax=100, linewidths=0.4,
                cbar_kws={"label": "Share lasting at most one day (%)"}, ax=ax,
            )
            ax.set(title="Short-Duration Share: Platform x Category", xlabel="Platform", ylabel="Event category")
            plt.tight_layout()
            plt.savefig("analysis/heat_short_bet_by_category.svg")
            # plt.show()
        else:
            print("No platform/category cells meet the minimum heatmap sample size.")

    # ========================================================
    # 10. INSPECT THE LONGEST SCHEDULED DURATIONS
    # ========================================================
    # Very long values might represent genuine future markets or source-data errors; inspect dates before excluding them
    columns_to_show = [
        "platform", "event_id", "market_id", "market_title",
        "event_category", "start_at", "scheduled_end_at", DURATION,
    ]
    section_number = section_number + 1
    print_section(section_number, "TEN LONGEST SCHEDULED MARKET DURATIONS")
    print(valid.nlargest(10, DURATION)[columns_to_show].to_string(index=False))

    section_number = section_number + 1
    print_section(section_number, "50 SHORTEST CRYPTO DURATIONS")
    print(valid[valid['event_category'] == "Crypto"].nsmallest(50, DURATION)[columns_to_show].to_string(index=False))

    # ========================================================
    # 11. COMPARE PLANNED AND OBSERVED CLOSING DURATIONS
    # ========================================================
    # This subset only includes markets with an observed close
    # It cannot measure an individual bettor's holding period
    observed = "start_to_observed_close_days"
    paired = valid.loc[valid[observed].notna() & (valid[observed] >= 0)].copy()
    section_number = section_number + 1
    print_section(section_number, "SCHEDULED VS OBSERVED DURATION (MATCHED ROWS)")
    if not paired.empty:
        paired["scheduled_minus_observed_days"] = paired[DURATION] - paired[observed]
        print(f"Markets with both durations:             {len(paired):,}")
        print(f"Median scheduled duration (days):        {paired[DURATION].median():.2f}")
        print(f"Median observed duration (days):         {paired[observed].median():.2f}")
        print(f"Closed earlier than scheduled (%):       {100 * (paired['scheduled_minus_observed_days'] > 0).mean():.1f}")
        print("Warning: observed-close data are missing for many markets; this is a selected subset.")
    else:
        print("No valid paired observed-close durations found.")
    
    # ========================================================
    # 12. HOW HAVE MARKET DURATIONS CHANGED OVER TIME?
    # ========================================================
    # Have the typical scheduled durations of markets on
    # Kalshi and Polymarket changed over time?
    # ========================================================
    # TODO Implement timeseries comparission of markets, e.g., through durations, volumes, category lifecylce
    
    
    # Show charts
    # plt.show() # show generated



if __name__ == "__main__":
    main()
