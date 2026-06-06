import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "feature_repo" / "data" / "aqi_features.parquet"
OUTPUT_DIR = ROOT / "eda_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PARQUET.exists():
    print(f"Parquet not found at: {PARQUET}")
    print("Run feature_store_pipeline.py first.")
    sys.exit(1)

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#FFC107", "#9C27B0", "#00BCD4"]

print("Loading data...")
df = pd.read_parquet(PARQUET)
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
df = df.sort_values("event_timestamp").reset_index(drop=True)

print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")
print(f"Range: {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")


def aqi_category(val):
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Moderate"
    elif val <= 150:
        return "Unhealthy (Sensitive)"
    elif val <= 200:
        return "Unhealthy"
    elif val <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


df["aqi_category"] = df["aqi"].apply(aqi_category)

print("\nPlot 1: Missing Values...")
fig, ax = plt.subplots(figsize=(12, 5))

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    ax.text(
        0.5,
        0.5,
        "No Missing Values!",
        transform=ax.transAxes,
        fontsize=20,
        ha="center",
        va="center",
        color="green",
    )
    ax.set_title("Missing Values Check", fontsize=14, fontweight="bold")
else:
    missing.plot(kind="bar", ax=ax, color=COLORS[1])
    ax.set_title("Missing Values per Column", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_missing_values.png", dpi=150)
plt.close()

print("Plot 2: AQI Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    df["aqi"].dropna(),
    bins=50,
    color=COLORS[0],
    edgecolor="white",
    alpha=0.85,
)

axes[0].axvline(
    df["aqi"].mean(),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {df['aqi'].mean():.1f}",
)

axes[0].axvline(
    df["aqi"].median(),
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Median: {df['aqi'].median():.1f}",
)

axes[0].set_title("AQI Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("AQI Value")
axes[0].set_ylabel("Frequency")
axes[0].legend()

cat_counts = df["aqi_category"].value_counts()
cat_colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336", "#9C27B0", "#B71C1C"]

axes[1].pie(
    cat_counts.values,
    labels=cat_counts.index,
    autopct="%1.1f%%",
    colors=cat_colors[: len(cat_counts)],
    startangle=140,
)

axes[1].set_title("AQI Category Breakdown", fontsize=13, fontweight="bold")

plt.suptitle(
    "Lahore AQI — Distribution Analysis",
    fontsize=15,
    fontweight="bold",
    y=1.01,
)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_aqi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("Plot 3: AQI Over Time...")
fig, ax = plt.subplots(figsize=(16, 5))

daily_avg = df.groupby(df["event_timestamp"].dt.date)["aqi"].mean()
daily_avg.index = pd.to_datetime(daily_avg.index)

ax.plot(
    daily_avg.index,
    daily_avg.values,
    color=COLORS[0],
    linewidth=1.2,
    alpha=0.7,
)

ax.fill_between(
    daily_avg.index,
    daily_avg.values,
    alpha=0.2,
    color=COLORS[0],
)

for threshold, color, label in [
    (50, "#4CAF50", "Good (50)"),
    (100, "#FFC107", "Moderate (100)"),
    (150, "#FF9800", "Unhealthy (150)"),
    (200, "#F44336", "Very Unhealthy (200)"),
]:
    ax.axhline(
        threshold,
        color=color,
        linestyle="--",
        alpha=0.7,
        linewidth=1,
        label=label,
    )

ax.set_title("Lahore AQI — Daily Average Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.legend(loc="upper right", fontsize=8)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_aqi_over_time.png", dpi=150)
plt.close()

print("Plot 4: Monthly Seasonal Pattern...")
fig, ax = plt.subplots(figsize=(12, 5))

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly = df.groupby(df["event_timestamp"].dt.month)["aqi"].mean()

bars = ax.bar(monthly.index, monthly.values, color=COLORS, width=0.6)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)

ax.set_title("Monthly Average AQI — Seasonal Pattern", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average AQI")

for bar, val in zip(bars, monthly.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_monthly_aqi.png", dpi=150)
plt.close()

print("Plot 5: Hourly AQI Pattern...")
fig, ax = plt.subplots(figsize=(12, 5))

hourly = df.groupby("hour")["aqi"].mean()

ax.plot(
    hourly.index,
    hourly.values,
    color=COLORS[1],
    linewidth=2.5,
    marker="o",
    markersize=5,
)

ax.fill_between(
    hourly.index,
    hourly.values,
    alpha=0.2,
    color=COLORS[1],
)

ax.set_title("Average AQI by Hour of Day (UTC)", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Average AQI")
ax.set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_hourly_aqi.png", dpi=150)
plt.close()