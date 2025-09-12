import numpy as np
import polars as pl
import altair as alt
from pathlib import Path
from typing import Dict, Any, Literal

from ..core.config import settings


def parse_bin_interval(interval_str: str, side: Literal["left", "right"]) -> float:
    # Remove brackets/parentheses
    cleaned = interval_str.strip("()[]")
    nums = cleaned.split(",")
    left, right = nums[0].strip(), nums[1].strip()

    def safe_float(val: str) -> float:
        if val == "-inf":
            val = "1"
        return float(val)

    return safe_float(left) if side == "left" else safe_float(right)


def convert_to_human_readable(n: int) -> str:
    """Format large ints nicely (e.g., 1000 -> '1k')."""
    if n >= 1_000_000:
        return f"{n//1_000_000}M"
    elif n >= 1_000:
        return f"{n//1_000}k"
    return str(n)


def generate_plot_spec(file_path: Path, operation: Dict[str, Any]) -> Dict[str, Any]:
    lf: pl.LazyFrame = pl.scan_parquet(file_path)

    column = operation.get("column")
    if not column:
        raise ValueError("Plot operation must specify a 'column'")

    chart_title = f"'{column}' in {file_path.parent.parent.parent.name}"

    # --- Config ---
    n_bins = operation.get("bins", 50)

    min_val = lf.select(column).min().collect().item()
    max_val = lf.select(column).max().collect().item()
    if min_val is None or max_val is None or min_val <= 0:
        raise ValueError(f"Column {column} has no valid values for histogram")

    # --- Precompute log-spaced bin edges ---
    bin_edges = np.logspace(np.log2(min_val), np.log2(max_val), n_bins, base=2)

    # --- Assign values to bins and aggregate ---
    binned = (
        lf.select(pl.col("token_count").cast(pl.Int64).alias("lengths"))
        .with_columns(pl.col("lengths").cut(bin_edges).alias("bin"))
        .group_by("bin")
        .agg(pl.count().alias("count"))
        .sort("bin")
        .collect(engine="streaming")
    )

    # Convert to pandas
    df = binned.to_pandas()

    # Parse bin edges from strings
    df["bin_left"] = (
        df["bin"].apply(lambda x: parse_bin_interval(x, side="left")).astype(float)
    )
    df["bin_right"] = (
        df["bin"].apply(lambda x: parse_bin_interval(x, side="right")).astype(float)
    )

    df = df.sort_values("bin_left")

    # Calculate actual bin widths for proper bar spacing
    df["bin_width"] = df["bin_right"] - df["bin_left"]

    # Handle infinite bin widths (for rightmost bin that goes to infinity)
    # Replace inf widths with a reasonable width based on the previous bin
    inf_mask = np.isinf(df["bin_width"])
    if inf_mask.any():
        # Use the width of the previous bin for the infinite bin
        prev_width = (
            df.loc[~inf_mask, "bin_width"].iloc[-1]
            if len(df) > 1
            else df["bin_left"].iloc[-1]
        )
        df.loc[inf_mask, "bin_width"] = prev_width
        df.loc[inf_mask, "bin_right"] = df["bin_left"].iloc[-1] + prev_width

    # Geometric mean for log bins
    df["bin_center"] = np.sqrt(df["bin_left"] * df["bin_right"])

    # --- Axis tick values (FIXED) ---
    # Generate nice tick values that align with the actual data range
    # Use powers of 2 or round numbers that make sense for the data
    log_min = np.log2(min_val)
    log_max = np.log2(max_val)

    # Generate approximately 5-8 tick marks across the range
    n_ticks = 6
    tick_positions = np.logspace(log_min, log_max, n_ticks, base=2)

    # Round to nice values
    nice_ticks = []
    for tick in tick_positions:
        if tick < 100:
            nice_ticks.append(int(round(tick)))
        elif tick < 1000:
            nice_ticks.append(int(round(tick, -1)))  # Round to nearest 10
        else:
            nice_ticks.append(int(round(tick, -2)))  # Round to nearest 100

    # Remove duplicates and sort
    x_vals = sorted(list(set(nice_ticks)))

    # Y-axis ticks
    y_max = df["count"].max()
    y_vals = list(range(0, y_max + 1, max(1, y_max // 5)))

    chart_width = 450

    # --- Altair chart (FIXED) ---
    chart = (
        alt.Chart(df, height=250, width=chart_width)
        .mark_bar(opacity=1, color="steelblue", width=(chart_width * 0.02))
        .encode(
            x=alt.X(
                "bin_center:Q",
                scale=alt.Scale(type="log"),
                title=column,
                axis=alt.Axis(
                    values=x_vals,  # Remove list() wrapper
                    labelExpr=f"datum.value >= 1000000 ? (datum.value/1000000) + 'M' : datum.value >= 1000 ? (datum.value/1000) + 'k' : datum.value",
                    tickCount=len(x_vals),
                    labelAngle=-45,
                    gridColor="white",
                ),
            ),
            y=alt.Y(
                "count:Q",
                title="Count",
                axis=alt.Axis(
                    values=y_vals,  # Remove list() wrapper
                    labelExpr=f"datum.value >= 1000000 ? (datum.value/1000000) + 'M' : datum.value >= 1000 ? (datum.value/1000) + 'k' : datum.value",
                    gridColor="white",
                ),
            ),
            tooltip=["bin_left", "bin_right", "count"],
        )
        .properties(title=f"Histogram of {chart_title}")
    )

    return chart.to_dict(format=settings.PLOT_BACKEND)
