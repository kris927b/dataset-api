import polars as pl
import altair as alt
from typing import Dict, Any, List
from pathlib import Path
from collections import defaultdict

from ..models.analysis import AnalysisRequest, ColumnStats
from ..services import indexing


def get_parquet_file_path(dataset: str, variant: str, version: str) -> Path:
    ds = indexing.get_dataset_by_slug(dataset)
    if not ds:
        raise ValueError("Dataset not found")

    # This is a simplification. A more robust solution would check the files list.
    base_path = Path(ds.path)
    return base_path / variant / version / f"{ds.slug}.parquet"


def run_analysis_on_file(
    file_path: Path, operations: List[Dict[str, Any]]
) -> Dict[str, ColumnStats]:
    lf = pl.scan_parquet(file_path)
    results = {}

    for op in operations:
        op_type = op.get("op")
        column = op.get("column")

        if op_type == "row_count":
            # Row count is a special case - not tied to a specific column
            # You might want to handle this differently based on your needs
            results["_row_count"] = ColumnStats(sum=lf.collect().height)

        elif column and op_type in ["sum", "mean", "min", "max", "distinct_count"]:
            # Initialize ColumnStats for this column if it doesn't exist
            if column not in results:
                results[column] = ColumnStats()

            # Update the specific stat for this column
            if op_type == "sum":
                results[column].sum = lf.select(pl.col(column).sum()).collect().item()
            elif op_type == "mean":
                results[column].mean = lf.select(pl.col(column).mean()).collect().item()
            elif op_type == "min":
                results[column].min = lf.select(pl.col(column).min()).collect().item()
            elif op_type == "max":
                results[column].max = lf.select(pl.col(column).max()).collect().item()
            elif op_type == "distinct_count":
                results[column].distinct_count = (
                    lf.select(pl.col(column).n_unique()).collect().item()
                )

    return results


def generate_plot_spec(file_path: Path, operation: Dict[str, Any]) -> Dict[str, Any]:
    df = pl.read_parquet(file_path)

    op_type = operation.get("op")
    column = operation.get("column")

    if not column:
        raise ValueError("Plot operation must specify a 'column'")

    chart_title = f"'{column}' in {file_path.parent.parent.parent.name}"

    if op_type == "histogram":
        bins = operation.get("bins", 20)
        chart = (
            alt.Chart(df.to_pandas())
            .mark_bar()
            .encode(
                alt.X(column, bin=alt.Bin(maxbins=bins), title=column),
                alt.Y("count()", title="Count of records"),
            )
            .properties(title=f"Histogram of {chart_title}")
        )
    elif op_type == "boxplot":
        chart = (
            alt.Chart(df.to_pandas())
            .mark_boxplot()
            .encode(alt.Y(column, title=column))
            .properties(title=f"Boxplot of {chart_title}")
        )
    else:
        raise ValueError(f"Unsupported plot operation: {op_type}")

    return chart.to_dict()
