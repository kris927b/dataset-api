import polars as pl
from typing import Any, Dict


# -------------------
# Phase 1 Checks
# -------------------


def check_row_count(df: pl.LazyFrame) -> int:
    """
    Count total rows in dataset.
    """
    return df.select(pl.count()).collect().item()


def check_missing_values(df: pl.LazyFrame) -> Dict[str, float]:
    """
    Compute missing value fraction per column.
    Returns {column_name: fraction_missing}.
    """
    out = {}
    n_rows = df.select(pl.count()).collect().item()

    for col in df.collect_schema().names():
        nulls = df.select(pl.col(col).null_count()).collect().item()
        out[col] = nulls
    return out


def check_duplicate_rows(df: pl.LazyFrame) -> int:
    """
    Count duplicate rows in dataset.
    """
    n_total = df.select(pl.count()).collect().item()
    n_unique = df.unique().select(pl.count()).collect().item()
    return n_total - n_unique


def check_column_uniqueness(df: pl.LazyFrame) -> Dict[str, int]:
    """
    For each column, compute ratio of unique values / total rows.
    Returns {column_name: uniqueness_ratio}.
    """
    out = {}
    n_rows = df.select(pl.count()).collect().item()

    for col in df.collect_schema().names():
        n_unique = df.select(pl.col(col).n_unique()).collect().item()
        out[col] = n_rows - n_unique
    return out


def check_encoding_issues(df: pl.LazyFrame) -> Dict[str, int]:
    """
    Lazily count likely encoding issues per text column.
    Returns {col: {"replacement_char": int, "mojibake": int, "control_chars": int}}
    """
    col = "text"
    exprs = [
        pl.col(col).str.count_matches("�").sum().alias(f"replacement_char"),
        pl.col(col).str.count_matches(r"[ÃÂ][ -~]").sum().alias(f"mojibake"),
        pl.col(col)
        .str.count_matches(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
        .sum()
        .alias(f"control_chars"),
    ]

    # Collect only aggregated counts → very cheap compared to pulling all text
    out = df.select(exprs).collect().to_dict(as_series=False)

    # Restructure into nested dict
    results = {
        "replacement_char": int(out[f"replacement_char"][0]),
        "mojibake": int(out[f"mojibake"][0]),
        "control_chars": int(out[f"control_chars"][0]),
    }

    return results


def derive_quality_grade(
    total_rows: int,
    duplicate_ids: int,
    duplicate_texts: int,
    missing_value_count: dict[str, float],
    encoding_issues: dict[str, int],
) -> str:
    grade = "Excellent"

    # --- duplicates ---
    dup_bad = duplicate_ids + duplicate_texts
    if dup_bad / max(total_rows, 1) > 0.01:
        return "Needs Attention"
    elif dup_bad > 0:
        grade = "Fair"

    # --- missing values ---
    for col, cnt in missing_value_count.items():
        frac = cnt / max(total_rows, 1)
        if frac > 0.10:
            return "Needs Attention"
        elif frac > 0.01:
            grade = "Fair"

    # --- encoding issues ---
    encoding_total = sum(encoding_issues.values())
    if encoding_total > 100:
        return "Needs Attention"
    elif encoding_total > 0:
        grade = "Fair"

    return grade
