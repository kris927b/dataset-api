import polars as pl
from collections import Counter
from typing import Any, Dict
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

pattern = re.compile(r"\b(\w+)\s+\1{4,}\b")

def repeated_word(text: str) -> int:
    return 1 if pattern.search(text) else 0
# -------------------
# Phase 1 Checks
# -------------------


def check_row_count(df: pl.LazyFrame) -> int:
    """
    Count total rows in dataset.
    """
    return df.select(pl.count()).collect(engine="streaming").item()


def check_missing_values(df: pl.LazyFrame) -> Dict[str, int]:
    """
    Compute missing value fraction per column.
    Returns {column_name: fraction_missing}.
    """
    out = {}
    n_rows = df.select(pl.count()).collect(engine="streaming").item()

    for col in df.collect_schema().names():
        nulls = df.select(pl.col(col).null_count()).collect(engine="streaming").item()
        out[col] = nulls
    return out


def check_duplicate_rows(df: pl.LazyFrame) -> int:
    """
    Count duplicate rows in dataset.
    """
    n_total = df.select(pl.count()).collect(engine="streaming").item()
    n_unique = df.unique().select(pl.count()).collect(engine="streaming").item()
    return n_total - n_unique


def check_column_uniqueness(df: pl.LazyFrame) -> Dict[str, int]:
    """
    For each column, compute ratio of unique values / total rows.
    Returns {column_name: uniqueness_ratio}.
    """
    out = {}
    n_rows = df.select(pl.count()).collect(engine="streaming").item()

    for col in df.collect_schema().names():
        n_unique = df.select(pl.col(col).n_unique()).collect(engine="streaming").item()
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
    out = df.select(exprs).collect(engine="streaming").to_dict(as_series=False)

    # Restructure into nested dict
    results = {
        "replacement_char": int(out[f"replacement_char"][0]),
        "mojibake": int(out[f"mojibake"][0]),
        "control_chars": int(out[f"control_chars"][0]),
    }

    return results

# -----------------------------
# Outlier Detection
# -----------------------------

def check_token_outliers(df: pl.LazyFrame, token_count_col: str = "token_count") -> Dict[str, int]:
    """
    Find rows with very low or very high token counts (based on whitespace split).
    Returns thresholds + counts.
    """
    # tokens = df.select(pl.col(text_col).str.count_matches(r"\S+").alias("n_tokens"))

    stats = df.select(
        pl.col(token_count_col).min().alias("min"),
        pl.col(token_count_col).max().alias("max"),
        pl.col(token_count_col).quantile(0.99).alias("p99"),
    ).collect(engine="streaming").to_dict(as_series=False)

    min_len = int(stats["min"][0])
    max_len = int(stats["max"][0])
    p99 = int(stats["p99"][0])

    # Count extreme rows
    counts = df.select([
        (pl.col(token_count_col) < 5).sum().alias("too_short"),
        (pl.col(token_count_col) > 10_000).sum().alias("too_long"),
        (pl.col(token_count_col) > p99).sum().alias("above_p99"),
    ]).collect(engine="streaming").to_dict(as_series=False)

    return {
        "min_tokens": min_len,
        "max_tokens": max_len,
        "p99_tokens": p99,
        "too_short": int(counts["too_short"][0]),
        "too_long": int(counts["too_long"][0]),
        "above_p99": int(counts["above_p99"][0]),
    }


# -----------------------------
# Noise Checks
# -----------------------------

def check_non_alpha_ratio(df: pl.LazyFrame, text_col: str = "text") -> float:
    """
    Average proportion of non-alphabetic characters in text.
    """
    expr = (
        (pl.col(text_col).str.count_matches(r"[^A-Za-zÆØÅæøå]") / pl.col(text_col).str.len_chars())
        .mean()
        .alias("avg_non_alpha_ratio")
    )
    return float(df.select(expr).collect(engine="streaming").item())


def check_repetition(df: pl.LazyFrame, text_col: str = "text") -> int:
    """
    Detect rows with excessive repetition (same word >50 times).
    Returns count of suspicious rows.
    """
    expr = pl.col(text_col).map_elements(repeated_word, return_dtype=pl.Int64).sum().alias("repetitions")
    count = df.select(expr).collect(engine="streaming").item()
    return int(count)


def check_html_or_code(df: pl.LazyFrame, text_col: str = "text") -> Dict[str, int]:
    """
    Heuristic: detect rows that look like HTML, code, or logs.
    """
    exprs = [
        pl.col(text_col).str.contains(r"<[^>]+>").sum().alias("html_like"),
        pl.col(text_col)
            .str.contains(r"[;{}]|\bif\b|\bfor\b")
            .sum()
            .alias("code_like"),
        pl.col(text_col).str.contains(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").sum().alias("log_like"),
    ]
    out = df.select(exprs).collect(engine="streaming").to_dict(as_series=False)
    return {k: int(v[0]) for k, v in out.items()}


# -----------------------------
# Per-column Insights
# -----------------------------

def check_column_cardinality(df: pl.LazyFrame) -> Dict[str, Any]:
    """
    Compute cardinality (n_unique / total) per column.
    """
    n_rows = df.select(pl.count()).collect(engine="streaming").item()
    results = {}
    for col in df.columns:
        n_unique = df.select(pl.col(col).n_unique()).collect(engine="streaming").item()
        results[col] = {
            "n_unique": int(n_unique),
            "cardinality_ratio": n_unique / max(n_rows, 1),
        }
    return results


# -----------------------------
# Language Distribution
# -----------------------------

def check_language_distribution(df: pl.LazyFrame, text_col: str = "text", sample_size: int = 10000) -> Dict[str, int]:
    """
    Sample rows and run langdetect. 
    NOTE: this requires langdetect (or fasttext) installed.
    """
    try:
        from langdetect import detect
    except ImportError:
        raise RuntimeError("Please install `langdetect` to use this check.")

    # Sample data (materialize only small subset!)
    sample = df.select(pl.col(text_col)).filter(pl.int_range(pl.len()).shuffle() < sample_size).collect(engine="streaming").to_series()

    langs = []
    for txt in sample:
        try:
            langs.append(detect(txt))
        except Exception:
            langs.append("unknown")

    return dict(Counter(langs))



def derive_quality_grade(
    total_rows: int,
    duplicate_ids: int,
    duplicate_texts: int,
    missing_value_count: dict[str, int],
    encoding_issues: dict[str, int],
    token_outliers: dict[str, int],
    non_alpha_ratio: float,
    repetition: int,
    html_code_log: dict[str, int],
    lang_dist: dict[str, int],
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

    # --- token outliers ---
    if token_outliers.get("above_p99", 0) / max(total_rows, 1) > 0.02:
        return "Needs Attention"
    elif token_outliers.get("too_short", 0) > 0 or token_outliers.get("too_long", 0) > 0:
        grade = "Fair"

    # --- non-alpha ratio ---
    if non_alpha_ratio > 0.50:  # more than half of chars are non-alpha
        return "Needs Attention"
    elif non_alpha_ratio > 0.20:
        grade = "Fair"

    # --- repetition ---
    if repetition / max(total_rows, 1) > 0.02:
        return "Needs Attention"
    elif repetition > 0:
        grade = "Fair"

    # --- html/code presence ---
    html_total = sum(html_code_log.values())
    if html_total / max(total_rows, 1) > 0.05:
        return "Needs Attention"
    elif html_total > 0:
        grade = "Fair"

    # --- language distribution ---
    if len(lang_dist) > 1: #  and max(lang_dist.values()) < 0.80:
        # No dominant language, probably mixed content
        grade = "Fair"

    return grade