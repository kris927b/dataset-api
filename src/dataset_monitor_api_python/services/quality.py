import math
from multiprocessing.pool import Pool
import polars as pl
from collections import Counter
from typing import Any, Dict

from langdetect import detect

from fastapi.logger import logger


_REP_PATTERN = r"\b(\w+)(?:\s+\1){4,}\b"  # Python regex with backref


# Pool initializer to avoid recompiling regex in each process
def _init_rep(pat: str):
    global _REP_COMPILED
    import re

    _REP_COMPILED = re.compile(pat, flags=re.IGNORECASE)


def _check_rep(text: str) -> int:
    global _REP_COMPILED
    try:
        if not isinstance(text, str):
            return 0
        return 1 if _REP_COMPILED.search(text) else 0
    except Exception:
        return 0


def estimate_repetitions_by_sampling(
    texts: list[str],
    total_rows: int,
    sample_size: int = 10_000,
    workers: int = 8,
    seed: int | None = None,
) -> Dict[str, int | float | tuple[int, int]]:
    # parallel check using Pool (initializer compiles regex once per worker)
    with Pool(workers, initializer=_init_rep, initargs=(_REP_PATTERN,)) as p:
        counts = p.map(
            _check_rep, texts, chunksize=max(1, sample_size // (workers * 4))
        )

    k = sum(counts)
    phat = k / sample_size if sample_size > 0 else 0.0

    # Wilson 95% CI for proportion
    z = 1.96
    if sample_size == 0:
        lower = upper = 0.0
    else:
        denom = 1 + z * z / sample_size
        center = (phat + (z * z) / (2 * sample_size)) / denom
        margin = (
            z
            * math.sqrt((phat * (1 - phat) + (z * z) / (4 * sample_size)) / sample_size)
            / denom
        )
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

    estimate_total = phat * total_rows
    ci_total = (lower * total_rows, upper * total_rows)

    return {
        "sample_n": sample_size,
        "sample_count": k,
        "sample_prop": phat,
        "estimate_total": int(round(estimate_total)),
        "ci_total": (int(round(ci_total[0])), int(round(ci_total[1]))),
        "total_rows": total_rows,
    }


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def detect_languages_sample(texts: list[str], workers=8):
    with Pool(workers) as p:
        langs = p.map(detect_lang, texts)

    return dict(sorted(Counter(langs).items(), key=lambda x: x[1], reverse=True))


def run_all_checks(
    df: pl.LazyFrame,
    token_count_col: str = "token_count",
    text_column: str = "text",
    sample_size: int = 10_000,
    workers: int = 8,
) -> Dict[str, Any]:
    schema = df.collect_schema()
    all_columns = schema.names()

    # total rows
    logger.info("Getting number of rows.")
    total_rows = df.select(pl.len()).collect().item()

    # --------------------------
    # Pass 1: numeric stats only
    # --------------------------
    num_expr = [
        *(pl.col(col).null_count().alias(f"missing_{col}") for col in all_columns),
        *(pl.col(col).n_unique().alias(f"unique_{col}") for col in all_columns),
        pl.col(token_count_col).min().alias("min_tokens"),
        pl.col(token_count_col).max().alias("max_tokens"),
        pl.col(token_count_col).quantile(0.99).alias("p99_tokens"),
        (pl.col(token_count_col) < 5).sum().alias("too_short"),
        (pl.col(token_count_col) > 10_000).sum().alias("too_long"),
    ]

    logger.info("Collecting the numerical results.")
    num_results = df.select(num_expr).collect().to_dict(as_series=False)

    # reuse p99 value to avoid recomputation
    p99_value = num_results["p99_tokens"][0]
    above_p99 = (
        df.filter(pl.col(token_count_col) > p99_value).select(pl.len()).collect().item()
    )

    # --------------------------
    # Pass 2: string stats only
    # --------------------------
    # More specific code patterns
    code_patterns = [
        r"\b(function|def|class|import|from|return|void|int|string|bool)\s*[\(\{]",  # Function/class declarations
        r"[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*\{",  # Function calls with braces
        r"[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^=]",  # Variable assignments (avoiding ==)
        r"\b(if|for|while|else|elif|switch|case)\s*\(",  # Control structures with parentheses
        r"[;\}]\s*\n\s*[a-zA-Z_]",  # Statement endings followed by code
        r"//.*|/\*.*\*/|#.*",  # Comments
        r"\b(try|catch|finally|throw|except)\b",  # Exception handling
        r'["\'][^"\']*["\']\..*\(',  # Method chaining on strings
    ]

    str_expr = [
        pl.col(text_column).str.count_matches("�").sum().alias("replacement_char"),
        pl.col(text_column).str.count_matches(r"[ÃÂ][ -~]").sum().alias("mojibake"),
        pl.col(text_column)
        .str.count_matches(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
        .sum()
        .alias("control_chars"),
        pl.col(text_column).str.contains(r"<[^>]+>").sum().alias("html_like"),
        pl.col(text_column)
        .str.contains(f"({'|'.join(code_patterns)})")
        .sum()
        .alias("code_like"),
        pl.col(text_column)
        .str.contains(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
        .sum()
        .alias("log_like"),
        (
            pl.col(text_column).str.count_matches(r"[^A-Za-zÆØÅæøå]")
            / pl.col(text_column).str.len_chars()
        )
        .mean()
        .alias("avg_non_alpha_ratio"),
        # regex version of repetition check
        # pl.col(text_column)
        # .str.contains(r"\b(\w+)(\s+\1){4,}\b")
        # .sum()
        # .alias("repetitions"),
    ]
    logger.info("Collecting the string results.")
    str_results = df.select(str_expr).collect().to_dict(as_series=False)

    # --------------------------
    # Language detection (outside Polars)
    # --------------------------
    n = min(total_rows, sample_size)
    texts = (
        df.select(pl.col(text_column).sample(n=n, shuffle=True))
        .collect()
        .to_series()
        .to_list()
    )

    logger.info("Detecting language distribution")
    lang_dist = detect_languages_sample(texts, workers=workers)

    logger.info("Estimating the repitions")
    repetitions_estimate = estimate_repetitions_by_sampling(
        texts, total_rows, sample_size=n, workers=workers
    )

    # --------------------------
    # Final result assembly
    # --------------------------
    output = {
        "row_count": total_rows,
        "missing_values": {
            col: int(num_results[f"missing_{col}"][0]) for col in all_columns
        },
        "column_uniqueness": {
            "id": total_rows - int(num_results.get("unique_id", [0])[0])
            if "id" in all_columns
            else 0,
            "text": total_rows - int(num_results["unique_text"][0]),
        },
        "encoding_issues": {
            "replacement_char": int(str_results["replacement_char"][0]),
            "mojibake": int(str_results["mojibake"][0]),
            "control_chars": int(str_results["control_chars"][0]),
        },
        "non_alpha_ratio": float(str_results["avg_non_alpha_ratio"][0] or 0.0),
        "html_code_log": {
            "html_like": int(str_results["html_like"][0]),
            "code_like": int(str_results["code_like"][0]),
            "log_like": int(str_results["log_like"][0]),
        },
        "token_outliers": {
            "min_tokens": int(num_results["min_tokens"][0] or 0),
            "max_tokens": int(num_results["max_tokens"][0] or 0),
            "p99_tokens": int(p99_value or 0),
            "too_short": int(num_results["too_short"][0]),
            "too_long": int(num_results["too_long"][0]),
            "above_p99": int(above_p99),
        },
        "repetition": repetitions_estimate,
        "lang_dist": lang_dist,
    }

    return output


def derive_quality_grade(
    total_rows: int,
    duplicate_ids: int,
    duplicate_texts: int,
    missing_value_count: dict[str, int],
    encoding_issues: dict[str, int],
    token_outliers: dict[str, int],
    non_alpha_ratio: float,
    repetition: int,  # estimated total documents with repeating words
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
    elif (
        token_outliers.get("too_short", 0) > 0 or token_outliers.get("too_long", 0) > 0
    ):
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
    if len(lang_dist) > 1:  #  and max(lang_dist.values()) < 0.80:
        # No dominant language, probably mixed content
        grade = "Fair"

    return grade
