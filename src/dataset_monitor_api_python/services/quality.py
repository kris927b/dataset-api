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


def derive_quality_score_and_grade(
    total_rows: int,
    duplicate_ids: int,
    duplicate_texts: int,
    missing_value_count: Dict[str, int],
    encoding_issues: Dict[str, int],
    token_outliers: Dict[str, int],
    non_alpha_ratio: float,
    repetition: int,
    html_code_log: Dict[str, int],
    lang_dist: Dict[str, int],
) -> tuple[float, str]:
    """
    Calculate both a numeric quality score (0-100) and a grade.

    Returns:
        Tuple of (score, grade) where score is 0-100 and grade is one of:
        "Excellent", "Good", "Fair", "Poor", "Needs Attention"
    """
    if total_rows == 0:
        return 0.0, "Needs Attention"

    # Initialize score at 100 and apply penalties
    score = 100.0

    # --- Duplicates (weight: high) ---
    dup_bad = duplicate_ids + duplicate_texts
    dup_rate = dup_bad / total_rows
    if dup_rate > 0.05:  # More than 5%
        score -= 30
    elif dup_rate > 0.01:  # More than 1%
        score -= 15
    elif dup_rate > 0:
        score -= 5

    # --- Missing Values (weight: medium-high) ---
    max_missing_rate = 0
    total_missing_penalty = 0
    for col, cnt in missing_value_count.items():
        missing_rate = cnt / total_rows
        max_missing_rate = max(max_missing_rate, missing_rate)
        if missing_rate > 0.20:  # More than 20%
            total_missing_penalty += 20
        elif missing_rate > 0.10:  # More than 10%
            total_missing_penalty += 10
        elif missing_rate > 0.01:  # More than 1%
            total_missing_penalty += 3

    score -= min(total_missing_penalty, 25)  # Cap missing value penalty

    # --- Encoding Issues (weight: medium) ---
    encoding_total = sum(encoding_issues.values())
    encoding_rate = encoding_total / total_rows
    if encoding_rate > 0.10:  # More than 10%
        score -= 20
    elif encoding_rate > 0.05:  # More than 5%
        score -= 10
    elif encoding_total > 0:
        score -= min(encoding_total / 10, 8)  # Gradual penalty up to 8 points

    # --- Token Outliers (weight: medium) ---
    outlier_penalty = 0
    above_p99_rate = token_outliers.get("above_p99", 0) / total_rows
    if above_p99_rate > 0.05:
        outlier_penalty += 15
    elif above_p99_rate > 0.02:
        outlier_penalty += 8
    elif above_p99_rate > 0:
        outlier_penalty += 3

    too_short = token_outliers.get("too_short", 0)
    too_long = token_outliers.get("too_long", 0)
    extreme_outlier_rate = (too_short + too_long) / total_rows
    if extreme_outlier_rate > 0.05:
        outlier_penalty += 10
    elif extreme_outlier_rate > 0:
        outlier_penalty += 5

    score -= outlier_penalty

    # --- Non-alpha Ratio (weight: medium) ---
    if non_alpha_ratio > 0.70:
        score -= 20
    elif non_alpha_ratio > 0.50:
        score -= 15
    elif non_alpha_ratio > 0.30:
        score -= 8
    elif non_alpha_ratio > 0.20:
        score -= 3

    # --- Repetition (weight: medium) ---
    repetition_rate = repetition / total_rows
    if repetition_rate > 0.10:
        score -= 15
    elif repetition_rate > 0.05:
        score -= 10
    elif repetition_rate > 0.02:
        score -= 5
    elif repetition > 0:
        score -= 2

    # --- HTML/Code Presence (weight: low-medium) ---
    html_total = sum(html_code_log.values())
    html_rate = html_total / total_rows
    if html_rate > 0.20:
        score -= 15
    elif html_rate > 0.10:
        score -= 10
    elif html_rate > 0.05:
        score -= 5
    elif html_total > 0:
        score -= 2

    # --- Language Distribution (weight: low) ---
    if len(lang_dist) > 1:
        total_lang_docs = sum(lang_dist.values())
        if total_lang_docs > 0:
            # Calculate entropy-based penalty for language diversity
            entropy = 0
            for count in lang_dist.values():
                if count > 0:
                    p = count / total_lang_docs
                    entropy -= p * math.log2(p)

            # High entropy (many languages) gets more penalty
            max_entropy = math.log2(len(lang_dist))
            if max_entropy > 0:
                diversity_ratio = entropy / max_entropy
                if diversity_ratio > 0.8:  # Very mixed languages
                    score -= 8
                elif diversity_ratio > 0.6:
                    score -= 5
                elif diversity_ratio > 0.4:
                    score -= 3

    # Ensure score doesn't go below 0
    score = max(0.0, score)

    # Determine grade based on final score
    if score >= 90:
        grade = "Excellent"
    elif score >= 80:
        grade = "Good"
    elif score >= 65:
        grade = "Fair"
    elif score >= 40:
        grade = "Poor"
    else:
        grade = "Needs Attention"

    return round(score, 1), grade


def get_quality_insights(
    score: float,
    total_rows: int,
    duplicate_ids: int,
    duplicate_texts: int,
    missing_value_count: Dict[str, int],
    encoding_issues: Dict[str, int],
    token_outliers: Dict[str, int],
    non_alpha_ratio: float,
    repetition: int,
    html_code_log: Dict[str, int],
    lang_dist: Dict[str, int],
) -> Dict[str, Any]:
    """
    Provide detailed insights about what's affecting the quality score.
    """
    insights = {
        "score": score,
        "total_rows": total_rows,
        "issues": [],
        "strengths": [],
        "recommendations": [],
    }

    if total_rows == 0:
        insights["issues"].append("No data to analyze")
        return insights

    # Analyze each dimension
    dup_rate = (duplicate_ids + duplicate_texts) / total_rows
    if dup_rate > 0.01:
        insights["issues"].append(
            f"Duplicates: {dup_rate:.1%} of records are duplicated"
        )
        insights["recommendations"].append(
            "Remove duplicate records to improve data quality"
        )
    elif dup_rate == 0:
        insights["strengths"].append("No duplicate records found")

    max_missing_rate = max(
        [cnt / total_rows for cnt in missing_value_count.values()], default=0
    )
    if max_missing_rate > 0.10:
        insights["issues"].append(
            f"Missing values: Up to {max_missing_rate:.1%} missing in some columns"
        )
        insights["recommendations"].append(
            "Address missing values through imputation or data collection"
        )
    elif max_missing_rate == 0:
        insights["strengths"].append("No missing values detected")

    encoding_rate = sum(encoding_issues.values()) / total_rows
    if encoding_rate > 0.05:
        insights["issues"].append(
            f"Encoding issues: {encoding_rate:.1%} of records have encoding problems"
        )
        insights["recommendations"].append("Fix character encoding issues")

    if non_alpha_ratio > 0.30:
        insights["issues"].append(
            f"Text quality: {non_alpha_ratio:.1%} non-alphabetic characters"
        )
        insights["recommendations"].append("Review and clean text content")
    elif non_alpha_ratio < 0.10:
        insights["strengths"].append("Good text quality with mostly alphabetic content")

    repetition_rate = repetition / total_rows
    if repetition_rate > 0.02:
        insights["issues"].append(
            f"Repetitive content: {repetition_rate:.1%} of documents have repetitive text"
        )
        insights["recommendations"].append("Review and deduplicate repetitive content")

    if len(lang_dist) > 2:
        insights["issues"].append(
            f"Language consistency: {len(lang_dist)} different languages detected"
        )
        insights["recommendations"].append(
            "Consider filtering or separating by language"
        )
    elif len(lang_dist) == 1:
        insights["strengths"].append("Consistent language throughout dataset")

    return insights
