from fastapi import APIRouter

from ..models.quality import BasicCheckResponse, BasicCheckRequest
from ..services import quality, analysis
from ..core.cache import analysis_cache
import hashlib
import json

import polars as pl

router = APIRouter()


def compute_quality_request_hash(request: BasicCheckRequest) -> str:
    """Compute a hash for the quality check request to use as cache key."""
    request_dict = request.model_dump()
    request_str = json.dumps(request_dict, sort_keys=True)
    return hashlib.sha256(request_str.encode()).hexdigest()


@router.post("/basic", response_model=BasicCheckResponse)
async def basic_check(request: BasicCheckRequest):
    # Compute cache key
    cache_key = compute_quality_request_hash(request)

    # Check cache first
    cached_result = await analysis_cache.get(cache_key)
    if cached_result:
        return cached_result

    parquet_path = await analysis.get_parquet_file_path(
        request.dataset, request.variant, request.version
    )

    lf = pl.scan_parquet(parquet_path)

    # row_count = quality.check_row_count(lf)
    # missing_values = quality.check_missing_values(lf)
    # duplicate_values = quality.check_column_uniqueness(lf)
    # encoding_issues = quality.check_encoding_issues(lf)
    # token_outliers = quality.check_token_outliers(lf)
    # non_alpha_ratio = quality.check_non_alpha_ratio(lf)
    # repetition = quality.check_repetition(lf)
    # html_code_log = quality.check_html_or_code(lf)
    # lang_dist = quality.check_language_distribution(lf)

    results = quality.run_all_checks(lf)

    quality_score, quality_grade = quality.derive_quality_score_and_grade(
        results["row_count"],
        results["column_uniqueness"]["id"],
        results["column_uniqueness"]["text"],
        results["missing_values"],
        results["encoding_issues"],
        results["token_outliers"],
        results["non_alpha_ratio"],
        results["repetition"]["estimate_total"],
        results["html_code_log"],
        results["lang_dist"],
    )

    response = BasicCheckResponse(
        row_count=results["row_count"],
        missing_value_count=results["missing_values"],
        duplicate_ids=results["column_uniqueness"]["id"],
        duplicate_texts=results["column_uniqueness"]["text"],
        encoding_issues=results["encoding_issues"],
        quality_grade=quality_grade,
        token_outliers=results["token_outliers"],
        non_alpha_ratio=results["non_alpha_ratio"],
        repetition=results["repetition"],
        html_code_log=results["html_code_log"],
        lang_dist=results["lang_dist"],
    )

    # Cache the result
    await analysis_cache.set(cache_key, response)

    return response
