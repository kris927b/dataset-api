from fastapi import APIRouter, HTTPException

from ..models.quality import BasicCheckResponse, BasicCheckRequest
from ..services import quality, analysis
from pathlib import Path

import polars as pl

router = APIRouter()


@router.post("/basic", response_model=BasicCheckResponse)
def basic_check(request: BasicCheckRequest):
    parquet_path = analysis.get_parquet_file_path(
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

    quality_grade = quality.derive_quality_grade(
        results["row_count"],
        results["column_uniqueness"]["id"],
        results["column_uniqueness"]["text"],
        results["missing_values"],
        results["encoding_issues"],
        results["token_outliers"],
        results["non_alpha_ratio"],
        results["repetition"],
        results["html_code_log"],
        results["lang_dist"]
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
        lang_dist=results["lang_dist"]
    )

    return response
