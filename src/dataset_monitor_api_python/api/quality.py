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

    row_count = quality.check_row_count(lf)
    missing_values = quality.check_missing_values(lf)
    duplicate_values = quality.check_column_uniqueness(lf)
    encoding_issues = quality.check_encoding_issues(lf)

    quality_grade = quality.derive_quality_grade(
        row_count,
        duplicate_values["id"],
        duplicate_values["text"],
        missing_values,
        encoding_issues,
    )

    response = BasicCheckResponse(
        row_count=row_count,
        missing_value_count=missing_values,
        duplicate_ids=duplicate_values["id"],
        duplicate_texts=duplicate_values["text"],
        encoding_issues=encoding_issues,
        quality_grade=quality_grade,
    )

    return response
