from fastapi import APIRouter, HTTPException
from ..models.analysis import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisPreviewResult,
    AnalysisPreviewRequest,
)
from ..services import analysis
from pathlib import Path

import polars as pl

router = APIRouter()


@router.post("/run", response_model=AnalysisResult)
def run_analysis_endpoint(request: AnalysisRequest):
    try:
        file_path = analysis.get_parquet_file_path(
            request.dataset, request.variant, request.version
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Parquet file not found")

        stats = analysis.run_analysis_on_file(file_path, request.operations)
        return AnalysisResult(columns=stats)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@router.post("/preview", response_model=list[AnalysisPreviewResult])
def run_preview_endpoint(request: AnalysisPreviewRequest):
    try:
        file_path = analysis.get_parquet_file_path(
            request.dataset, request.variant, request.version
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Parquet file not found")

        lf = pl.scan_parquet(file_path)
        samples = lf.filter(pl.int_range(pl.len()).shuffle() < 5).collect().to_dicts()

        preview = []

        for sample in samples:
            if isinstance(sample["created"], dict):
                sample["created"] = (
                    f"{sample['created']['start']}, {sample['created']['end']}"
                )
            p = AnalysisPreviewResult(
                id=sample["id"],
                source=sample["source"],
                added=sample["added"],
                created=sample["created"],
                text=sample["text"],
                token_count=sample["token_count"],
                metadata=sample["metadata"] if "metadata" in sample else None,
            )
            preview.append(p)

        return preview

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
