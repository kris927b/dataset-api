import json
from fastapi import APIRouter, HTTPException
from ..models.analysis import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisPreviewResult,
    AnalysisPreviewRequest,
)
from ..services import analysis
from ..core.cache import analysis_cache
from pathlib import Path

import polars as pl

router = APIRouter()


@router.post("/run", response_model=AnalysisResult)
async def run_analysis_endpoint(request: AnalysisRequest):
    try:
        # Compute cache key
        cache_key = analysis.compute_request_hash(request)

        # Check cache first
        cached_result = await analysis_cache.get(cache_key)
        if cached_result:
            return cached_result

        file_path = await analysis.get_parquet_file_path(
            request.dataset, request.variant, request.version
        )
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Parquet file not found")

        stats = analysis.run_analysis_on_file(file_path, request.operations)
        result = AnalysisResult(columns=stats)

        # Cache the result
        await analysis_cache.set(cache_key, result)

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@router.get("/cache/{cache_key}", response_model=AnalysisResult)
async def get_cached_analysis(cache_key: str):
    """Fetch a cached analysis result by its key."""
    cached_result = await analysis_cache.get(cache_key)
    if cached_result is None:
        raise HTTPException(status_code=404, detail="Cached result not found or expired")
    return cached_result


@router.post("/preview", response_model=list[AnalysisPreviewResult])
async def run_preview_endpoint(request: AnalysisPreviewRequest):
    try:
        file_path = await analysis.get_parquet_file_path(
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
                metadata=json.dumps(sample["metadata"]) if "metadata" in sample else None,
            )
            preview.append(p)

        return preview

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )
