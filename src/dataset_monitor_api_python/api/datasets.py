from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from ..services import indexing
from ..models.domain import Dataset, DatasetVersion

router = APIRouter()

@router.get("/", response_model=List[Dataset])
def list_datasets():
    return indexing.get_all_datasets()

@router.get("/{slug}", response_model=Dataset)
def get_dataset(slug: str):
    dataset = indexing.get_dataset_by_slug(slug)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {slug} not found")
    return dataset

@router.get("/{slug}/{variant}", response_model=List[DatasetVersion])
def get_variant(slug: str, variant: str):
    dataset = indexing.get_dataset_by_slug(slug)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {slug} not found")
    variants = [var.variant for var in dataset.variants]
    if variant not in variants:
        raise HTTPException(status_code=404, detail=f"Variant {variant} not found in {dataset}")
    # This is a simplified response. The README implies just versions.
    dataset_variant = [var for var in dataset.variants if var.variant == variant][0]
    return dataset_variant.versions


@router.get("/{slug}/{variant}/{version}", response_model=DatasetVersion)
def get_version(slug: str, variant: str, version: str):
    dataset = indexing.get_dataset_by_slug(slug)
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset {slug} not found")
    
    variants = [var.variant for var in dataset.variants]
    if variant not in variants:
        raise HTTPException(status_code=404, detail=f"Variant {variant} not found in {dataset}")

    dataset_variant = [var for var in dataset.variants if var.variant == variant][0]
    versions = [ver.version for ver in dataset_variant.versions]

    if version not in versions:
        raise HTTPException(status_code=404, detail=f"Version not found in {dataset}/{variant}")
    
    # Find the specific file and schema info for this version
    dataset_version = [ver for ver in dataset_variant.versions if ver.version == version][0]
            
    return dataset_version
