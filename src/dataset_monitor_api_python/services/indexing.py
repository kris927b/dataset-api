import os
import polars as pl
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..core.config import settings
from ..models.domain import (
    Dataset,
    DatasetVariant,
    DatasetVersion,
    DatasetFile,
    ColumnSchema,
)

# Module-level cache
_cache: List[Dataset] = []


def scan_dataset(dataset_path: Path) -> Dataset:
    slug = dataset_path.name
    variant_folders = [p for p in dataset_path.iterdir() if p.is_dir()]

    all_variants = []

    for variant_path in variant_folders:
        variant_name = variant_path.name
        version_folders = [p for p in variant_path.iterdir() if p.is_dir()]

        all_versions = []

        for version_path in version_folders:
            version_name = version_path.name
            try:
                parquet_file = next(version_path.glob("*.parquet"))

                # Create DatasetFile
                file_stat = parquet_file.stat()
                dataset_file = DatasetFile(
                    path=str(parquet_file),
                    size_bytes=file_stat.st_size,
                    modified_at=datetime.fromtimestamp(
                        file_stat.st_mtime
                    ).isoformat(),
                )

                # Get Schema and Row Count from metadata (fast)
                dataset = pl.scan_parquet(parquet_file)
                arrow_schema: pl.Schema = dataset.collect_schema()
                row_count = dataset.collect().height

                file_schema = [
                    ColumnSchema(
                        name=field,  # field.name,
                        dtype=arrow_schema[field].__str__(),  # str(field.type),
                        nullable=False,  # field.nullable,
                    )
                    for field in arrow_schema.names()
                ]

                # Create DatasetVersion
                dataset_version = DatasetVersion(
                    version=version_name,
                    path=str(version_path),
                    file=dataset_file,
                    file_schema=file_schema,
                    row_count=row_count,
                )
                all_versions.append(dataset_version)

            except StopIteration:
                # No parquet file found, skip this version directory
                continue

        if all_versions:
            all_versions.sort(key=lambda v: v.version, reverse=True)
            dataset_variant = DatasetVariant(
                variant=variant_name,
                versions=all_versions,
            )
            all_variants.append(dataset_variant)

    return Dataset(
        slug=slug,
        path=str(dataset_path),
        variants=all_variants,
    )


def perform_scan() -> List[Dataset]:
    """Performs the actual filesystem scan."""
    root_path = Path(settings.ROOT_DIR)
    if not root_path.exists():
        return []

    datasets = []
    for dataset_dir in root_path.iterdir():
        if dataset_dir.is_dir():
            datasets.append(scan_dataset(dataset_dir))

    return datasets


def populate_cache():
    """Scans the filesystem and populates the in-memory cache."""
    global _cache
    _cache = perform_scan()


def get_all_datasets() -> List[Dataset]:
    """Returns all datasets from the cache."""
    return _cache


def get_dataset_by_slug(slug: str) -> Optional[Dataset]:
    """Finds a dataset by its slug in the cache."""
    for dataset in _cache:
        if dataset.slug == slug:
            return dataset
    return None