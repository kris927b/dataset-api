from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class DatasetFile(BaseModel):
    path: str
    size_bytes: int
    modified_at: str


class ColumnSchema(BaseModel):
    name: str
    dtype: str
    nullable: bool


class DatasetVersion(BaseModel):
    version: str
    path: str
    file: DatasetFile
    file_schema: List[ColumnSchema]
    row_count: int


class DatasetVariant(BaseModel):
    variant: str
    versions: List[DatasetVersion]


class Dataset(BaseModel):
    slug: str
    path: str
    variants: List[DatasetVariant]


class TextBlasterJobRequest(BaseModel):
    input_file: str
    output_file: str
    excluded_file: str
    text_column: str
    id_column: str
