from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ColumnStats(BaseModel):
    sum: Optional[int] = None
    mean: Optional[float] = None
    min: Optional[int] = None
    max: Optional[int] = None
    distinct_count: Optional[int] = None


class AnalysisPreviewRequest(BaseModel):
    dataset: str
    variant: str
    version: str


class AnalysisPreviewResult(BaseModel):
    id: Optional[str]
    source: Optional[str]
    added: Optional[str]
    created: Optional[str]
    text: Optional[str]
    token_count: Optional[int]
    metadata: Optional[str] = None


class AnalysisResult(BaseModel):
    columns: Dict[str, ColumnStats]
    plots: Optional[List[Dict[str, Any]]] = None


class AnalysisRequest(BaseModel):
    dataset: str
    variant: str = "original"
    version: str
    operations: List[Dict[str, Any]]
    filters: Optional[List[Dict[str, Any]]] = None
