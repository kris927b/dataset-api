from pydantic import BaseModel


class BasicCheckResponse(BaseModel):
    row_count: int
    missing_value_count: dict[str, float]
    duplicate_ids: float
    duplicate_texts: float
    encoding_issues: dict[str, int]
    quality_grade: str


class BasicCheckRequest(BaseModel):
    dataset: str
    variant: str
    version: str
