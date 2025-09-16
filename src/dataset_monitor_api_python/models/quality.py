from pydantic import BaseModel


class BasicCheckResponse(BaseModel):
    row_count: int
    missing_value_count: dict[str, int]
    duplicate_ids: float
    duplicate_texts: float
    encoding_issues: dict[str, int]
    quality_grade: str
    token_outliers: dict[str, int]
    non_alpha_ratio: float
    repetition: int
    html_code_log: dict[str, int]
    lang_dist: dict[str, int]


class BasicCheckRequest(BaseModel):
    dataset: str
    variant: str
    version: str
