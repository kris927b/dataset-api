from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    ROOT_DIR: str = "data/datasets"
    BIND_ADDR: str = "0.0.0.0"
    BIND_PORT: int = 8080
    MAX_PARALLEL_SCANS: int = 4
    PLOT_BACKEND: Literal["vega", "vega-lite"] = "vega"
    CACHE_TTL_SECONDS: int = 600
    WATCH: bool = True
    RABBITMQ_URL: str = "amqp://guest:guest@localhost/"
    JOB_QUEUE_NAME: str = "textblaster_jobs"
    API_KEY: Optional[str] = None


settings = Settings()
