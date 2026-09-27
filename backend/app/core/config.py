from functools import lru_cache
from pathlib import Path

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Module-relative, not cwd-relative: the container runs uvicorn from /workspace/backend
# and a developer runs it from the checkout root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Тормоза API"
    app_env: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+asyncpg://tramflow:tramflow_local@localhost:5432/tramflow"
    backend_cors_origins: list[AnyHttpUrl] = Field(
        default_factory=lambda: [
            AnyHttpUrl("http://localhost:5173"),
            AnyHttpUrl("http://localhost:8080"),
        ]
    )

    # See docs/decisions/0003-tram-graph-file-repository.md.
    forecast_geometry_mapping: Path | None = None
    tram_graph_json: Path = DATA_DIR / "tram_graph.json"
    tram_graph_geojson: Path = DATA_DIR / "tram_graph.geojson"

    # Self-hosted Overpass instance; HTTP only, no auth (docs/overpass-api.md).
    # The base, not the interpreter: /interpreter and /status are siblings under it,
    # so deriving both from one value leaves no way for them to disagree.
    overpass_base_url: str = "http://2.29.6.201/api"
    overpass_timeout: float = Field(default=60.0, gt=0)
    overpass_health_timeout: float = Field(default=3.0, gt=0)
    overpass_max_query_chars: int = Field(default=8000, gt=0)
    overpass_query_enabled: bool = False

    @property
    def overpass_interpreter_url(self) -> str:
        return f"{self.overpass_base_url.rstrip('/')}/interpreter"

    @property
    def overpass_status_url(self) -> str:
        return f"{self.overpass_base_url.rstrip('/')}/status"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
