"""
Central config. Reads from environment variables / Replit Secrets.
Never hardcode the TMDB key anywhere else in the codebase.

.env loading is robust: we locate the .env file relative to this file's
directory (backend/.env) and fall back to the current working directory, so
the same config works whether uvicorn/pytest is launched from backend/ or
from the repo root. Real env vars / Replit Secrets always win over the file.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str | None:
    """Locate the .env file robustly regardless of the current working dir.

    Priority: backend/.env (next to this module), then ./env in the CWD.
    Returns None if neither exists, in which case BaseSettings falls back to
    real process environment variables (e.g. Replit Secrets).
    """
    here = Path(__file__).resolve().parent.parent  # backend/
    candidates = [here / ".env", Path.cwd() / ".env"]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return None


class Settings(BaseSettings):
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    tmdb_watch_provider_region: str = "US"
    tmdb_backfill_delay_seconds: float = 0.25  # throttle for ml_training/backfill_assets.py
    database_url: str = "sqlite:///./movie_rec.db"
    min_reviews_for_sentiment: int = 3  # below this -> "insufficient_data"

    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
