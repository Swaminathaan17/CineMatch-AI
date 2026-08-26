"""
Central config. Reads from environment variables / Replit Secrets.
Never hardcode the TMDB key anywhere else in the codebase.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    tmdb_api_key: str = ""
    tmdb_base_url: str = "https://api.themoviedb.org/3"
    database_url: str = "sqlite:///./movie_rec.db"
    min_reviews_for_sentiment: int = 3  # below this -> "insufficient_data"

    class Config:
        env_file = ".env"


settings = Settings()
