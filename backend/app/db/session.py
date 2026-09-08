from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Columns that were added to models after the first deployed DB was created.
# SQLite has no ALTER TABLE IF NOT EXISTS and Base.metadata.create_all() never
# touches existing tables, so the smallest safe approach is to add missing
# columns one at a time. ALTER TABLE ADD COLUMN is a metadata-only change in
# SQLite and never rewrites the file, so it is cheap and preserves all data.
_ADDITIVE_COLUMNS = {
    "library_movies": [
        {"name": "runtime", "definition": "ADD COLUMN runtime INTEGER"},
    ],
}


def _ensure_schema_columns():
    """Idempotently add any model columns that existing SQLite tables lack.

    Runs automatically once at import time (and again at app startup) so the
    schema is correct before any ORM query - including tests that touch the
    DB before a TestClient boot. No-op on non-SQLite backends.
    """
    if not (settings.database_url or "").startswith("sqlite"):
        return
    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        for table, columns in _ADDITIVE_COLUMNS.items():
            if table not in existing_tables:
                continue  # create_all() builds brand-new tables with full schema
            existing_cols = {c["name"] for c in inspector.get_columns(table)}
            with engine.begin() as conn:
                for col in columns:
                    if col["name"] not in existing_cols:
                        conn.execute(text(f"ALTER TABLE {table} {col['definition']}"))
    except Exception:
        # Never take the app down over a missing optional column.
        pass


_ensure_schema_columns()


def init_db():
    Base.metadata.create_all(bind=engine)
    _ensure_schema_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
