import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "ml_training"))

import backfill_assets as ba

import httpx
import pytest

from app import config
from app.db.session import SessionLocal, init_db
from app.db.models import MovieAsset

init_db()


@pytest.fixture(autouse=True)
def _reset_tmdb_state():
    from app.services.tmdb_client import tmdb_client as shared_client

    original = config.settings.tmdb_api_key
    shared_client.clear_cache()
    yield
    config.settings.tmdb_api_key = original
    shared_client.clear_cache()


def _resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def _detail(movie_id, title, poster=None, backdrop=None, release_date=None, runtime=None):
    return {
        "id": movie_id,
        "title": title,
        "overview": "Test overview.",
        "poster_path": poster,
        "backdrop_path": backdrop,
        "release_date": release_date,
        "runtime": runtime,
        "vote_average": 7.0,
        "popularity": 10.0,
    }


def _search_results(items):
    return _resp({"results": items, "page": 1, "total_results": len(items)})


def _cleanup_movie_assets(*movie_ids):
    db = SessionLocal()
    try:
        for mid in movie_ids:
            row = db.get(MovieAsset, mid)
            if row:
                db.delete(row)
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def test_normalize_title():
    assert ba.normalize_title("The Dark Knight") == "dark knight"
    assert ba.normalize_title("  Star Wars: The Force Awakens!  ") == "star wars the force awakens"
    assert ba.normalize_title("Mad Max: Fury Road") == "mad max fury road"
    assert ba.normalize_title("Café") == "cafe"
    assert ba.normalize_title("") == ""


def test_parse_year():
    assert ba.parse_year("1999-10-15") == 1999
    assert ba.parse_year("2010") == 2010
    assert ba.parse_year("") is None
    assert ba.parse_year(None) is None


def test_select_match_exact_title():
    results = [
        {"id": 1, "title": "Inception", "release_date": "2010-07-16"},
        {"id": 2, "title": "Inside Out", "release_date": "2015-06-19"},
    ]
    match = ba.select_match(ba.normalize_title("Inception"), 2010, results)
    assert match["id"] == 1


def test_select_match_uses_year_proximity_among_exact_titles():
    results = [
        {"id": 10, "title": "The Thing", "release_date": "1982-06-25"},
        {"id": 11, "title": "The Thing", "release_date": "2011-10-14"},
    ]
    assert ba.select_match(ba.normalize_title("The Thing"), 1982, results)["id"] == 10
    assert ba.select_match(ba.normalize_title("The Thing"), 2011, results)["id"] == 11


def test_select_match_containment_with_matching_year():
    results = [
        {"id": 20, "title": "Jurassic World: Fallen Kingdom", "release_date": "2018-06-06"},
    ]
    match = ba.select_match(ba.normalize_title("Jurassic World"), 2018, results)
    assert match["id"] == 20


def test_select_match_returns_none_without_reliable_match():
    results = [
        {"id": 30, "title": "The Shawshank Redemption", "release_date": "1994-09-23"},
    ]
    assert ba.select_match(ba.normalize_title("Titanic"), 1997, results) is None


def test_is_complete():
    assert ba.is_complete({"poster_path": "/a.jpg", "backdrop_path": "/b.jpg", "release_date": "1999-01-01"})
    assert not ba.is_complete({"poster_path": "/a.jpg", "backdrop_path": None, "release_date": "1999-01-01"})
    assert not ba.is_complete({})


def test_movie_assets_from_detail_extracts_fields_and_missing_are_none():
    detail = _detail(550, "Fight Club", poster="/p.jpg", backdrop="/b.jpg", release_date="1999-10-15", runtime=139)
    assets = ba.movie_assets_from_detail(detail)
    assert assets == {
        "id": 550,
        "title": "Fight Club",
        "poster_path": "/p.jpg",
        "backdrop_path": "/b.jpg",
        "release_date": "1999-10-15",
        "runtime": 139,
    }

    sparse = ba.movie_assets_from_detail(_detail(551, "Sparse"))
    assert sparse["poster_path"] is None
    assert sparse["backdrop_path"] is None
    assert sparse["release_date"] is None
    assert sparse["runtime"] is None


# ---------------------------------------------------------------------------
# run_backfill behavior
# ---------------------------------------------------------------------------


def test_run_backfill_matches_persists_and_reports():
    config.settings.tmdb_api_key = "fake_key_for_test"
    _cleanup_movie_assets(1001, 1002)
    movies = [
        {"id": 1001, "title": "Fight Club", "release_date": "1999-10-15"},
        {"id": 1002, "title": "Inception B", "release_date": "2010-07-16"},
    ]
    by_id = {
        1001: _detail(1001, "Fight Club", poster="/fc.jpg", backdrop="/fc-b.jpg", release_date="1999-10-15", runtime=139),
        1002: _detail(1002, "Inception B", release_date="2010-07-16"),
    }

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            return _search_results([])
        return _resp(by_id[int(url.rstrip("/").rsplit("/", 1)[1])])

    lines = []
    db = SessionLocal()
    try:
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
            from app.services.tmdb_client import tmdb_client

            counters = asyncio.run(
                ba.run_backfill(tmdb_client, movies, set(), db, delay=0.0, state={}, out=lines.append)
            )

        assert counters["matched"] == 2
        assert counters["errors"] == 0
        assert counters["unmatched"] == 0

        db.expire_all()
        fc = db.get(MovieAsset, 1001)
        assert fc is not None
        assert fc.poster_path == "/fc.jpg"
        assert fc.backdrop_path == "/fc-b.jpg"
        assert fc.release_date == "1999-10-15"
        assert fc.runtime == 139
        inc = db.get(MovieAsset, 1002)
        assert inc.runtime is None  # missing runtime stays None - never fabricated
    finally:
        db.close()
        _cleanup_movie_assets(1001, 1002)

    assert "Fight Club" in "\n".join(lines)
    assert "matched" in "\n".join(lines).lower()


def test_already_complete_movies_are_skipped_without_requests():
    config.settings.tmdb_api_key = "fake_key_for_test"
    movies = [{"id": 2001, "title": "Done Movie", "release_date": "2001-01-01"}]
    mock_get = AsyncMock(return_value=_resp(_detail(2001, "Done Movie")))

    db = SessionLocal()
    try:
        with patch.object(httpx.AsyncClient, "get", mock_get):
            from app.services.tmdb_client import tmdb_client

            counters = asyncio.run(
                ba.run_backfill(tmdb_client, movies, already_done={2001}, db=db, delay=0.0, state={}, out=lambda _: None)
            )
        assert mock_get.call_count == 0
        assert counters["skipped_complete"] == 1
        assert counters["processed"] == 0
    finally:
        db.close()


def test_failed_request_does_not_abort_batch():
    config.settings.tmdb_api_key = "fake_key_for_test"

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            if params["query"] == "Alpha Movie":
                raise httpx.ConnectTimeout("simulated")
            return _search_results([{"id": 999302, "title": "Beta Movie", "release_date": "2015-05-05"}])
        movie_id = int(url.rstrip("/").rsplit("/", 1)[1])
        if movie_id == 999301:
            raise httpx.ConnectTimeout("simulated")
        return _resp(_detail(movie_id, "Beta Movie", poster="/beta.jpg"))

    from app.services.tmdb_client import TMDBClient

    client = TMDBClient(max_retries=1, retry_base_delay=0.01, retry_max_delay=0.01)
    movies = [
        {"id": 999301, "title": "Alpha Movie", "release_date": "1999-01-01"},
        {"id": 999302, "title": "Beta Movie", "release_date": "2015-05-05"},
    ]
    lines = []
    db = SessionLocal()
    try:
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
            counters = asyncio.run(
                ba.run_backfill(client, movies, set(), db, delay=0.0, state={}, out=lines.append)
            )
        assert counters["errors"] == 1
        assert counters["matched"] == 1
        observed = [line for line in lines if "error" in line]
        assert any("Alpha Movie" in line for line in observed)
        assert db.get(MovieAsset, 999302) is not None
    finally:
        db.close()
        _cleanup_movie_assets(999301, 999302)


def test_resume_skips_previous_unmatched_unless_retry_flag():
    config.settings.tmdb_api_key = "fake_key_for_test"
    movies = [
        {"id": 4001, "title": "Old Unmatched", "release_date": "2004-04-04"},
        {"id": 4002, "title": "Fresh Match", "release_date": "2005-05-05"},
    ]
    state = {"unmatched": [4001], "error": []}

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            return _search_results([])
        movie_id = int(url.rstrip("/").rsplit("/", 1)[1])
        if movie_id == 4002:
            return _resp(_detail(4002, "Fresh Match", poster="/f.jpg"))
        # 4001: matched id no longer resolves to the right title, and the title
        # search finds nothing - so it stays genuinely unmatched.
        return _resp(_detail(4001, "Not The Same Film"))

    from app.services.tmdb_client import tmdb_client

    db = SessionLocal()
    try:
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)) as mock_get:
            counters = asyncio.run(
                ba.run_backfill(tmdb_client, movies, set(), db, delay=0.0, state=state, out=lambda _: None)
            )
        assert counters["skipped_previous"] == 1
        assert counters["matched"] == 1

        # With --retry-unmatched the previously-unmatched id is attempted again
        # (and this time stays unmatched since the search has no results).
        state2 = {"unmatched": [4001], "error": []}
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)) as mock_get2:
            counters2 = asyncio.run(
                ba.run_backfill(tmdb_client, movies, set(), db, delay=0.0, retry_unmatched=True, state=state2, out=lambda _: None)
            )
        assert counters2["skipped_previous"] == 0
        assert counters2["unmatched"] == 1
        assert counters2["matched"] == 1
    finally:
        db.close()
        _cleanup_movie_assets(4002)


def test_dry_run_persists_nothing_and_writes_no_state_file(tmp_path):
    config.settings.tmdb_api_key = "fake_key_for_test"
    movies = [
        {"id": 9995001, "title": "Dry Movie", "release_date": "2006-06-06"},
    ]
    state_path = tmp_path / "state.json"
    lines = []

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            return _search_results([])
        return _resp(_detail(9995001, "Dry Movie", poster="/d.jpg"))

    from app.services.tmdb_client import tmdb_client

    db = SessionLocal()
    try:
        with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
            counters = asyncio.run(
                ba.run_backfill(
                    tmdb_client, movies, set(), db, delay=0.0, dry_run=True,
                    state={"unmatched": [], "error": []}, state_path=str(state_path), out=lines.append,
                )
            )
        assert counters["matched"] == 1
        assert db.get(MovieAsset, 9995001) is None
        assert not state_path.exists()
        assert any("[dry-run]" in line for line in lines)
    finally:
        db.close()


def test_state_file_is_written_and_never_contains_api_key(tmp_path):
    config.settings.tmdb_api_key = "supersecret-test-key"
    movies = [
        {"id": 6001, "title": "No Match Here", "release_date": "2007-07-07"},
    ]
    state = {"unmatched": [], "error": []}
    state_path = tmp_path / "state.json"

    async def fake_get(url, **kwargs):
        return _search_results([])

    from app.services.tmdb_client import tmdb_client

    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
        asyncio.run(
            ba.run_backfill(tmdb_client, movies, set(), None, delay=0.0, state=state, state_path=str(state_path), out=lambda _: None)
        )

    assert state_path.exists()
    content = state_path.read_text(encoding="utf-8")
    assert "supersecret-test-key" not in content
    import json

    assert 6001 in json.loads(content)["unmatched"]


def test_printed_output_never_contains_the_api_key():
    config.settings.tmdb_api_key = "prints-never-key-42"
    movies = [{"id": 7001, "title": "Keyless Movie", "release_date": "2008-08-08"}]
    lines = []

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            return _search_results([])
        return _resp(_detail(7001, "Keyless Movie"))

    from app.services.tmdb_client import tmdb_client

    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
        asyncio.run(
            ba.run_backfill(tmdb_client, movies, set(), None, delay=0.0, state=None, out=lines.append)
        )
    assert "prints-never-key-42" not in "\n".join(lines)


def test_configurable_delay_is_respected(monkeypatch):
    config.settings.tmdb_api_key = "fake_key_for_test"
    movies = [{"id": 8001, "title": "Slow Movie", "release_date": "2009-09-09"}]
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    async def fake_get(url, **kwargs):
        params = kwargs["params"]
        if "query" in params:
            return _search_results([])
        return _resp(_detail(8001, "Slow Movie"))

    from app.services.tmdb_client import tmdb_client

    monkeypatch.setattr(ba.asyncio, "sleep", fake_sleep)
    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=fake_get)):
        asyncio.run(
            ba.run_backfill(tmdb_client, movies, set(), None, delay=1.25, state=None, out=lambda _: None)
        )
    assert 1.25 in sleeps


# ---------------------------------------------------------------------------
# Persistence / request-time integration
# ---------------------------------------------------------------------------


def test_recommendation_service_serves_enriched_assets_for_catalogue_row():
    from app.services.library_service import upsert_movie_asset
    from app.services.recommendation_service import recommendation_service

    # Snapshot the real asset (if any) so the test never destroys a populated
    # backfill row: it restores the exact prior values instead of deleting.
    db = SessionLocal()
    try:
        original = db.get(MovieAsset, 135397)
        original_values = (
            original.title,
            original.poster_path,
            original.backdrop_path,
            original.release_date,
            original.runtime,
        ) if original is not None else None
    finally:
        db.close()

    db = SessionLocal()
    try:
        upsert_movie_asset(
            db,
            {
                "id": 135397,
                "title": "Jurassic World",
                "poster_path": "/jw-poster.jpg",
                "backdrop_path": "/jw-backdrop.jpg",
                "release_date": "2015-06-12",
                "runtime": 124,
            },
        )
    finally:
        db.close()

    try:
        recommendation_service.refresh()
        row = recommendation_service.get_movie_row(135397)
        assert row is not None
        assert row["poster_path"] == "/jw-poster.jpg"
        assert row["backdrop_path"] == "/jw-backdrop.jpg"
        assert row["release_date"] == "2015-06-12"
        assert row["runtime"] == 124
        # Local record stays authoritative for everything else.
        assert row["title"] == "Jurassic World"
        assert "Jurassic Park" in row["overview"]
        assert row["source"] == "local"
    finally:
        db = SessionLocal()
        try:
            if original_values is None:
                leftover = db.get(MovieAsset, 135397)
                if leftover:
                    db.delete(leftover)
            else:
                upsert_movie_asset(
                    db,
                    {
                        "id": 135397,
                        "title": original_values[0],
                        "poster_path": original_values[1],
                        "backdrop_path": original_values[2],
                        "release_date": original_values[3],
                        "runtime": original_values[4],
                    },
                )
        finally:
            db.close()
        recommendation_service.refresh()

    row = recommendation_service.get_movie_row(135397)
    if original_values is None:
        assert row["poster_path"] is None  # cleanup returns it to a blank asset
    else:
        assert row["poster_path"] == original_values[1]
        assert row["backdrop_path"] == original_values[2]
        assert row["release_date"] == original_values[3]
        assert row["runtime"] == original_values[4]


def test_backfill_preserves_existing_library_and_user_data():
    from app.db.models import LibraryMovie, User, UserInteraction, WatchlistItem
    from app.services.library_service import upsert_library_movie, upsert_movie_asset

    db = SessionLocal()
    user = None
    try:
        _cleanup_movie_assets(999010)
        existing = db.get(LibraryMovie, 999010)
        if existing:
            db.delete(existing)
            db.commit()

        upsert_library_movie(
            db,
            {
                "id": 999010,
                "title": "Library Ten",
                "overview": "kept overview",
                "genres": "Science Fiction",
                "cast": "Kept Actor",
                "vote_average": 8.0,
                "popularity": 3.0,
            },
        )
        user = User(session_id="bf-preserve-user")
        db.add(user)
        db.commit()
        db.refresh(user)
        db.add(UserInteraction(user_id=user.id, movie_id=999010, interaction_type="liked"))
        db.add(WatchlistItem(user_id=user.id, movie_id=999010, movie_title="Library Ten"))
        db.commit()

        upsert_movie_asset(
            db,
            {
                "id": 999010,
                "title": "Library Ten",
                "poster_path": "/asset-poster.jpg",
                "backdrop_path": "/asset-backdrop.jpg",
                "release_date": "2020-01-01",
                "runtime": 90,
            },
        )

        lib = db.get(LibraryMovie, 999010)
        asset = db.get(MovieAsset, 999010)
        assert lib.poster_path is None  # asset enrichment did not touch LibraryMovie
        assert asset.poster_path == "/asset-poster.jpg"  # stored separately
        assert db.query(UserInteraction).filter_by(user_id=user.id, movie_id=999010).count() == 1
        assert db.query(WatchlistItem).filter_by(user_id=user.id, movie_id=999010).count() == 1
    finally:
        if user is not None:
            db.query(UserInteraction).filter_by(user_id=user.id).delete()
            db.query(WatchlistItem).filter_by(user_id=user.id).delete()
            db.delete(db.get(User, user.id))
        leftover = db.get(LibraryMovie, 999010)
        if leftover:
            db.delete(leftover)
        db.commit()
        db.close()
        _cleanup_movie_assets(999010)