import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import config


@pytest.fixture(autouse=True)
def _reset_tmdb_state():
    from app.services.tmdb_client import tmdb_client as shared_client

    original = config.settings.tmdb_api_key
    shared_client.clear_cache()
    yield
    config.settings.tmdb_api_key = original
    shared_client.clear_cache()


def _movie_detail(**overrides) -> dict:
    """A realistic TMDB movie-detail payload (as returned by
    append_to_response) with credits, keywords, videos, and watch providers."""
    detail = {
        "id": 550,
        "title": "Fight Club",
        "overview": "An insomniac office worker and a devil-may-care soap maker.",
        "runtime": 139,
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "backdrop_path": "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg",
        "release_date": "1999-10-15",
        "vote_average": 8.4,
        "popularity": 60.0,
        "genres": [{"name": "Drama"}, {"name": "Thriller"}],
        "credits": {
            "cast": [{"name": "Brad Pitt"}, {"name": "Edward Norton"}],
            "crew": [{"name": "David Fincher", "job": "Director"}],
        },
        "keywords": {"keywords": [{"name": "insomnia"}]},
        "videos": {
            "results": [
                {
                    "site": "YouTube",
                    "key": "trailer_key",
                    "name": "Official Trailer",
                    "type": "Trailer",
                    "official": True,
                    "iso_639_1": "en",
                }
            ]
        },
        "watch/providers": {
            "results": {
                "US": {
                    "link": "https://www.themoviedb.org/movie/550/watch?locale=US",
                    "flatrate": [
                        {"provider_name": "Netflix", "logo_path": "/netflix.jpg", "display_priority": 0}
                    ],
                    "rent": [
                        {"provider_name": "Apple TV", "logo_path": "/apple.jpg", "display_priority": 1}
                    ],
                    "buy": [
                        {"provider_name": "Amazon Prime Video", "logo_path": "/amzn.jpg", "display_priority": 2}
                    ],
                },
                "GB": {
                    "link": "https://www.themoviedb.org/movie/550/watch?locale=GB",
                    "flatrate": [{"provider_name": "Sky Store", "logo_path": "/sky.jpg", "display_priority": 0}],
                    "rent": [],
                    "buy": [],
                },
            }
        },
    }
    detail.update(overrides)
    return detail


def _ok_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def test_get_movie_requests_videos_and_watch_providers():
    """get_movie must keep credits+keywords AND also fetch videos and
    watch/providers in the same append_to_response call."""
    from app.services.tmdb_client import TMDBClient
    import asyncio

    async def run():
        client = TMDBClient(max_retries=1)
        config.settings.tmdb_api_key = "fake_key_for_test"
        req = httpx.Request("GET", "http://tmdb")
        ok = AsyncMock(return_value=httpx.Response(200, request=req, json={"id": 550}))
        with patch.object(httpx.AsyncClient, "get", ok):
            data = await client.get_movie(550)
        assert data == {"id": 550}
        params = ok.call_args.kwargs["params"]
        appended = {p.strip() for p in params["append_to_response"].split(",")}
        assert {"credits", "keywords", "videos", "watch/providers"} <= appended

    asyncio.run(run())


def test_tmdb_movie_to_row_maps_runtime_and_paths():
    from app.services.data_prep import tmdb_movie_to_row

    row = tmdb_movie_to_row(_movie_detail())
    assert row["runtime"] == 139
    assert row["poster_path"] == "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    assert row["backdrop_path"] == "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg"
    assert row["release_date"] == "1999-10-15"
    assert row["vote_average"] == 8.4
    assert row["popularity"] == 60.0
    assert row["genres"] == ["Drama", "Thriller"]
    assert row["cast"] == ["Brad Pitt", "Edward Norton"]
    assert row["director"] == ["David Fincher"]
    assert row["keywords"] == ["insomnia"]


def test_tmdb_movie_to_row_handles_missing_fields():
    from app.services.data_prep import tmdb_movie_to_row

    row = tmdb_movie_to_row({"id": 1})
    assert row["id"] == 1
    assert row["runtime"] is None
    assert row["poster_path"] is None
    assert row["backdrop_path"] is None
    assert row["release_date"] == ""
    assert row["genres"] == []
    assert row["cast"] == []
    assert row["director"] == []
    assert row["keywords"] == []
    assert row["vote_average"] == 0
    assert row["popularity"] == 0


def test_select_trailer_prefers_official_english_youtube_trailer():
    from app.services.tmdb_details import select_trailer

    videos = [
        {"site": "Vimeo", "key": "v1", "type": "Trailer", "official": True, "iso_639_1": "en"},
        {"site": "YouTube", "key": "y1", "type": "Featurette", "official": True, "iso_639_1": "en"},
        {"site": "YouTube", "key": "y2", "type": "Trailer", "official": False, "iso_639_1": "en"},
        {"site": "YouTube", "key": "y3", "type": "Trailer", "official": True, "iso_639_1": "es"},
        {"site": "YouTube", "key": "y4", "type": "Trailer", "official": True, "iso_639_1": "en"},
    ]
    trailer = select_trailer(videos)
    assert trailer["key"] == "y4"
    assert trailer["url"] == "https://www.youtube.com/watch?v=y4"


def test_select_trailer_returns_none_when_no_videos():
    from app.services.tmdb_details import select_trailer

    assert select_trailer(None) is None
    assert select_trailer([]) is None


def test_select_trailer_falls_back_to_first_reasonable_video():
    from app.services.tmdb_details import select_trailer

    videos = [
        {"site": "YouTube", "key": "c1", "type": "Clip", "official": False, "iso_639_1": "fr"},
        {"site": "Vimeo", "key": "c2", "type": "Trailer", "official": False, "iso_639_1": "de"},
    ]
    trailer = select_trailer(videos)
    # A YouTube clip beats a non-YouTube trailer per our preference order.
    assert trailer["key"] == "c1"
    assert trailer["site"] == "YouTube"


def test_parse_watch_providers_normalizes_region_groups():
    from app.services.tmdb_details import parse_watch_providers

    providers = parse_watch_providers(_movie_detail()["watch/providers"], "US")
    assert providers is not None
    assert providers["region"] == "US"
    assert providers["available"] is True
    assert [p["provider_name"] for p in providers["flatrate"]] == ["Netflix"]
    assert [p["provider_name"] for p in providers["rent"]] == ["Apple TV"]
    assert [p["provider_name"] for p in providers["buy"]] == ["Amazon Prime Video"]
    assert providers["link"].startswith("https://")


def test_parse_watch_providers_returns_none_when_no_data():
    from app.services.tmdb_details import parse_watch_providers

    assert parse_watch_providers(None, "US") is None
    assert parse_watch_providers({"results": {}}, "US") is None
    assert parse_watch_providers({"results": {"US": {}}}, "US") is None
    # Only other regions available -> nothing claimable for the configured one.
    assert parse_watch_providers({"results": {"FR": _movie_detail()["watch/providers"]["results"]["US"]}}, "US") is None


def test_parse_watch_providers_does_not_claim_availability_without_providers():
    from app.services.tmdb_details import parse_watch_providers

    providers = parse_watch_providers(
        {"results": {"US": {"link": "https://x", "flatrate": [], "rent": [], "buy": [], "ads": []}}},
        "US",
    )
    assert providers is not None
    assert providers["available"] is False
    assert providers["flatrate"] == []
    assert providers["rent"] == []


def test_movie_detail_endpoint_exposes_rich_fields():
    config.settings.tmdb_api_key = "fake_key_for_test"

    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_ok_response(_movie_detail()))):
        with TestClient(app) as client:
            res = client.get("/movies/550/tmdb-detail")
            assert res.status_code == 200
            body = res.json()

    assert body["source"] == "tmdb_external"
    assert body["runtime"] == 139
    assert body["poster_path"] == "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
    assert body["backdrop_path"] == "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg"
    assert body["release_date"] == "1999-10-15"
    assert body["genres"] == ["Drama", "Thriller"]
    assert body["cast"] == ["Brad Pitt", "Edward Norton"]
    assert body["director"] == ["David Fincher"]
    assert body["trailer"]["key"] == "trailer_key"
    assert body["trailer"]["url"] == "https://www.youtube.com/watch?v=trailer_key"
    assert body["watch_providers"]["available"] is True
    assert body["watch_providers"]["flatrate"][0]["provider_name"] == "Netflix"


def test_movie_detail_endpoint_returns_404_when_tmdb_movie_missing():
    config.settings.tmdb_api_key = "fake_key_for_test"
    req = httpx.Request("GET", "http://tmdb")
    missing = AsyncMock(return_value=httpx.Response(404, request=req, text="not found"))

    with patch.object(httpx.AsyncClient, "get", missing):
        with TestClient(app) as client:
            res = client.get("/movies/987654321/tmdb-detail")
            assert res.status_code == 404


def test_movie_detail_endpoint_returns_502_on_service_error():
    config.settings.tmdb_api_key = "fake_key_for_test"
    req = httpx.Request("GET", "http://tmdb")
    bad = AsyncMock(return_value=httpx.Response(400, request=req, text="bad request"))

    with patch.object(httpx.AsyncClient, "get", bad):
        with TestClient(app) as client:
            res = client.get("/movies/550/tmdb-detail")
            assert res.status_code == 502


def test_library_persists_runtime_round_trip():
    """runtime survives the imported-library round trip (upsert -> reload)."""
    from app.db.session import SessionLocal
    from app.db.models import LibraryMovie
    from app.services.library_service import upsert_library_movie, library_movie_to_row
    from app.services.data_prep import tmdb_movie_to_row

    movie_id = 999003
    db = SessionLocal()
    try:
        existing = db.get(LibraryMovie, movie_id)
        if existing:
            db.delete(existing)
            db.commit()

        row = tmdb_movie_to_row(_movie_detail(id=movie_id))
        movie, was_new = upsert_library_movie(db, row)
        assert was_new is True
        assert movie.runtime == 139
        assert movie.poster_path == "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
        assert movie.backdrop_path == "/hZkgoQYus5vegHoetLkCJzb17zJ.jpg"
        assert movie.release_date == "1999-10-15"

        restored = library_movie_to_row(db.get(LibraryMovie, movie_id))
        assert restored["runtime"] == 139
        assert restored["poster_path"] == "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg"
        assert restored["release_date"] == "1999-10-15"
    finally:
        leftover = db.get(LibraryMovie, movie_id)
        if leftover:
            db.delete(leftover)
            db.commit()
        db.close()


def test_missing_tmdb_fields_do_not_crash_detail_endpoint():
    config.settings.tmdb_api_key = "fake_key_for_test"
    sparse = {"id": 1234, "title": "Sparse", "credits": {}, "keywords": {}}

    with patch.object(httpx.AsyncClient, "get", AsyncMock(return_value=_ok_response(sparse))):
        with TestClient(app) as client:
            res = client.get("/movies/1234/tmdb-detail")
            assert res.status_code == 200
            body = res.json()

    assert body["runtime"] is None
    assert body["trailer"] is None
    assert body["watch_providers"] is None
    assert body["poster_path"] is None
    assert body["release_date"] == ""