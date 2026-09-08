"""
Phase 1 Step 3 - TMDB asset backfill for the existing local catalogue.

Scans the original CSV catalogue, and for every movie missing poster/backdrop/
release-date assets, looks the movie up on TMDB and persists the found assets
into the movie_assets enrichment table. The CSV itself is never rewritten.

Resumable: movies that already have a movie_assets row (or non-blank assets in
the CSV) are skipped, and previously failed/unmatched ids are recorded in a
small state file so a re-run does not re-request them all.

    python ml_training/backfill_assets.py            # normal run
    python ml_training/backfill_assets.py --dry-run  # match but persist nothing
    python ml_training/backfill_assets.py --limit 50 # process at most 50

Throttling is controlled by TMDB_BACKFILL_DELAY_SECONDS (config, default 0.25s)
or --delay. The script only talks to TMDB through the shared tmdb_client.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

# Load app modules the same way ml_training/seed_dataset.py does, so this can be
# run as `python ml_training/backfill_assets.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.db.models import MovieAsset  # noqa: E402
from app.db.session import SessionLocal, init_db  # noqa: E402
from app.services.data_prep import tmdb_movie_to_row  # noqa: E402
from app.services.library_service import upsert_movie_asset  # noqa: E402
from app.services.tmdb_client import (  # noqa: E402
    tmdb_client,
    TMDBError,
    TMDBNotFoundError,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "movies_dataset.csv"
DEFAULT_STATE = DATA_DIR / "backfill_state.json"

ASSET_FIELDS = ("poster_path", "backdrop_path", "release_date")
CACHE_CLEAR_EVERY = 25  # bound memory: the shared client caches every 200 OK

TITLE_ARTICLES = ("the ", "a ", "an ")


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return str(value).strip() == ""


def normalize_title(title) -> str:
    """Lower-cased, punctuation-stripped, article-trimmed title for matching."""
    if not title:
        return ""
    text = unicodedata.normalize("NFKD", str(title))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    for article in TITLE_ARTICLES:
        if text.startswith(article) and len(text) > len(article):
            text = text[len(article):]
    return text


def parse_year(value) -> int | None:
    if _is_blank(value):
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def _movie_year(result: dict) -> int | None:
    return parse_year(result.get("release_date"))


def select_match(local_title_norm: str, local_year: int | None, results: list[dict]) -> dict | None:
    """Deterministic best-match from TMDB search results.

    Preference order:
    1. Exact normalized title match; among several, closest release year.
    2. Containment match (one normalized title inside the other, >= 4 chars)
       AND a matching release year.
    Anything else is treated as "no reliable match".
    """
    exact = [r for r in results if normalize_title(r.get("title")) == local_title_norm]
    if exact:
        if local_year is not None:
            exact = sorted(exact, key=lambda r: abs((_movie_year(r) or 9999) - local_year))
        return exact[0]

    for result in results:
        other = normalize_title(result.get("title"))
        if not other or len(other) < 4:
            continue
        if other in local_title_norm or local_title_norm in other:
            other_year = _movie_year(result)
            if local_year is not None:
                if other_year == local_year:
                    return result
            elif other_year is not None:
                return result
    return None


def is_complete(row: dict) -> bool:
    """A movie is 'complete' only when ALL three core assets are present.
    Runtime may legitimately be missing from TMDB, so it is not required."""
    return all(not _is_blank(row.get(field)) for field in ASSET_FIELDS)


def movie_assets_from_detail(detail: dict) -> dict:
    row = tmdb_movie_to_row(detail)
    return {key: row.get(key) for key in ("id", "title", "poster_path", "backdrop_path", "release_date", "runtime")}


def load_catalogue(path: str) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def db_asset_ids(db) -> set[int]:
    return {asset.tmdb_id for asset in db.query(MovieAsset).all()}


def load_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, OSError, ValueError):
        return {}


def save_state(path: str, state: dict) -> None:
    """Atomic-ish write so a crash mid-write cannot corrupt the previous state."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError:
        return
    try:
        os.replace(tmp_path, path)
    except OSError:
        pass


def _summary(out, prefix: str, counters: dict) -> None:
    out(
        f"{prefix}Processed: {counters['processed']}  "
        f"Matched: {counters['matched']}  "
        f"Unmatched: {counters['unmatched']}  "
        f"Errors: {counters['errors']}  "
        f"Skipped/already complete: {counters['skipped_complete']}"
    )


def _print_matched(out, prefix, assets) -> None:
    out(
        f"{prefix}  poster: {'yes' if assets['poster_path'] else 'no'}  "
        f"backdrop: {'yes' if assets['backdrop_path'] else 'no'}  "
        f"release_date: {assets['release_date'] or '-'}  "
        f"runtime: {assets['runtime'] or '-'}"
    )


async def _throttled(delay: float, awaitable):
    """Run one TMDB call, then dwell `delay` seconds to spread out the load."""
    result = await awaitable
    if delay and delay > 0:
        await asyncio.sleep(delay)
    return result


async def process_movie(client, row: dict, delay: float = 0.0) -> dict | None:
    """Match a local catalogue row to TMDB and return its detail JSON (or None).

    Strategy: our CSV `id` column IS the TMDB id, so try fetching detail by id
    first (highest confidence) and validate the title. Only fall back to a
    title search when that fails, so genuinely mismatched ids can still be
    found by name.
    """
    title_norm = normalize_title(row.get("title"))
    local_year = parse_year(row.get("release_date"))
    detail = None

    try:
        candidate = await _throttled(delay, client.get_movie(int(row["id"])))
        if candidate and select_match(title_norm, local_year, [candidate]) is not None:
            detail = candidate
    except TMDBNotFoundError:
        detail = None
    except TMDBError:
        detail = None  # service hiccup on the id fetch - try the search path once

    if detail is not None:
        return detail

    try:
        search = await _throttled(
            delay, client.search_movies(row.get("title", ""), year=local_year or None)
        )
    except TMDBNotFoundError:
        return None
    except TMDBError:
        raise

    candidate = select_match(title_norm, local_year, search.get("results", []))
    if candidate is None:
        return None
    return await _throttled(delay, client.get_movie(candidate["id"]))


async def run_backfill(
    client,
    movies: list[dict],
    already_done: set[int],
    db,
    *,
    delay: float = 0.0,
    every: int = 100,
    dry_run: bool = False,
    retry_unmatched: bool = False,
    state: dict | None = None,
    state_path: str | None = None,
    out=print,
) -> dict:
    """Process `movies`, skipping ones in `already_done` and previously-failed
    ids (unless retry_unmatched). Persists assets via `db` unless dry_run.

    Returns counters dict. `state` (mutated lists of unmatched/error ids) is
    flushed to `state_path` periodically and at the end - never on dry-run.
    """
    previous_unmatched = set()
    if state and not dry_run:
        previous_unmatched.update(state.get("unmatched", []))
        previous_unmatched.update(state.get("error", []))
    if retry_unmatched:
        previous_unmatched = set()

    counters = {
        "processed": 0,
        "matched": 0,
        "unmatched": 0,
        "errors": 0,
        "skipped_complete": 0,
        "skipped_previous": 0,
    }
    prefix = "[dry-run] " if dry_run else ""

    for movie in movies:
        movie_id = int(movie["id"])
        title = movie.get("title", "") or "Unknown"
        ordinal = counters["processed"] + counters["skipped_complete"] + counters["skipped_previous"] + 1

        if movie_id in already_done:
            counters["skipped_complete"] += 1
            continue
        if movie_id in previous_unmatched:
            counters["skipped_previous"] += 1
            continue

        try:
            detail = await process_movie(client, movie, delay)
        except (TMDBError, TMDBNotFoundError, ValueError, TypeError) as exc:
            counters["errors"] += 1
            if state is not None and not dry_run:
                state.setdefault("error", []).append(movie_id)
            out(f"{prefix}[{ordinal}] {title}: error - {type(exc).__name__}: {exc}")
        else:
            if detail is None:
                counters["unmatched"] += 1
                if state is not None and not dry_run:
                    state.setdefault("unmatched", []).append(movie_id)
                out(f"{prefix}[{ordinal}] {title}: no reliable TMDB match - skipped")
            else:
                assets = movie_assets_from_detail(detail)
                if not dry_run and db is not None:
                    upsert_movie_asset(db, assets)
                counters["matched"] += 1
                if state is not None and not dry_run:
                    state.setdefault("matched", []).append(movie_id)
                out(f"{prefix}[{ordinal}] {title}: matched {assets['title']}")
                _print_matched(out, prefix, assets)

        counters["processed"] += 1
        if counters["processed"] % CACHE_CLEAR_EVERY == 0:
            client.clear_cache()
        if counters["processed"] % every == 0:
            _summary(out, prefix, counters)
            if not dry_run and state is not None and state_path:
                save_state(state_path, state)

    _summary(out, prefix, counters)
    if not dry_run and state is not None and state_path:
        save_state(state_path, state)
    return counters


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill missing poster/backdrop/release-date/runtime "
                    "assets for the local catalogue from TMDB."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Local catalogue CSV")
    parser.add_argument("--state", default=str(DEFAULT_STATE), help="Progress/state JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Match but persist nothing")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N movies (0 = all)")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between TMDB requests")
    parser.add_argument("--retry-unmatched", action="store_true", help="Re-attempt previously unmatched/errored ids")
    parser.add_argument("--every", type=int, default=100, help="Progress summary interval")
    parser.add_argument("--no-state", action="store_true", help="Do not write the state file")
    args = parser.parse_args(argv)

    if args.delay is None:
        args.delay = settings.tmdb_backfill_delay_seconds

    init_db()
    movies = load_catalogue(args.csv)
    if args.limit > 0:
        movies = movies[: args.limit]

    db = SessionLocal()
    try:
        already_done = {int(r["id"]) for r in movies if is_complete(r)} | db_asset_ids(db)
    finally:
        db.close()

    state = None if (args.dry_run or args.no_state) else load_state(args.state)
    if state and not args.retry_unmatched:
        blocked = set(state.get("unmatched", [])) | set(state.get("error", []))
    else:
        blocked = set()
    remaining = sum(1 for r in movies if int(r["id"]) not in already_done and int(r["id"]) not in blocked)

    print("TMDB Asset Backfill")
    print("-------------------")
    print(f"Total movies: {len(movies)}")
    print(f"Already complete: {len(already_done)}")
    print(f"Remaining: {remaining}")
    if args.dry_run:
        print("Dry run - no changes will be persisted.")

    # Re-open a session for the run itself (the first one was used for the
    # completeness query and closed above).
    db = SessionLocal()
    try:
        asyncio.run(
            run_backfill(
                client=tmdb_client,
                movies=movies,
                already_done=already_done,
                db=None if args.dry_run else db,
                delay=args.delay,
                every=args.every,
                dry_run=args.dry_run,
                retry_unmatched=args.retry_unmatched,
                state=state,
                state_path=None if args.dry_run else args.state,
                out=print,
            )
        )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())