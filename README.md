# Advanced Movie Recommendation System — Week 1

## What's built so far (Module 1)

- `backend/app/` — FastAPI app (main.py, config, routers, services)
- `backend/ml_engine/` — standalone ML package, no FastAPI/TMDB imports:
  - `content_similarity.py` — TF-IDF + cosine similarity engine
  - `explainer.py` — turns similarity scores into human-readable reasons
- `backend/app/services/tmdb_client.py` — TMDB API wrapper
- `backend/app/services/data_prep.py` — reshapes TMDB JSON into engine input
- `ml_training/seed_dataset.py` — one-time script to pull ~real movies from TMDB
- `data/sample_movies.csv` — 10-movie test set (used until you seed real data)

## Run it in Replit

1. Import this project (upload the zip, or push to your own GitHub repo first
   and import from there)
2. `cd backend && pip install -r requirements.txt`
3. Add your TMDB key as a Replit Secret: `TMDB_API_KEY`
4. Test with the sample dataset first (no key needed):
   `uvicorn app.main:app --host 0.0.0.0 --port 8000`
   Then visit `/docs` for interactive Swagger API docs, or try:
   `curl "http://localhost:8000/recommendations/1?top_n=5"`
5. Once your TMDB key works, seed a real dataset:
   `python ml_training/seed_dataset.py --pages 20`
   This creates `data/movies_dataset.csv` — the app will automatically use it
   over the sample data the next time it starts.

## Verified working right now

- `GET /health` → `{"status": "ok"}`
- `GET /movies/` → list of movies in the local dataset
- `GET /movies/{id}` → single movie detail
- `GET /movies/search/tmdb?q=inception` → live TMDB search (needs API key)
- `GET /recommendations/{id}?top_n=10` → similar movies with match % and reasons

Example (tested against the sample data): recommending against Inception (id=1)
correctly surfaces the other Christopher Nolan films, with the reason text
naming the shared director/genres — not just a bare score.

## Next: Week 2

Sentiment model (overall + aspect-based) and the hybrid ranker that blends
sentiment + rating + popularity into `content_similarity`'s output.

---

## Week 2 — Sentiment + Hybrid Ranking (DONE)

### New pieces

- `ml_engine/sentiment_model.py` — loads trained TF-IDF + Logistic Regression
  artifacts, predicts label + confidence for single reviews or batches
- `ml_engine/aspect_sentiment.py` — splits reviews into sentences, anchors
  sentences to 6 aspects (story, acting, direction, visuals, music, pacing)
  via keyword dictionaries, aggregates sentiment per aspect. Aspects with
  fewer than `min_reviews_for_sentiment` (default 3) anchored sentences
  return `"insufficient_data"` instead of a guessed score.
- `ml_engine/hybrid_ranker.py` — combines content similarity + sentiment +
  rating + popularity into one score, weights configurable in
  `ml_engine/weights.yaml`. When sentiment data is missing, its weight is
  redistributed across the other signals rather than assumed neutral.
- `ml_training/train_sentiment.py` — trains and evaluates the sentiment
  classifier, writes accuracy/F1/confusion matrix to
  `ml_training/sentiment_eval_report.txt`
- `app/services/review_fetcher.py` — pulls reviews from TMDB's official
  `/reviews` endpoint
- `app/routers/sentiment.py` — `GET /sentiment/{movie_id}`
- New endpoint: `GET /recommendations/{movie_id}/hybrid` — re-ranks the
  content-similarity candidate pool using the full hybrid score

### IMPORTANT — before citing any accuracy numbers

The sentiment model shipped in this zip was trained on a 40-example
hand-written dataset (`data/sample_reviews_labeled.csv`) — that's only
enough to prove the pipeline runs correctly end to end, not a real result.
5-fold CV on it comes out around 55% (barely above chance), which is
expected and fine for a sanity check, not something to report as your
model's accuracy.

**Before your submission**, download the IMDB 50k Movie Reviews dataset
(Kaggle: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`) into
`data/imdb_50k.csv` and retrain:

```
python ml_training/train_sentiment.py --data data/imdb_50k.csv --review-col review --label-col sentiment
```

This will give you a real, citable accuracy/F1 score (expect roughly
85-90% with this approach on that dataset) — check
`ml_training/sentiment_eval_report.txt` after running it.

### Verified working (tested in this environment without a TMDB key)

- Sentiment endpoint correctly returns `insufficient_data` rather than a
  fabricated score when no reviews are available
- Hybrid ranker correctly redistributes weight away from sentiment when
  it's missing, instead of assuming a neutral 0.5
- Aspect anchoring correctly buckets a single sentence into multiple
  aspects when it mentions more than one (e.g. a sentence about both
  pacing and story)

## Next: Week 3

Personalization (user preference vectors from liked movies), explainability
extended to include sentiment/rating reasoning, and the frontend build-out.


---

## Week 3 — Personalization + Frontend (DONE)

### Backend additions

- `app/db/models.py` + `app/db/session.py` — SQLite via SQLAlchemy: users,
  genre preferences, interaction log (liked/viewed/searched)
- `app/services/user_service.py` — session-based (no auth) user lookup,
  interaction logging, preference retrieval
- `ml_engine/personalization.py` — builds a user preference vector from the
  TF-IDF vectors of movies they've liked (reuses the same vector space as
  content similarity, no separate embedding step), optionally boosted by
  explicit favorite genres
- New endpoints:
  - `POST /users/interactions` — log a like/view/search
  - `POST /users/favorite-genres` — set explicit genre preferences
  - `GET /users/preferences` — current profile
  - `GET /recommendations/personalized?session_id=` — personalized ranking,
    **clearly labeled** `"mode": "trending"` vs `"mode": "personalized"` so
    the frontend (and you, presenting this) never claims personalization
    that isn't actually happening yet for a cold-start user

### Verified working (tested end-to-end in this environment)

- Cold-start session → `mode: trending`, not fake personalization
- After liking Inception + Interstellar → `mode: personalized`, with The
  Prestige (also Christopher Nolan) correctly surfacing near the top
- Preferences persist correctly across requests via SQLite

### Frontend (React + Vite + Tailwind + Framer Motion)

Design direction: cinematic, not generic dark-mode-AI-template. Deep charcoal
background, wine/burgundy accent (film curtain, not sci-fi neon), warm gold
for ratings/highlights. Display serif (Fraunces) for titles, Inter for UI,
JetBrains Mono for data read-outs (percentages, tags). Signature element: a
horizontal "filmstrip" scroll rail for movie rows instead of a generic grid.

Pages built:
- `Landing.jsx` — cinematic hero with entrance animation
- `Home.jsx` — "For You" (personalized) + "Trending Now" rows
- `MovieDetail.jsx` — backdrop, genres, like button, sentiment gauge +
  per-aspect sentiment cards, similar movies row

### Run the frontend

```
cd frontend
npm install
cp .env.example .env    # point VITE_API_URL at your backend
npm run dev
```

Verified: `npm run build` completes cleanly (361KB JS / 12KB CSS gzipped),
and the built app serves correctly via `npm run preview`.

## Next: Week 4

AI Discovery (natural-language / mood search using sentence-embeddings),
animation/polish pass, responsiveness testing, and deployment.

---

## Week 4 — AI Discovery, Testing, Deployment (DONE)

### AI Discovery (mood/NL search)

- `ml_engine/nl_query_parser.py` — two-tier design:
  - **Preferred**: `sentence-transformers` (all-MiniLM-L6-v2) encodes the
    query and movie overviews into the same semantic space, so "mind-bending"
    can match "reality-bending" with zero shared keywords
  - **Fallback**: if the model can't load (not installed, or no internet to
    download weights), degrades to TF-IDF cosine similarity over the same
    text. Every response reports which mode produced it (`"mode": "semantic"`
    vs `"mode": "fallback"`) — never silently pretends both are equivalent
- `POST /discovery/query` — verified working end-to-end in fallback mode in
  this environment (no internet for the HF model download here); correctly
  surfaced Inception for "a mind-bending sci-fi movie with emotional depth"
- Frontend: `AIDiscovery.jsx` page with example query chips

**Note:** the sentence-transformers model will download automatically the
first time `python ml_training/seed_dataset.py` or the app starts in Replit
(open internet there) — nothing extra needed on your end beyond having
`sentence-transformers` installed from requirements.txt.

### Test suite

`backend/tests/` — 23 tests, all passing:
- `test_content_similarity.py` — similarity ranking correctness, self-exclusion,
  a regression test for the space-stripping display bug from Week 1
- `test_hybrid_ranker.py` — score bounds, missing-sentiment weight
  redistribution (explicitly checks it does NOT default to a fake neutral
  0.5), zero-division safety
- `test_aspect_sentiment.py` — sentence splitting, multi-aspect anchoring,
  insufficient-data thresholding, aggregation math (using a stubbed model so
  the test doesn't depend on how good the trained model currently is)
- `test_api_integration.py` — every endpoint, including the cold-start →
  personalized mode transition

Run with:
```
cd backend && python -m pytest tests/ -v
```

### Deployment (Replit)

**Backend:**
1. Set Repl Secrets: `TMDB_API_KEY`
2. `cd backend && pip install -r requirements.txt`
3. Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
4. Seed real data: `python ml_training/seed_dataset.py --pages 20`
5. Train the real sentiment model on the full IMDB 50k dataset (see Week 2
   notes above) — don't ship the toy 40-example model

**Frontend:**
1. `cd frontend && npm install`
2. Set `VITE_API_URL` in `.env` to your backend's Repl URL
3. For development: `npm run dev -- --host`
4. For the actual deployed version: `npm run build`, then either serve
   `frontend/dist` as a static site (Replit's static deployment, or
   Vercel/Netlify), or point FastAPI's `StaticFiles` at it to serve
   everything from one Repl

**Before presenting/submitting:**
- Tighten CORS in `app/main.py` (`allow_origins=["*"]` → your actual frontend
  origin)
- Retrain the sentiment model on the full dataset and update
  `ml_training/sentiment_eval_report.txt` — this is the number you'll cite
- Run the test suite once more against the real (non-sample) dataset to make
  sure nothing dataset-shape-specific broke

### What's genuinely yours vs. what's standard practice

For your own reference when discussing this project: the overall system
architecture (hybrid ranking with configurable weights, aspect-based
sentiment via keyword-anchored sentence splitting, graceful degradation
everywhere data is insufficient, session-based preference vectors reusing
the TF-IDF space) reflects real design decisions made and tested in this
build — not boilerplate copied from a tutorial. The individual building
blocks (TF-IDF, cosine similarity, logistic regression, sentence-transformers)
are standard, well-known ML techniques, which is normal and expected; the
value is in how they're combined and the honesty guarantees built around
missing data.

---

## Real data, no TMDB required (this build)

Since TMDB's API isn't reachable from this environment (and won't be needed
if your Replit quota runs out), this build ships with **real data from
public GitHub-hosted mirrors** instead of live TMDB calls:

- **`data/movies_dataset.csv`** — 2,500 real, well-known movies (Jurassic
  World, Interstellar, Guardians of the Galaxy, Star Wars, John Wick, etc.),
  reshaped from a public TMDB-sourced dataset (originally the Udacity/Kaggle
  "tmdb-movies.csv", ~10,866 movies, filtered here to `vote_count >= 50` and
  the top 2,500 by popularity). Real genres, cast, director, keywords,
  overview, rating, and popularity for every movie — this is what
  `RecommendationService` loads by default now (it prefers
  `movies_dataset.csv` over the old 10-movie `sample_movies.csv`).
- **`data/imdb_50k.csv`** — the real, full IMDB 50k Reviews dataset (not the
  40-example toy set from Week 2), pulled from a public GitHub mirror of the
  same Kaggle dataset. The sentiment model shipped in
  `backend/ml_engine/artifacts/` is now trained on this: **88.99% held-out
  test accuracy, 0.89 weighted F1** (5-fold CV: 88.67% ± 0.23%) — see
  `ml_training/sentiment_eval_report.txt`. This is a real, citable number.

### What this means for TMDB

The TMDB integration code (`tmdb_client.py`, live search, live reviews,
poster images, `seed_dataset.py`) is still fully built and will work the
moment you add a `TMDB_API_KEY` — nothing needs to change. But you don't
need it to have a fully working, demo-ready system: movie search,
recommendations, hybrid ranking, personalization, and AI discovery all run
against the real local dataset with no TMDB dependency. Sentiment on
individual movie pages will show "insufficient review data" without TMDB
(no live reviews to analyze) — that's the one feature that genuinely needs
either TMDB or another review source.

### Verified against the real dataset (in this environment)

- 2,500 movies load correctly; `/recommendations/{id}` for Interstellar
  correctly surfaces every other Christopher Nolan film in the dataset
  (The Dark Knight Rises, The Dark Knight, Inception, Batman Begins, The
  Prestige), with correct shared-cast detection (Michael Caine, Anne
  Hathaway) and correct director match
- Hybrid ranking works end-to-end: content similarity + rating + popularity
  combine correctly, sentiment component is honestly `null` (not TMDB, so
  no reviews) rather than faked
- All 23 tests pass against this real dataset (two tests had a hardcoded
  movie ID left over from the old 10-movie sample - fixed to look up a real
  ID dynamically instead)
- **One honest limitation found and worth knowing about**: the AI
  Discovery fallback mode (TF-IDF, no sentence-transformers) is noticeably
  weaker on this dataset's long free-text overviews than it was on the
  small hand-written sample - e.g. "a mind-bending sci-fi movie about
  dreams and reality" incorrectly ranked "Disaster Movie" first, purely
  from TF-IDF noise, not any real keyword match. This is exactly the
  scenario the two-tier design exists for: once `sentence-transformers`
  can actually download its model (needs real internet, unlike this
  sandbox), semantic mode will handle this correctly. Worth testing this
  specific query once you have that running, and mentioning the two-tier
  fallback design if asked about it - it's a legitimate, honest
  engineering tradeoff, not a bug.

### No Replit needed

Everything in this zip runs standalone:
```
cd backend && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
```
cd frontend && npm install && cp .env.example .env && npm run dev
```
No API keys required for the core system to work end-to-end.

---

## Movie search + "not in the database" handling (added after review)

A real gap got flagged: the frontend had no way to search for a specific
movie by title, and without TMDB there was no fallback either - a search
for anything not in the dataset would just silently fail.

### What was built

- **`GET /movies/search?q=`** - fuzzy title search against the local
  dataset. Design choices that matter:
  - Substring matches are trusted and returned as-is, even if there are
    fewer than requested - never padded with unrelated fuzzy matches just
    to fill a quota
  - Fuzzy fallback (typo tolerance) only activates when there's zero
    substring match, using a **0.75 similarity cutoff** - a lower cutoff
    (tried 0.4, then 0.6) let through false positives like "parasite"
    matching "Stargate" purely from coincidental letter overlap on short
    words, which is worse than an honest "not found"
  - Found and fixed a case-sensitivity bug in the fuzzy comparison itself
    (mixed-case titles were being compared against a lowercase query,
    artificially lowering the similarity score and causing real matches
    like "godfther" → "The Godfather" to be missed)
  - Returns an explicit `{"found": false, "message": "..."}` shape when
    nothing matches, rather than an empty array with no explanation
- **Regression test added**: `/movies/search` and `/movies/search/tmdb`
  had the *exact same route-ordering bug* fixed in `/recommendations`
  back in Week 3 - literal paths were registered after the `/{movie_id}`
  catch-all, so "search" was being swallowed as a failed int conversion
  (422) instead of ever reaching the handler. This had been silently
  broken since Week 1's `/search/tmdb` endpoint and was never caught
  because it was never actually called with a working key until now.
  `test_search_does_not_regress_to_422_routing_bug` guards against this
  happening a third time.
- **Frontend**: `SearchBar.jsx` - debounced live search in the nav bar,
  with an honest not-found state that links to AI Discovery as an
  alternative (since that searches by description, not exact title)

### Dataset coverage, widened

`movies_dataset.csv` expanded from 2,500 to **6,856 movies** (relaxed the
quality filter from `vote_count >= 50` to `>= 20`, removed the arbitrary
top-2,500-by-popularity cap). Still real, still sourced the same way.

**Known, honest limitation**: this TMDB-sourced dataset only covers movies
**up to 2015**. Anything more recent (Parasite, Dune, Oppenheimer, Barbie,
etc.) genuinely will not be found, regardless of dataset size - this isn't
a coverage gap that "more movies" fixes, it's a hard cutoff in the source
data. Worth knowing before a demo: pick pre-2016 examples, or be ready to
explain this cutoff plainly if asked. It's a legitimate, explainable
limitation of using a static dataset instead of a live TMDB connection -
not a bug.

---

## "With a real TMDB key, will search find anything?" (answered + built)

Short answer now: **yes, with a graceful two-tier fallback** - local dataset
first (instant, free), then live TMDB search if nothing matches locally and
a key is configured. This wasn't true before - it's new.

### What was actually wrong before, and what's fixed

1. **Nothing was wired together.** `/movies/search` (local) and
   `/movies/search/tmdb` (live) existed as two disconnected endpoints. The
   frontend only ever called the local one. Added a unified
   `GET /movies/search` that tries local first, falls back to TMDB
   automatically when configured, and reports which source answered
   (`"source": "local" | "tmdb" | null`).

2. **A TMDB-found movie would 404 on the detail page.** Local
   recommendations/sentiment only exist for the ~6,900 movies in the
   trained dataset - a movie found via TMDB search was never part of that.
   Added `GET /movies/{id}/tmdb-detail` for these, and the frontend now
   shows an honest "Found via TMDB — not in local recommendation model"
   banner instead of silently hiding or breaking the similar-movies/
   sentiment sections.

3. **Real bug found via testing, not guessing**: `TMDBClient` only caught
   TMDB responding with a non-200 status - it did **not** catch httpx
   raising an exception outright (timeout, DNS failure, connection refused),
   which is exactly what happens when TMDB is unreachable. That exception
   was leaking straight past every `except TMDBError` handler in the
   codebase, meaning a real key + real network hiccup would have crashed
   the search endpoint with a raw 500 instead of falling back gracefully.
   Fixed in `tmdb_client.py`'s `_get()` to wrap all httpx exceptions in
   `TMDBError`. Caught this by mocking a connection failure and watching it
   leak through, not by inspecting the code and assuming it was fine.

4. **Smaller bug, same testing process**: `TMDBClient` captured its API key
   once at import time instead of reading it dynamically, so changing the
   key after startup silently had no effect. Fixed with a property that
   reads from settings on every call.

All four of the above are now covered by `tests/test_tmdb_fallback.py`
(mocked, no real network needed) - 4 new tests, 41 total in the suite.


### Phase 2 — Persistent TMDB movie library

The app now has a durable extension layer on top of the original CSV catalogue.

- `library_movies` SQLite table stores TMDB-imported movie metadata.
- `POST /movies/{id}/library` fetches full TMDB metadata and upserts it.
- Importing the same movie twice is idempotent and refreshes metadata instead of duplicating it.
- The recommendation service rebuilds its TF-IDF/NL indexes after an import, so the imported movie becomes a normal recommendation source.
- Dynamic movies survive backend restarts because they are loaded from SQLite at startup.
- `DELETE /movies/{id}/library` removes only explicitly imported TMDB movies; the original CSV catalogue remains immutable.
- The movie detail page now exposes **Add to library / Remove from library** for TMDB-only movies.
- Once imported, the movie uses the normal hybrid recommendation + sentiment path instead of the Phase 1 temporary-query path.
- Existing local movies are never duplicated into `library_movies`.
- Test suite covers persistence, recommendation participation, idempotent re-import, and removal.

### What this means practically, once you have a real key

- Searching for something in the local ~6,900-movie dataset: instant,
  full recommendations + sentiment support (once reviews exist for it)
- Searching for something TMDB has but the original catalogue doesn't:
  found via live TMDB search. Phase 1 can recommend against the original
  catalogue; Phase 2 lets the user import the movie so it becomes a
  first-class member of the growing recommendation library.
- Searching for something neither has: honest "not found, try AI Discovery"
  message, same as before

---

## AI Discovery fallback quality — actually fixed, not just documented

Previously I'd found and *documented* that fallback-mode search was weak
(Disaster Movie, Boxing Helena ranking top for unrelated queries), but never
fixed the root cause. Fixed now:

### The fix

`extract_mood_hints()` was already extracting genre hints from the query for
the "interpreted as" display - but never actually used them to influence
ranking. Added `_apply_genre_boost()`: in fallback mode, movies sharing a
genre with the interpreted mood get boosted (×1.6), movies sharing none get
mildly penalized (×0.5). Semantic mode doesn't need this - embeddings
already capture genre-like signal implicitly.

### A real bug found while fixing this

The hint dictionary said `"Sci-Fi"` but the actual dataset's genre string is
`"Science Fiction"` - the two never matched, so every sci-fi-related hint had
been silently inert since it was written. Confirmed by checking Mulholland
Drive's real genre string (`Thriller,Drama,Mystery` - no sci-fi at all); it
had only been boosted via the `Mystery` hint. Fixed the naming mismatch and
added a regression test (`test_sci_fi_hint_matches_dataset_genre_string_exactly`)
specifically asserting `"Sci-Fi"` never appears as a hint value, so this
exact silent-failure shape can't reappear under a different genre name later.

### Also widened hint coverage

Queries with **no mood adjective at all** (just literal content words -
"astronauts stranded on mars") previously got zero interpreted genres and
fell back to pure noisy TF-IDF. Added `_detect_literal_genre_mentions()` -
a second detection layer that catches genre names mentioned directly in the
query, plus ~20 new content-word mappings (space/astronaut/alien → Science
Fiction, heist/detective → Crime, zombie/ghost → Horror, etc.).

### Before/after, same three queries

| Query | Before | After |
|---|---|---|
| "heist movie with a clever twist ending" | Disaster Movie (wrong) | Reindeer Games, The Getaway (correct) |
| "mind-bending sci-fi about dreams and reality" | Boxing Helena (wrong) | Coherence, Interstella 5555 (correct) |
| "astronauts stranded on mars trying to survive" | The Purge: Anarchy (wrong, no hints at all) | The Martian, Red Planet, Mission to Mars (correct) |

34 tests total now (2 new regression tests added for this fix specifically).

---

## Three new features (Watchlist, Feedback Loop, Confidence Score)

### 1. Watchlist
- `POST /users/watchlist`, `DELETE /users/watchlist/{id}`, `GET /users/watchlist`
- Idempotent add (adding twice doesn't duplicate) - tested
- Works for both local and TMDB-only movies (stores the title directly, no
  join needed back to the local dataset)
- Frontend: bookmark icon on movie detail pages, `/watchlist` page, nav link

### 2. Recommendation feedback loop (thumbs up/down)
- `POST /recommendations/feedback` - `{session_id, source_movie_id, recommended_movie_id, feedback: "up"|"down"}`
- Downvoted movies are excluded from that user's **future** hybrid and
  personalized recommendations - verified end-to-end (downvoted Dark Knight
  Rises while viewing The Dark Knight, confirmed it disappeared from the
  next hybrid request for that session)
- Upsert behavior - re-voting on the same pair updates rather than
  duplicates
- Frontend: thumbs icons appear on hover over a recommendation card

### 3. Recommendation confidence score
- Separate from match_percentage (how good the match is) - confidence
  answers "how much data backs this specific recommendation"
- Combines: raw content-similarity strength, whether sentiment data exists
  and how many reviews back it, and how popular/well-known the movie is
  relative to the candidate pool
- Returns `{label: "High"|"Medium"|"Low", score: 0-1}` - tested across
  strong/weak/medium scenarios to confirm the gradient behaves sensibly
- Only computed on the `/hybrid` endpoint (needs the review-count/sentiment
  data that only that endpoint fetches) - the plain `/recommendations/{id}`
  endpoint doesn't include it
- Frontend: small label under the match % on each recommendation card

### Design decision worth noting

`MovieDetail`'s "Similar movies" row now calls the **hybrid** endpoint
instead of the plain content-similarity one, specifically to get confidence
scores. This is slower per the hybrid endpoint's existing tradeoff (fetches
reviews per candidate), but without a TMDB key that fetch fails instantly
(no network attempt made), so there's no real latency cost in the current
no-key setup - only relevant once a real TMDB key is added.

39 tests total (5 new: watchlist add/list/remove, idempotent add, feedback
exclusion, invalid feedback value rejection, confidence field presence).

## Phase 4 — AI Movie Discovery

Phase 4 upgrades AI Discovery from simple mood/genre matching into an intent-aware discovery layer.
It can understand common natural-language constraints such as:
- `something like Interstellar but darker`
- `recent sci-fi movies about survival in space`
- `highly rated thrillers under 120 minutes`
- `mind-bending movies about dreams`

The backend extracts structured intent locally (genres, themes, reference title, year/decade, runtime, rating and sort preference), uses the semantic movie index when available, and supplements the catalogue with live TMDB search/discovery when useful. Results are merged and re-ranked rather than blindly returning either source.

No LLM API key is required for Phase 4. The semantic model remains optional; TF-IDF is the graceful fallback.

## Offline recommendation evaluation

Run the accuracy benchmark with:

```bash
cd backend
PYTHONPATH=. python evaluation/benchmark.py --k 10 --output ../reports
```

The benchmark reports Precision@K, Recall@K, NDCG@K, MAP@K, MRR, 95% bootstrap confidence intervals, and catalogue coverage. The labels are an **independent metadata proxy** (genre/director/cast overlap), so these numbers are useful for regression tracking but are **not human-judged production accuracy**. Use a human relevance set or implicit user-feedback log before making claims about real-world recommendation quality.

## Large Recommendation Benchmark

The project includes a reproducible, stratified offline benchmark over 500 catalogue movies by default. It evaluates the current recommender against the legacy metadata-only TF-IDF baseline at K=5, 10, and 20.

Run:

```bash
PYTHONPATH=backend python backend/evaluation/benchmark.py --seed-count 500 --ks 5 10 20
```

The benchmark samples across rating/popularity strata, reports Precision, Recall, NDCG, MAP, and MRR with bootstrap 95% confidence intervals, and writes JSON/Markdown reports under `reports/benchmark_500/`.

> The relevance labels are an offline metadata proxy (genre/director/cast overlap), not human judgments. Treat these metrics as regression signals, not production accuracy claims.

## 500+ movie evaluation benchmark

The recommender now includes a reproducible **500-movie stratified benchmark** over the full 6,856-movie catalogue. It evaluates the current engine against the legacy metadata-only baseline at K=5, 10, and 20 with Precision, Recall, NDCG, MAP, MRR, bootstrap 95% confidence intervals, and catalogue coverage.

Run it with:

```bash
PYTHONPATH=backend python backend/evaluation/benchmark.py --seed-count 500 --ks 5 10 20 --output reports/benchmark_500
```

## Human-rated benchmark

A human annotation set is generated from **40 reproducibly sampled seed movies** and the union of the current and baseline top-10 recommendations. The included template contains **568 blank human judgments**. Each candidate is rated once on a 0-3 relevance scale.

Create or regenerate the annotation set:

```bash
PYTHONPATH=backend python backend/evaluation/human_benchmark.py create --seed-count 40 --k 10
```

After a real human fills the `rating` column, evaluate it with:

```bash
PYTHONPATH=backend python backend/evaluation/human_benchmark.py evaluate --ratings reports/human_benchmark/human_rating_template.csv
```

The evaluator reports mean relevance, Precision@10, graded NDCG@10, MAP@10, MRR, paired current-vs-baseline win/tie/loss, and optional Cohen's kappa when two annotators are supplied. **Do not use fabricated or model-generated labels as human ground truth.**
