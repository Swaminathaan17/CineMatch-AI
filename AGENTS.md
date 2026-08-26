# AGENTS.md

## Project overview

"Reel" — a movie recommendation engine with a React 19 frontend (Vite + Tailwind) and a Python/FastAPI backend. Two independent deployables, not a monorepo workspace. No TypeScript on the frontend.

## Quick commands

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000   # dev server
python -m pytest tests/ -v                           # run all tests
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env
npm run dev       # Vite dev server on localhost:5173
npm run build     # production build
npm run lint      # oxlint (no eslint)
```

### Benchmarks & training
```bash
# Offline accuracy benchmark (500-movie stratified)
PYTHONPATH=backend python backend/evaluation/benchmark.py --seed-count 500 --ks 5 10 20

# Train sentiment model (IMDB 50k, produces ml_engine/artifacts/*.pkl)
python ml_training/train_sentiment.py

# Seed dataset from TMDB (requires TMDB_API_KEY)
python ml_training/seed_dataset.py --pages 20

# Human benchmark
PYTHONPATH=backend python backend/evaluation/human_benchmark.py create --seed-count 40 --k 10
```

## Architecture

### Backend (`backend/`)

- **Entry point:** `app/main.py` — FastAPI app, registers 5 routers under `/movies`, `/recommendations`, `/sentiment`, `/users`, `/discovery`
- **Config:** `app/config.py` — pydantic-settings, reads `.env`. `TMDB_API_KEY` is optional (system works without it). `DATABASE_URL` defaults to `sqlite:///./movie_rec.db`
- **DB:** SQLAlchemy + SQLite (`app/db/session.py`). Tables created via `Base.metadata.create_all` on startup — no migrations. `movie_rec.db` also exists at project root (ignore it, backend creates its own in the working dir)
- **Singleton pattern:** `recommendation_service` and `ai_discovery_service` are module-level singletons instantiated at import time. They lazy-load on first call. After library mutations (add/remove TMDB movie), call `recommendation_service.refresh()` to rebuild in-memory indexes

### Frontend (`frontend/`)

- **Entry:** `src/App.jsx` — BrowserRouter with 6 routes. No auth guards
- **Pages:** Landing, Home (personalized feed), MovieDetail, AIDiscovery, Watchlist
- **API layer:** `src/services/api.js` — single file, all fetch calls. Generates a `reel_session_id` UUID in localStorage on first visit (session-based auth, no passwords)
- **Styling:** Tailwind with custom design tokens in `tailwind.config.js` (colors: void, curtain, gold, ivory, smoke, panel). Fonts: Fraunces (display), Inter (body), JetBrains Mono (mono)
- **No TypeScript, no eslint** — uses `oxlint` for linting

### Recommendation pipeline (end-to-end flow)

```
User search/query
  → Query understanding (nl_query_parser.py or discovery_intent.py)
    → Mood/genre hints extracted, semantic or TF-IDF mode selected
  → Candidate retrieval (content_similarity.py)
    → Dual-vector TF-IDF: metadata soup (genres×4, director×4, cast×2, keywords) + overview text
    → 72% metadata / 28% overview blend
  → Ranking (hybrid_ranker.py)
    → Multi-signal: content_similarity(0.42), preference_match(0.24), sentiment(0.12), rating(0.12), popularity(0.06), recency(0.04)
    → Missing signals have their weight redistributed proportionally
  → Diversity (MMR re-ranking, λ=0.84) — removes near-duplicate results
  → Response with match_percentage, score_breakdown, reasons, confidence
```

### Content similarity engine (`ml_engine/content_similarity.py`)

- Two separate TF-IDF vectorizers: `meta_vectorizer` (structured metadata) and `overview_vectorizer` (plot text)
- Soup tokens are space-joined and lowercase. Cast limited to top 5. Metadata fields are repeated for weighting (genres×4, director×4)
- `get_similar_to_external()` uses the same vector space for TMDB-only movies without inserting them into the dataset

### NL search (`ml_engine/nl_query_parser.py`)

- **Preferred mode:** sentence-transformers (all-MiniLM-L6-v2) — downloads model weights on first use
- **Fallback mode:** TF-IDF cosine similarity over overviews. In fallback mode only, mood-genre hints get ×1.6 boost / ×0.5 penalty on non-matching genres
- Mode is reported in every result (`"semantic"` or `"fallback"`)

### AI Discovery (`app/services/ai_discovery_service.py`)

- Parses intent via `discovery_intent.parse_intent()` (regex-based, no LLM)
- Merges local NL search results with live TMDB data (discover/search/recommendations endpoints)
- Re-ranks with: match_score×0.58 + word_overlap×0.15 + genre_bonus×0.12 + rating×0.12 + popularity×0.05 − exclusion_penalty×0.35

### Personalization (`ml_engine/personalization.py`)

- Builds a user profile vector as weighted average of liked movies' TF-IDF vectors (reuses the same vector space as content_similarity)
- Negative feedback pushes profile away (×0.45 subtraction)
- Explicit genre preferences get ×1.5 boost on matching TF-IDF features
- Cold start → falls back to trending (rating×0.6 + popularity×0.4)

### Sentiment analysis

- **Model:** TF-IDF (5000 features) + LogisticRegression, trained on IMDB 50k. Artifacts in `ml_engine/artifacts/sentiment_model.pkl` and `tfidf_vectorizer.pkl`
- **Aspects:** 6 aspects (story, acting, direction, visuals, music, pacing) anchored by keyword dictionaries. Aspects with fewer than `min_reviews_for_sentiment` anchored sentences → `"insufficient_data"`
- **Reviews source:** TMDB `/movie/{id}/reviews` endpoint, fetched via `review_fetcher.py`

## Critical gotchas

### FastAPI route ordering
Literal-path routes (`/search`, `/search/tmdb`) MUST be registered before `/{movie_id}` catch-all routes. FastAPI/Starlette matches in registration order — "search" would otherwise be swallowed as a failed int conversion (422). This bug has occurred in both `movies.py` and `recommendations.py` routers.

### Lazy loading
`recommendation_service._load()` is called implicitly by most public methods. The first call loads the CSV dataset + DB library movies, builds TF-IDF indexes. This is slow. Tests that hit the API for the first time pay this cost.

### Dataset
Primary dataset is `data/movies_dataset.csv` (TMDB-seeded, ~2015 cutoff). `data/sample_movies.csv` is a fallback. Movie IDs are TMDB numeric IDs. The CSV is immutable — new movies come from `LibraryMovie` DB table and are merged at load time, deduplicating by ID.

### Database
Two `movie_rec.db` files can exist: one at project root (stale), one in `backend/` (active, created by uvicorn working dir). The backend creates its own based on `DATABASE_URL` config. Don't assume the root one is current.

### Hybrid endpoint is expensive
`/recommendations/{id}/hybrid` fetches TMDB reviews for every candidate (up to `candidate_pool=40`) concurrently via `asyncio.gather`. Slower than the plain content-only endpoint.

### No conftest.py
Tests use `sys.path.insert(0, ...)` for module resolution. Each test file sets up its own paths.

## Testing

- **Framework:** pytest (no special config)
- **Run:** `cd backend && python -m pytest tests/ -v`
- **Pattern:** module-scoped fixtures for TestClient startup, dynamic movie ID resolution (no hardcoded IDs), monkeypatching for sentiment model stubs, mocked httpx for TMDB tests
- **11 test files, 41+ tests** covering: API integration, content similarity, hybrid ranker, aspect sentiment, watchlist/feedback, TMDB fallback, NL query genre boost, discovery intent, Phase 5 accuracy, benchmark helpers, human benchmark

## Environment variables

| Variable | Where | Default | Required |
|---|---|---|---|
| `TMDB_API_KEY` | `backend/.env` | `""` (disabled) | No — search/AI Discovery degrade gracefully |
| `DATABASE_URL` | `backend/.env` | `sqlite:///./movie_rec.db` | No |
| `VITE_API_URL` | `frontend/.env` | `http://localhost:8000` | No (dev default) |

## Files that matter most

- `backend/app/services/recommendation_service.py` — central orchestrator, all recommendation logic flows through this singleton
- `backend/ml_engine/content_similarity.py` — dual-vector TF-IDF engine, the core retrieval layer
- `backend/ml_engine/hybrid_ranker.py` — multi-signal scoring with weight redistribution
- `backend/ml_engine/nl_query_parser.py` — two-tier NL search (semantic + TF-IDF fallback)
- `backend/app/services/ai_discovery_service.py` — AI Discovery orchestration (local + TMDB merge)
- `backend/ml_engine/weights.yaml` — hybrid ranker weights (edit here, not in code)
- `frontend/src/services/api.js` — all frontend API calls in one file
- `backend/app/routers/movies.py` — route ordering matters, read the NOTE comment at line 20
