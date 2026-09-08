import { api } from "./api";

const MAX_POSTER_BACKFILL = 50;
const BACKFILL_BATCH = 5;

let cataloguePromise = null;
const posterCache = new Map();
const posterFetchState = new Map();

function loadCatalogue() {
  if (!cataloguePromise) {
    cataloguePromise = api.listMovies().catch(() => []);
  }
  return cataloguePromise;
}

function fetchPosterOnce(id) {
  if (posterCache.has(id)) return posterCache.get(id);
  if (posterFetchState.has(id)) return posterFetchState.get(id);
  const pending = api
    .getTmdbDetail(id)
    .then((d) => d?.poster_path || null)
    .catch(() => null);
  posterFetchState.set(id, pending);
  pending.then((value) => {
    posterCache.set(id, value);
    posterFetchState.delete(id);
  });
  return pending;
}

export async function enrichMovies(movies, { fillMissingPosters = false } = {}) {
  const list = Array.isArray(movies)
    ? movies
    : movies?.results || [];
  if (!list.length) return list;

  const catalogue = await loadCatalogue();
  const byId = new Map(catalogue.map((m) => [Number(m.id), m]));

  const enriched = list.map((m) => {
    const id = Number(m.id);
    const cached = byId.get(id);
    const poster =
      m.poster_path ||
      posterCache.get(id) ||
      cached?.poster_path ||
      null;
    if (poster) posterCache.set(id, poster);
    return {
      ...m,
      poster_path: poster,
      backdrop_path: m.backdrop_path || cached?.backdrop_path || null,
      release_date: m.release_date || cached?.release_date || "",
      runtime: m.runtime ?? cached?.runtime ?? null,
      vote_average: m.vote_average ?? cached?.vote_average ?? 0,
      genres: m.genres ?? cached?.genres ?? "",
    };
  });

  if (fillMissingPosters) {
    const missing = enriched.filter((m) => !m.poster_path && Number.isFinite(Number(m.id)));
    const batch = missing.slice(0, MAX_POSTER_BACKFILL);
    for (let i = 0; i < batch.length; i += BACKFILL_BATCH) {
      const chunk = batch.slice(i, i + BACKFILL_BATCH);
      const posters = await Promise.all(chunk.map((m) => fetchPosterOnce(Number(m.id))));
      posters.forEach((poster, j) => {
        if (poster) chunk[j].poster_path = poster;
      });
    }
  }

  return enriched;
}