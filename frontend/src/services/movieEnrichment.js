import { api } from "./api";

let cataloguePromise = null;

function loadCatalogue() {
  if (!cataloguePromise) {
    cataloguePromise = api.listMovies().catch(() => []);
  }
  return cataloguePromise;
}

export async function enrichMovies(movies) {
  const list = Array.isArray(movies)
    ? movies
    : movies?.results || [];
  if (!list.length) return list;

  const catalogue = await loadCatalogue();
  const byId = new Map(catalogue.map((m) => [Number(m.id), m]));

  return list.map((m) => {
    const id = Number(m.id);
    const cached = byId.get(id);
    if (!cached) return m;
    return {
      ...m,
      poster_path: m.poster_path || cached.poster_path || null,
      backdrop_path: m.backdrop_path || cached.backdrop_path || null,
      release_date: m.release_date || cached.release_date || "",
      runtime: m.runtime ?? cached.runtime ?? null,
      vote_average: m.vote_average ?? cached.vote_average ?? 0,
      genres: m.genres ?? cached.genres ?? "",
    };
  });
}