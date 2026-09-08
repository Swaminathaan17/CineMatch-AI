const TMDB_BASE = "https://image.tmdb.org/t/p/";

export const TMDB_POSTER_SIZE = "w500";
export const TMDB_BACKDROP_SIZE = "w1280";

function isValidPath(path) {
  return typeof path === "string" && path.trim().length > 0;
}

export function tmdbImageUrl(path, size) {
  if (!isValidPath(path)) return null;

  const trimmed = path.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed;

  if (!trimmed.startsWith("/")) return null;

  return `${TMDB_BASE}${size}${trimmed}`;
}

export function posterUrl(path, size = TMDB_POSTER_SIZE) {
  return tmdbImageUrl(path, size);
}

export function backdropUrl(path, size = TMDB_BACKDROP_SIZE) {
  return tmdbImageUrl(path, size);
}

export function thumbnailUrl(path) {
  return tmdbImageUrl(path, "w92");
}