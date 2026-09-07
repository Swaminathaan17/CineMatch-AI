export function formatRuntime(minutes) {
  const n = Number(minutes);
  if (!Number.isFinite(n) || n <= 0) return "";
  const h = Math.floor(n / 60);
  const m = Math.round(n % 60);
  if (h > 0 && m > 0) return `${h}h ${m}m`;
  if (h > 0) return `${h}h`;
  return `${m}m`;
}

export function formatReleaseYear(releaseDate) {
  if (!releaseDate) return "";
  const m = String(releaseDate).trim().match(/^(\d{4})/);
  return m ? m[1] : "";
}

export function formatRating(voteAverage) {
  const n = Number(voteAverage);
  if (!Number.isFinite(n) || n <= 0) return "";
  return n.toFixed(1);
}

export function normalizeGenres(genres) {
  if (Array.isArray(genres)) return genres.filter(Boolean).map((g) => String(g).trim());
  if (!genres) return [];
  return String(genres)
    .split(",")
    .map((g) => g.trim())
    .filter(Boolean);
}

export function firstGenre(genres) {
  return normalizeGenres(genres)[0] || "";
}