const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function getSessionId() {
  let id = localStorage.getItem("reel_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("reel_session_id", id);
  }
  return id;
}

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  sessionId: getSessionId(),

  listMovies: () => request("/movies/"),
  getMovie: (id) => request(`/movies/${id}`),
  getTmdbDetail: (id) => request(`/movies/${id}/tmdb-detail`),
  addToLibrary: (id) => request(`/movies/${id}/library`, { method: "POST" }),
  removeFromLibrary: (id) => request(`/movies/${id}/library`, { method: "DELETE" }),
  searchTmdb: (q) => request(`/movies/search/tmdb?q=${encodeURIComponent(q)}`),

  getRecommendations: (movieId, topN = 10) =>
    request(`/recommendations/${movieId}?top_n=${topN}`),
  getTmdbRecommendations: (movieId, topN = 10) =>
    request(`/recommendations/tmdb/${movieId}?top_n=${topN}`),
  getHybridRecommendations: (movieId, topN = 10) =>
    request(`/recommendations/${movieId}/hybrid?top_n=${topN}&session_id=${getSessionId()}`),
  getPersonalized: (topN = 10) =>
    request(`/recommendations/personalized?session_id=${getSessionId()}&top_n=${topN}`),

  getSentiment: (movieId) => request(`/sentiment/${movieId}`),

  searchMovies: (q, topN = 8) =>
    request(`/movies/search?q=${encodeURIComponent(q)}&top_n=${topN}`),

  discoverByMood: (query, topN = 10) =>
    request(`/discovery/query?top_n=${topN}`, {
      method: "POST",
      body: JSON.stringify({ query }),
    }),

  likeMovie: (movieId) =>
    request("/users/interactions", {
      method: "POST",
      body: JSON.stringify({
        session_id: getSessionId(),
        movie_id: movieId,
        interaction_type: "liked",
      }),
    }),
  getPreferences: () => request(`/users/preferences?session_id=${getSessionId()}`),

  addToWatchlist: (movieId, movieTitle) =>
    request("/users/watchlist", {
      method: "POST",
      body: JSON.stringify({ session_id: getSessionId(), movie_id: movieId, movie_title: movieTitle }),
    }),
  removeFromWatchlist: (movieId) =>
    request(`/users/watchlist/${movieId}?session_id=${getSessionId()}`, { method: "DELETE" }),
  getWatchlist: () => request(`/users/watchlist?session_id=${getSessionId()}`),

  submitRecommendationFeedback: (sourceMovieId, recommendedMovieId, feedback) =>
    request("/recommendations/feedback", {
      method: "POST",
      body: JSON.stringify({
        session_id: getSessionId(),
        source_movie_id: sourceMovieId,
        recommended_movie_id: recommendedMovieId,
        feedback,
      }),
    }),
};
