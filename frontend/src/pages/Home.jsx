import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../services/api";
import { enrichMovies } from "../services/movieEnrichment";
import FilmstripRow from "../components/movie/FilmstripRow";
import MovieGrid from "../components/movie/MovieGrid";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";
import ErrorState from "../components/ui/ErrorState";

const TRENDING_COUNT = 50;
const MAX_ATTEMPTS = 3;
const RETRY_DELAYS_MS = [800, 1600];

export default function Home() {
  const [allMovies, setAllMovies] = useState([]);
  const [personalized, setPersonalized] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [likedIds, setLikedIds] = useState(new Set());
  const [watchlistIds, setWatchlistIds] = useState(new Set());
  const loadGenRef = useRef(0);

  const load = async (attempt = 1) => {
    const gen = ++loadGenRef.current;
    setLoading(true);
    setError(null);
    try {
      const movies = await enrichMovies(await api.getTrending(TRENDING_COUNT), {
        fillMissingPosters: true,
      });
      const rec = await api.getPersonalized(10).catch(() => null);
      const prefs = await api.getPreferences().catch(() => null);
      const wl = await api.getWatchlist().catch(() => ({ results: [] }));
      if (gen !== loadGenRef.current) return;

      setAllMovies(movies);
      if (rec) {
        setPersonalized({ ...rec, results: await enrichMovies(rec.results || []) });
      } else {
        setPersonalized(null);
      }
      if (prefs?.liked_movie_ids) setLikedIds(new Set(prefs.liked_movie_ids));
      setWatchlistIds(new Set((wl.results || []).map((m) => Number(m.id))));
      setLoading(false);
    } catch (e) {
      if (gen !== loadGenRef.current) return;
      if (attempt < MAX_ATTEMPTS) {
        const delayMs = RETRY_DELAYS_MS[attempt - 1] ?? 2000;
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        if (gen !== loadGenRef.current) return;
        await load(attempt + 1);
        return;
      }
      setError(e.message);
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const personalizedMovies = useMemo(() => {
    return (personalized?.results || []).map((r) => ({
      id: r.id,
      title: r.title,
      poster_path: r.poster_path || null,
      release_date: r.release_date || "",
      source: r.source || "local",
      reasons: r.reasons || [],
      explanation: r.explanation || null,
      runtime: r.runtime ?? null,
      vote_average: r.vote_average ?? null,
      genres: r.genres ?? "",
    }));
  }, [personalized]);

  const personalizedScores = useMemo(() => {
    return (personalized?.results || []).reduce((acc, r) => {
      if (r.match_percentage != null) {
        acc[r.id] = r.match_percentage;
      } else if (r.preference_match_score != null) {
        acc[r.id] = Math.round(r.preference_match_score * 100);
      }
      return acc;
    }, {});
  }, [personalized]);

  const toggleLike = async (id) => {
    await api.likeMovie(id);
    setLikedIds((prev) => {
      const next = new Set(prev);
      next.add(Number(id));
      return next;
    });
  };

  const toggleWatchlist = async (id, title) => {
    const numericId = Number(id);
    if (watchlistIds.has(numericId)) {
      await api.removeFromWatchlist(numericId);
      setWatchlistIds((prev) => {
        const next = new Set(prev);
        next.delete(numericId);
        return next;
      });
    } else {
      await api.addToWatchlist(numericId, title);
      setWatchlistIds((prev) => new Set(prev).add(numericId));
    }
  };

  if (error) {
    return (
      <div className="min-h-screen pt-24 px-6">
        <ErrorState
          title="Can't reach the API"
          message="Check that the backend is running and VITE_API_URL is set correctly."
          onRetry={load}
        />
      </div>
    );
  }

  return (
    <div className="min-h-screen pt-10 pb-20">
      <div className="px-6 md:px-12 mb-10">
        <h1 className="font-display text-3xl text-ivory">Welcome back</h1>
        <p className="text-smoke text-sm mt-1">
          Like a few movies below and your "For You" row will start adapting.
        </p>
      </div>

      {loading ? (
        <>
          <LoadingSkeleton variant="row" />
          <LoadingSkeleton variant="grid" count={10} />
        </>
      ) : (
        <>
          {personalized?.mode === "trending" && (
            <div className="px-6 md:px-12 mb-2">
              <p className="text-smoke text-xs font-mono uppercase tracking-wider">
                {personalized.reason}
              </p>
            </div>
          )}
          {personalized?.mode === "personalized" ? (
            <FilmstripRow
              title="For You"
              movies={personalizedMovies}
              matchScores={personalizedScores}
              onLikeToggle={toggleLike}
              likedIds={likedIds}
              onWatchlistToggle={toggleWatchlist}
              watchlistIds={watchlistIds}
            />
          ) : null}
          <section className="mb-10">
            <div className="flex items-center gap-3 mb-4 px-6 md:px-12">
              <h2 className="font-display text-xl md:text-2xl text-ivory">Trending Now</h2>
              <div className="h-px flex-1 bg-gradient-to-r from-white/10 to-transparent" />
            </div>
            <div className="px-6 md:px-12">
              <MovieGrid
                movies={allMovies}
                onLikeToggle={toggleLike}
                likedIds={likedIds}
                onWatchlistToggle={toggleWatchlist}
                watchlistIds={watchlistIds}
              />
            </div>
          </section>
        </>
      )}
    </div>
  );
}