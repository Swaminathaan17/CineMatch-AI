import { useEffect, useState } from "react";
import { api } from "../services/api";
import FilmstripRow from "../components/movie/FilmstripRow";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

export default function Home() {
  const [allMovies, setAllMovies] = useState([]);
  const [personalized, setPersonalized] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const movies = await api.getTrending(20);
        setAllMovies(movies);

        const rec = await api.getPersonalized(10);
        setPersonalized(rec);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="text-center">
          <p className="font-display text-xl text-ivory mb-2">Can't reach the API</p>
          <p className="text-smoke text-sm">{error}</p>
          <p className="text-smoke text-xs mt-4">
            Check that the backend is running and VITE_API_URL is set correctly.
          </p>
        </div>
      </div>
    );
  }

  // personalized.results shape differs slightly (id/title/preference_match_score
  // vs the plain movie list) - map to what FilmstripRow expects
  const personalizedMovies = personalized?.results?.map((r) => ({
    id: r.id,
    title: r.title,
    poster_path: r.poster_path || null,
    release_date: r.release_date || "",
    source: r.source || "local",
    reasons: r.reasons || [],
  }));
  const personalizedScores = personalized?.results?.reduce((acc, r) => {
    if (r.match_percentage != null) {
      acc[r.id] = r.match_percentage;
    } else if (r.preference_match_score != null) {
      acc[r.id] = Math.round(r.preference_match_score * 100);
    }
    return acc;
  }, {});

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
          <LoadingSkeleton />
          <LoadingSkeleton />
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
            />
          ) : null}
          <FilmstripRow title="Trending Now" movies={allMovies} />
        </>
      )}
    </div>
  );
}
