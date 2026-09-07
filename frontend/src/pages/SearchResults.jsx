import { useEffect, useState } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { api } from "../services/api";
import { enrichMovies } from "../services/movieEnrichment";
import MovieGrid from "../components/movie/MovieGrid";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";
import EmptyState from "../components/ui/EmptyState";
import ErrorState from "../components/ui/ErrorState";

export default function SearchResults() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const query = params.get("q") || "";
  const live = params.get("live") === "1";

  const [results, setResults] = useState([]);
  const [source, setSource] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retryToken, setRetryToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!query.trim()) {
        setResults([]);
        setLoading(false);
        return;
      }
      setLoading(true);
      setError("");
      setResults([]);
      setSource(null);
      try {
        let res;
        if (live) {
          const raw = await api.searchTmdb(query);
          const mapped = (raw.results || []).map((r) => ({
            id: r.id,
            title: r.title || "",
            source: "tmdb",
            poster_path: r.poster_path || null,
            backdrop_path: r.backdrop_path || null,
            release_date: r.release_date || "",
            vote_average: r.vote_average || 0,
            overview: r.overview || "",
          }));
          res = { found: true, source: "tmdb", results: mapped };
        } else {
          res = await api.searchMovies(query, 12);
        }
        if (cancelled) return;
        const enriched = await enrichMovies(res.results || []);
        if (cancelled) return;
        setResults(enriched);
        setSource(res.source);
      } catch {
        if (!cancelled) setError("We couldn't reach the search service right now.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [query, live, retryToken]);

  return (
    <div className="min-h-screen pt-12 pb-24 px-6 md:px-12">
      <div className="max-w-3xl mb-10">
        <div className="flex items-center gap-2 mb-2">
          <Search size={14} className="text-gold" aria-hidden="true" />
          <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase">
            Search
          </p>
        </div>
        <h1 className="font-display text-3xl text-ivory mb-2">
          Results for “{query}”
        </h1>
        <p className="text-smoke text-sm">
          {source === "tmdb"
            ? "Found live on TMDB — open a title to preview it, or add it to your library."
            : source === "local"
              ? "From your recommendation library."
              : " "}
        </p>
      </div>

      {loading ? (
        <LoadingSkeleton count={10} />
      ) : error ? (
        <ErrorState
          title="Search unavailable"
          message={error}
          onRetry={() => setRetryToken((t) => t + 1)}
        />
      ) : results.length === 0 ? (
        <EmptyState
          title="Nothing matched"
          message={`We couldn't find “${query}” in your library or on TMDB. Try different wording, or let AI Discovery figure out what you're after.`}
          action={
            <button
              onClick={() => navigate(`/search?q=${encodeURIComponent(query)}&live=1`)}
              className="mt-5 px-5 py-2.5 rounded-sm bg-curtain hover:bg-curtain-dim text-ivory font-medium transition-colors"
            >
              Search live TMDB
            </button>
          }
        />
      ) : (
        <MovieGrid movies={results} />
      )}
    </div>
  );
}