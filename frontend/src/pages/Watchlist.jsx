import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Bookmark } from "lucide-react";
import { api } from "../services/api";
import { enrichMovies } from "../services/movieEnrichment";
import MovieGrid from "../components/movie/MovieGrid";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";
import EmptyState from "../components/ui/EmptyState";

export default function Watchlist() {
  const [items, setItems] = useState(null);
  const [watchlistIds, setWatchlistIds] = useState(new Set());

  useEffect(() => {
    let cancelled = false;
    api
      .getWatchlist()
      .then(async (r) => {
        const ids = new Set((r.results || []).map((m) => Number(m.id)));
        if (cancelled) return;
        setWatchlistIds(ids);
        const enriched = await enrichMovies(r.results || []);
        if (!cancelled) setItems(enriched);
      })
      .catch(() => !cancelled && setItems([]));
    return () => {
      cancelled = true;
    };
  }, []);

  const removeFromWatchlist = async (id) => {
    await api.removeFromWatchlist(id);
    setItems((prev) => prev.filter((m) => Number(m.id) !== Number(id)));
    setWatchlistIds((prev) => {
      const next = new Set(prev);
      next.delete(Number(id));
      return next;
    });
  };

  return (
    <div className="min-h-screen pt-16 pb-24 px-6 md:px-12">
      <div className="mb-10">
        <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase mb-2">
          Your List
        </p>
        <h1 className="font-display text-3xl text-ivory">Watchlist</h1>
      </div>

      {items === null ? (
        <LoadingSkeleton count={10} />
      ) : items.length === 0 ? (
        <EmptyState
          icon={Bookmark}
          title="Nothing saved yet"
          message="Tap the bookmark icon on a movie page to add it here."
          action={
            <Link
              to="/home"
              className="mt-5 px-5 py-2.5 rounded-sm bg-curtain hover:bg-curtain-dim text-ivory font-medium transition-colors"
            >
              Explore movies
            </Link>
          }
        />
      ) : (
        <MovieGrid
          movies={items}
          onWatchlistToggle={(id) => removeFromWatchlist(id)}
          watchlistIds={watchlistIds}
        />
      )}
    </div>
  );
}