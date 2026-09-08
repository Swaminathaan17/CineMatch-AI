import { motion } from "framer-motion";
import MovieCard from "./MovieCard";

export default function MovieGrid({
  movies,
  matchScores,
  confidenceScores,
  onFeedback,
  onLikeToggle,
  likedIds,
  onWatchlistToggle,
  watchlistIds,
  emptyTitle = "Nothing here yet",
  emptyMessage = "",
}) {
  if (!movies || movies.length === 0) {
    return (
      <div className="py-16 text-center">
        <p className="font-display text-lg text-ivory mb-1">{emptyTitle}</p>
        {emptyMessage && <p className="text-smoke text-sm">{emptyMessage}</p>}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {movies.map((m) => {
        const key = `${m.source}-${m.id}`;
        return (
          <motion.div
            key={key}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: Math.min((movies.indexOf(m) % 10) * 0.03, 0.3) }}
            className="min-w-0"
          >
            <MovieCard
              movie={m}
              matchPercentage={
                matchScores != null && matchScores[m.id] != null
                  ? matchScores[m.id]
                  : m.match_percentage != null
                    ? m.match_percentage
                    : null
              }
              confidence={confidenceScores ? confidenceScores[m.id] : m.confidence || null}
              onFeedback={onFeedback}
              liked={likedIds ? likedIds.has(m.id) : undefined}
              onLikeToggle={onLikeToggle}
              inWatchlist={watchlistIds ? watchlistIds.has(m.id) : undefined}
              onWatchlistToggle={onWatchlistToggle}
            />
          </motion.div>
        );
      })}
    </div>
  );
}