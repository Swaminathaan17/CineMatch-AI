import MovieCard from "./MovieCard";

export default function FilmstripRow({
  title,
  movies,
  matchScores,
  confidenceScores,
  onFeedback,
  onLikeToggle,
  likedIds,
  onWatchlistToggle,
  watchlistIds,
}) {
  if (!movies || movies.length === 0) return null;

  return (
    <section className="mb-10">
      <div className="flex items-center gap-3 mb-4 px-6 md:px-12">
        <h2 className="font-display text-xl md:text-2xl text-ivory">{title}</h2>
        <div className="h-px flex-1 bg-gradient-to-r from-white/10 to-transparent" />
      </div>
      <div className="filmstrip flex gap-4 overflow-x-auto px-6 md:px-12 pb-4">
        {movies.map((m) => (
          <MovieCard
            key={m.id}
            movie={m}
            className="shrink-0 w-44 sm:w-52 md:w-56"
            matchPercentage={matchScores ? matchScores[m.id] : null}
            confidence={confidenceScores ? confidenceScores[m.id] : null}
            onFeedback={onFeedback}
            liked={likedIds ? likedIds.has(m.id) : undefined}
            onLikeToggle={onLikeToggle}
            inWatchlist={watchlistIds ? watchlistIds.has(m.id) : undefined}
            onWatchlistToggle={onWatchlistToggle}
          />
        ))}
      </div>
    </section>
  );
}