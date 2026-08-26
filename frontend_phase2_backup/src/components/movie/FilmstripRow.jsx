import MovieCard from "./MovieCard";

export default function FilmstripRow({ title, movies, matchScores, confidenceScores, onFeedback }) {
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
            matchPercentage={matchScores ? matchScores[m.id] : null}
            confidence={confidenceScores ? confidenceScores[m.id] : null}
            onFeedback={onFeedback}
          />
        ))}
      </div>
    </section>
  );
}
