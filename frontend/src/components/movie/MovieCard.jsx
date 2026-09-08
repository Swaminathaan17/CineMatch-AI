import { useState } from "react";
import { Link } from "react-router-dom";
import { Star, ThumbsUp, ThumbsDown, Bookmark, BookmarkCheck } from "lucide-react";
import { posterUrl } from "../../utils/tmdbImage";
import {
  formatReleaseYear,
  formatRuntime,
  formatRating,
  firstGenre,
} from "../../utils/format";
import PosterFallback from "./PosterFallback";

const CONFIDENCE_COLOR = {
  High: "text-gold-soft",
  Medium: "text-smoke",
  Low: "text-curtain",
};

export default function MovieCard({
  movie,
  matchPercentage,
  matchLabel,
  confidence,
  onFeedback,
  liked,
  onLikeToggle,
  inWatchlist,
  onWatchlistToggle,
  className = "",
  showFooter = true,
}) {
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  const handleFeedback = (e, value) => {
    e.preventDefault();
    e.stopPropagation();
    setFeedbackGiven(value);
    onFeedback?.(movie.id, value);
  };

  const stop = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleLike = (e) => {
    stop(e);
    onLikeToggle?.(movie.id);
  };

  const handleWatchlist = (e) => {
    stop(e);
    onWatchlistToggle?.(movie.id, movie.title);
  };

  const detailPath =
    movie.source === "tmdb" || movie.source === "tmdb_external"
      ? `/movie/tmdb/${movie.id}`
      : `/movie/${movie.id}`;

  const posterSrc = posterUrl(movie.poster_path);
  const year = formatReleaseYear(movie.release_date);
  const runtime = formatRuntime(movie.runtime);
  const rating = formatRating(movie.vote_average);
  const genre = firstGenre(movie.genres);
  const hasMeta = year || runtime || rating;
  const showControls = onFeedback || onLikeToggle || onWatchlistToggle;

  return (
    <Link
      to={detailPath}
      className={`group block focus:outline-none ${className}`}
      aria-label={movie.title ? `${movie.title}${year ? ` (${year})` : ""}` : undefined}
    >
      <div className="movie-card rounded-lg overflow-hidden bg-panel border border-white/5 hover:border-gold/40 transition-colors relative">
        <div className="relative aspect-[2/3] overflow-hidden">
          {posterSrc ? (
            <img
              src={posterSrc}
              alt=""
              loading="lazy"
              className="w-full h-full object-cover group-hover:scale-[1.06] transition-transform duration-[900ms] ease-out"
            />
          ) : (
            <PosterFallback title={movie.title} />
          )}

          <div className="absolute inset-0 bg-gradient-to-t from-void/90 via-void/10 to-transparent" />

          {matchPercentage != null && (
            <div className="absolute top-2 right-2 bg-void/80 backdrop-blur-sm px-2 py-0.5 rounded-full border border-white/10">
              <span className="font-mono text-xs text-gold-soft">
                {matchLabel || `${matchPercentage}%`}
              </span>
            </div>
          )}

          {showControls && (
            <div className="absolute top-2 left-2 flex gap-1.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity">
              {onFeedback && (
                <>
                  <button
                    onClick={(e) => handleFeedback(e, "up")}
                    className={`p-1.5 rounded-full backdrop-blur-sm border transition-colors ${
                      feedbackGiven === "up"
                        ? "bg-gold text-void border-gold"
                        : "bg-void/70 text-ivory border-white/10 hover:text-gold hover:border-gold/40"
                    }`}
                    aria-label="Good recommendation"
                  >
                    <ThumbsUp size={13} />
                  </button>
                  <button
                    onClick={(e) => handleFeedback(e, "down")}
                    className={`p-1.5 rounded-full backdrop-blur-sm border transition-colors ${
                      feedbackGiven === "down"
                        ? "bg-curtain text-ivory border-curtain"
                        : "bg-void/70 text-ivory border-white/10 hover:text-curtain hover:border-curtain/60"
                    }`}
                    aria-label="Not relevant"
                  >
                    <ThumbsDown size={13} />
                  </button>
                </>
              )}
              {onLikeToggle && (
                <button
                  onClick={handleLike}
                  className={`p-1.5 rounded-full backdrop-blur-sm border transition-colors ${
                    liked
                      ? "bg-curtain text-ivory border-curtain"
                      : "bg-void/70 text-ivory border-white/10 hover:text-curtain hover:border-curtain/60"
                  }`}
                  aria-label={liked ? "Unlike this movie" : "Like this movie"}
                  aria-pressed={liked}
                >
                  <ThumbsUp size={13} />
                </button>
              )}
              {onWatchlistToggle && (
                <button
                  onClick={handleWatchlist}
                  className={`p-1.5 rounded-full backdrop-blur-sm border transition-colors ${
                    inWatchlist
                      ? "bg-gold text-void border-gold"
                      : "bg-void/70 text-ivory border-white/10 hover:text-gold hover:border-gold/40"
                  }`}
                  aria-label={inWatchlist ? "Remove from watchlist" : "Add to watchlist"}
                  aria-pressed={inWatchlist}
                >
                  {inWatchlist ? <BookmarkCheck size={13} /> : <Bookmark size={13} />}
                </button>
              )}
            </div>
          )}

          <div className="absolute bottom-0 inset-x-0 p-3">
            {hasMeta && (
              <div className="flex items-center gap-2 font-mono text-[10px] text-smoke uppercase tracking-wider mb-1.5">
                {year && <span>{year}</span>}
                {runtime && <span className="text-smoke/70">{runtime}</span>}
                {rating && (
                  <span className="inline-flex items-center gap-0.5 text-gold-soft">
                    <Star size={9} fill="currentColor" /> {rating}
                  </span>
                )}
              </div>
            )}
            {genre && !showFooter && (
              <p className="text-[10px] font-mono uppercase tracking-wider text-gold/80 mb-1">
                {genre}
              </p>
            )}
            <h3 className="font-display text-sm text-ivory leading-snug line-clamp-2">
              {movie.title}
            </h3>
          </div>
        </div>

{showFooter && (
            <div className="px-3 py-2.5 bg-panel-raised">
              {genre && (
                <p className="text-[10px] font-mono uppercase tracking-wider text-gold/80 mb-1 truncate">
                  {genre}
                </p>
              )}
              {confidence && (
                <p
                  className={`font-mono text-[10px] uppercase tracking-wider ${
                    CONFIDENCE_COLOR[confidence.label] || "text-smoke"
                  }`}
                >
                  {confidence.label} confidence
                </p>
              )}
              {movie.explanation?.primary && (
                <p
                  className="text-gold-soft/90 text-[10px] font-mono mt-0.5 line-clamp-1"
                  title={
                    movie.explanation.secondary
                      ? `${movie.explanation.primary} • ${movie.explanation.secondary}`
                      : movie.explanation.primary
                  }
                >
                  {movie.explanation.primary}
                </p>
              )}
              {!movie.explanation?.primary && movie.reasons?.[0] && (
                <p
                  className="text-smoke text-[10px] mt-0.5 line-clamp-1"
                  title={movie.reasons.join(" • ")}
                >
                  {movie.reasons[0]}
                </p>
              )}
              {feedbackGiven === "down" && (
                <p className="text-smoke text-[10px] mt-0.5">Won't suggest this again</p>
              )}
            </div>
          )}
      </div>
    </Link>
  );
}