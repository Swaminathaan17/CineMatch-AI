import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Bookmark, BookmarkCheck, Library, LibraryBig } from "lucide-react";
import { api } from "../services/api";
import FilmstripRow from "../components/movie/FilmstripRow";
import SentimentGauge from "../components/sentiment/SentimentGauge";
import AspectSentimentCard from "../components/sentiment/AspectSentimentCard";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

function normalizeGenres(genres) {
  if (Array.isArray(genres)) return genres;
  return (genres || "").split(",").map((g) => g.trim()).filter(Boolean);
}

export default function MovieDetail() {
  const { id, tmdbId } = useParams();
  const isTmdbRoute = Boolean(tmdbId);
  const movieId = Number(id ?? tmdbId);

  const [movie, setMovie] = useState(null);
  const [similar, setSimilar] = useState([]);
  const [sentiment, setSentiment] = useState(null);
  const [liked, setLiked] = useState(false);
  const [inWatchlist, setInWatchlist] = useState(false);
  const [isLibraryMovie, setIsLibraryMovie] = useState(false);
  const [loading, setLoading] = useState(true);
  const [libraryBusy, setLibraryBusy] = useState(false);

  const loadRecommendations = async (libraryMovie) => {
    if (libraryMovie) {
      try {
        const recs = await api.getHybridRecommendations(movieId, 8);
        setSimilar(recs);
      } catch {
        setSimilar([]);
      }
      api.getSentiment(movieId).then(setSentiment).catch(() => {
        setSentiment({ status: "insufficient_data", review_count: 0 });
      });
    } else {
      try {
        const recs = await api.getTmdbRecommendations(movieId, 8);
        setSimilar(recs);
      } catch {
        setSimilar([]);
      }
      setSentiment({ status: "insufficient_data", review_count: 0 });
    }
  };

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        let movieData;
        let libraryMovie = false;

        if (isTmdbRoute) {
          // Phase 2: a TMDB URL can now point at a movie that has already been
          // imported into our persistent library. Prefer our library copy so
          // it uses the full recommendation engine.
          try {
            movieData = await api.getMovie(movieId);
            libraryMovie = true;
          } catch {
            const tmdbData = await api.getTmdbDetail(movieId);
            movieData = { ...tmdbData, source: "tmdb_external" };
          }
        } else {
          movieData = await api.getMovie(movieId);
          libraryMovie = true;
        }

        movieData.genres = normalizeGenres(movieData.genres);
        setMovie(movieData);
        setIsLibraryMovie(libraryMovie);
        await loadRecommendations(libraryMovie);

        const watchlist = await api.getWatchlist().catch(() => ({ results: [] }));
        setInWatchlist(watchlist.results.some((m) => m.id === movieId));
      } catch (e) {
        console.error(e);
        setMovie(null);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [movieId, isTmdbRoute]);

  const handleLike = async () => {
    await api.likeMovie(movieId);
    setLiked(true);
  };

  const handleWatchlistToggle = async () => {
    if (inWatchlist) {
      await api.removeFromWatchlist(movieId);
      setInWatchlist(false);
    } else {
      await api.addToWatchlist(movieId, movie.title);
      setInWatchlist(true);
    }
  };

  const handleLibraryToggle = async () => {
    if (libraryBusy) return;
    setLibraryBusy(true);
    try {
      if (isLibraryMovie) {
        await api.removeFromLibrary(movieId);
        const tmdbData = await api.getTmdbDetail(movieId);
        const external = { ...tmdbData, source: "tmdb_external", genres: normalizeGenres(tmdbData.genres) };
        setMovie(external);
        setIsLibraryMovie(false);
        await loadRecommendations(false);
      } else {
        const response = await api.addToLibrary(movieId);
        const imported = response.movie;
        imported.genres = normalizeGenres(imported.genres);
        setMovie(imported);
        setIsLibraryMovie(true);
        await loadRecommendations(true);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLibraryBusy(false);
    }
  };

  const handleRecommendationFeedback = (recommendedMovieId, value) => {
    api.submitRecommendationFeedback(movieId, recommendedMovieId, value).catch(console.error);
  };

  if (loading || !movie) {
    return (
      <div className="min-h-screen pt-24">
        <LoadingSkeleton />
      </div>
    );
  }

  const similarMatchScores = similar.reduce((acc, r) => {
    acc[r.id] = r.match_percentage;
    return acc;
  }, {});
  const similarConfidenceScores = similar.reduce((acc, r) => {
    if (r.confidence) acc[r.id] = r.confidence;
    return acc;
  }, {});

  const isOriginalMovie = movie.source === "local";
  const isImportedTmdbMovie = isLibraryMovie && !isOriginalMovie;
  const badgeText = isOriginalMovie
    ? null
    : isImportedTmdbMovie
      ? "In your library — powered by the full recommendation engine"
      : "Found via TMDB — not imported yet";

  return (
    <div className="min-h-screen pb-20">
      <div className="relative h-[50vh] flex items-end">
        <div
          className="absolute inset-0"
          style={{
            background:
              movie?.backdrop_path
                ? `linear-gradient(to bottom, rgba(11,11,14,0.1), #0B0B0E 88%), url(https://image.tmdb.org/t/p/w1280${movie.backdrop_path}) center/cover`
                : "linear-gradient(to bottom, rgba(11,11,14,0.2), #0B0B0E), radial-gradient(ellipse 60% 60% at 30% 10%, rgba(139,41,66,0.3), transparent 60%)",
          }}
        />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative z-10 px-6 md:px-12 pb-10 max-w-4xl"
        >
          <h1 className="font-display text-4xl md:text-5xl text-ivory mb-3">
            {movie.title}
          </h1>
          {badgeText && (
            <p className="text-smoke text-xs font-mono uppercase tracking-wider mb-3 border border-white/10 rounded-full px-3 py-1 inline-block">
              {badgeText}
            </p>
          )}
          <div className="flex items-center gap-3 flex-wrap mb-4">
            {normalizeGenres(movie.genres).map((g) => (
              <span
                key={g}
                className="text-xs font-mono uppercase tracking-wider text-smoke border border-white/10 rounded-full px-3 py-1"
              >
                {g}
              </span>
            ))}
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <button
              onClick={handleLike}
              disabled={liked}
              className={`px-5 py-2.5 rounded-sm font-body font-medium transition-colors ${
                liked
                  ? "bg-panel text-smoke cursor-default"
                  : "bg-curtain hover:bg-curtain-dim text-ivory"
              }`}
            >
              {liked ? "Added to your taste profile" : "I like this movie"}
            </button>
            <button
              onClick={handleWatchlistToggle}
              className={`p-2.5 rounded-sm border transition-colors flex items-center gap-2 ${
                inWatchlist
                  ? "border-gold/50 text-gold bg-gold/10"
                  : "border-white/10 text-smoke hover:text-ivory hover:border-white/30"
              }`}
              aria-label={inWatchlist ? "Remove from watchlist" : "Add to watchlist"}
            >
              {inWatchlist ? <BookmarkCheck size={18} /> : <Bookmark size={18} />}
            </button>

            {!isOriginalMovie && (
              <button
                onClick={handleLibraryToggle}
                disabled={libraryBusy}
                className="px-4 py-2.5 rounded-sm border border-gold/40 text-gold hover:bg-gold/10 transition-colors flex items-center gap-2 text-sm disabled:opacity-50"
              >
                {isImportedTmdbMovie ? <LibraryBig size={17} /> : <Library size={17} />}
                {libraryBusy
                  ? "Updating…"
                  : isImportedTmdbMovie
                    ? "Remove from library"
                    : "Add to library"}
              </button>
            )}
          </div>
        </motion.div>
      </div>

      {isLibraryMovie && (
        <section className="px-6 md:px-12 py-10">
          <h2 className="font-display text-xl text-ivory mb-6">Audience sentiment</h2>
          {!sentiment ? (
            <p className="text-smoke text-sm">Loading sentiment…</p>
          ) : sentiment.status === "insufficient_data" ? (
            <p className="text-smoke text-sm">
              Not enough reviews yet to compute reliable sentiment
              {sentiment.review_count != null ? ` (${sentiment.review_count} found)` : ""}.
            </p>
          ) : (
            <>
              <SentimentGauge
                positivePct={sentiment.overall.positive_pct}
                label={sentiment.overall.label}
              />
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mt-8">
                {Object.entries(sentiment.aspects).map(([aspect, data]) => (
                  <AspectSentimentCard key={aspect} aspect={aspect} data={data} />
                ))}
              </div>
            </>
          )}
        </section>
      )}

      <FilmstripRow
        title={isLibraryMovie ? "Similar movies" : "Because you searched for this"}
        movies={similar}
        matchScores={similarMatchScores}
        confidenceScores={similarConfidenceScores}
        onFeedback={handleRecommendationFeedback}
      />
    </div>
  );
}
