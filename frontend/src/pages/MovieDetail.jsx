import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Bookmark,
  BookmarkCheck,
  Library,
  LibraryBig,
  Play,
  Star,
} from "lucide-react";
import { api } from "../services/api";
import FilmstripRow from "../components/movie/FilmstripRow";
import PosterFallback from "../components/movie/PosterFallback";
import SentimentGauge from "../components/sentiment/SentimentGauge";
import AspectSentimentCard from "../components/sentiment/AspectSentimentCard";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";
import ErrorState from "../components/ui/ErrorState";
import { posterUrl, backdropUrl } from "../utils/tmdbImage";
import { enrichMovies } from "../services/movieEnrichment";
import {
  formatReleaseYear,
  formatRuntime,
  formatRating,
  normalizeGenres,
} from "../utils/format";

function WatchProviderGroup({ label, providers }) {
  if (!providers || providers.length === 0) return null;
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-wider text-smoke mb-1.5">
        {label}
      </p>
      <div className="flex items-center gap-2 flex-wrap">
        {providers.map((p) =>
          p.logo_path ? (
            <img
              key={p.provider_name || p.logo_path}
              src={`https://image.tmdb.org/t/p/w92${p.logo_path}`}
              alt={p.provider_name || "streaming provider"}
              title={p.provider_name || "streaming provider"}
              loading="lazy"
              className="w-9 h-9 rounded-md bg-white/10 object-cover"
            />
          ) : (
            <span
              key={p.provider_name}
              className="text-xs text-ivory border border-white/10 rounded-md px-2 py-1.5 bg-panel"
            >
              {p.provider_name}
            </span>
          )
        )}
      </div>
    </div>
  );
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
  const [trailer, setTrailer] = useState(null);
  const [watchProviders, setWatchProviders] = useState(null);
  const [playingTrailer, setPlayingTrailer] = useState(false);

  const loadRecommendations = useCallback(
    async (libraryMovie) => {
      if (libraryMovie) {
        try {
          const recs = await api.getHybridRecommendations(movieId, 8);
          setSimilar(await enrichMovies(recs));
        } catch {
          setSimilar([]);
        }
        api.getSentiment(movieId).then(setSentiment).catch(() => {
          setSentiment({ status: "insufficient_data", review_count: 0 });
        });
      } else {
        try {
          const recs = await api.getTmdbRecommendations(movieId, 8);
          setSimilar(await enrichMovies(recs));
        } catch {
          setSimilar([]);
        }
        setSentiment({ status: "insufficient_data", review_count: 0 });
      }
    },
    [movieId]
  );

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setPlayingTrailer(false);
      try {
        let movieData;
        let libraryMovie = false;

        if (isTmdbRoute) {
          // A TMDB URL can now point at a movie already imported into our
          // persistent library. Prefer the library copy so it uses the full
          // recommendation engine.
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
        if (cancelled) return;
        setMovie(movieData);
        setIsLibraryMovie(libraryMovie);

        // Library rows don't carry trailer + watch providers - enrich from the
        // TMDB detail endpoint (TMDB-only rows already have them).
        if (movieData.source === "tmdb_external") {
          setTrailer(movieData.trailer || null);
          setWatchProviders(movieData.watch_providers || null);
        } else {
          api
            .getTmdbDetail(movieId)
            .then((detail) => {
              if (cancelled) return;
              setTrailer(detail.trailer || null);
              setWatchProviders(detail.watch_providers || null);
            })
            .catch(() => {
              setTrailer(null);
              setWatchProviders(null);
            });
        }

        await loadRecommendations(libraryMovie);

        const watchlist = await api.getWatchlist().catch(() => ({ results: [] }));
        if (cancelled) return;
        setInWatchlist((watchlist.results || []).some((m) => m.id === movieId));

        const prefs = await api.getPreferences().catch(() => null);
        if (!cancelled && prefs?.liked_movie_ids) {
          setLiked(prefs.liked_movie_ids.includes(movieId));
        }
      } catch (e) {
        console.error(e);
        if (!cancelled) setMovie(null);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [movieId, isTmdbRoute, loadRecommendations]);

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
        const external = {
          ...tmdbData,
          source: "tmdb_external",
          genres: normalizeGenres(tmdbData.genres),
        };
        setMovie(external);
        setTrailer(tmdbData.trailer || null);
        setWatchProviders(tmdbData.watch_providers || null);
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

  if (loading) {
    return (
      <div className="min-h-screen pt-24">
        <LoadingSkeleton variant="row" />
      </div>
    );
  }

  if (!movie) {
    return (
      <div className="min-h-screen pt-24 px-6">
        <ErrorState
          title="Movie not found"
          message="We couldn't load this title from your library or TMDB."
        />
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

  const heroBackdrop = backdropUrl(movie.backdrop_path);
  const heroPoster = posterUrl(movie.poster_path);
  const year = formatReleaseYear(movie.release_date);
  const runtime = formatRuntime(movie.runtime);
  const rating = formatRating(movie.vote_average);
  const genres = normalizeGenres(movie.genres);
  const trailerKey = trailer?.key || null;
  const trailerUrl = trailer?.url || (trailerKey ? `https://www.youtube.com/watch?v=${trailerKey}` : null);
  const providers = watchProviders && watchProviders.available ? watchProviders : null;

  return (
    <div className="min-h-screen pb-20">
      <div className="relative">
        {heroBackdrop && (
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{ backgroundImage: `url(${heroBackdrop})` }}
            aria-hidden="true"
          />
        )}
        <div
          className={`absolute inset-0 ${
            heroBackdrop
              ? "bg-gradient-to-t from-void via-void/60 to-void/30"
              : "bg-gradient-to-b from-void via-panel to-void"
          }`}
          aria-hidden="true"
        />

        <div className="relative z-10 px-6 md:px-12 pt-16 md:pt-24 pb-10 max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row gap-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="shrink-0 w-full flex justify-center md:block md:w-56 lg:w-72"
            >
              <div className="relative aspect-[2/3] w-44 md:w-56 lg:w-72 rounded-lg overflow-hidden border border-white/10 shadow-2xl">
                {heroPoster ? (
                  <img
                    src={heroPoster}
                    alt={`${movie.title} poster`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <PosterFallback title={movie.title} />
                )}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="flex-1"
            >
              {badgeText && (
                <p className="text-smoke text-xs font-mono uppercase tracking-wider mb-3 border border-white/10 rounded-full px-3 py-1 inline-block">
                  {badgeText}
                </p>
              )}

              <h1 className="font-display text-4xl md:text-5xl text-ivory mb-3">
                {movie.title}
              </h1>

              <div className="flex items-center gap-3 flex-wrap mb-4 font-mono text-sm text-smoke">
                {year && <span>{year}</span>}
                {runtime && <span>{runtime}</span>}
                {rating && (
                  <span className="inline-flex items-center gap-1 text-gold-soft">
                    <Star size={13} fill="currentColor" /> {rating}
                  </span>
                )}
              </div>

              {genres.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap mb-5">
                  {genres.map((g) => (
                    <span
                      key={g}
                      className="text-xs font-mono uppercase tracking-wider text-smoke border border-white/10 rounded-full px-3 py-1"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              )}

              {movie.overview ? (
                <p className="text-smoke leading-relaxed max-w-3xl mb-6">
                  {movie.overview}
                </p>
              ) : (
                <p className="text-smoke text-sm italic mb-6">
                  No overview available for this title yet.
                </p>
              )}

              {(trailerUrl || providers) && (
                <div className="flex items-center gap-3 flex-wrap mb-8">
                  {trailerUrl && (
                    <button
                      onClick={() => setPlayingTrailer((v) => !v)}
                      className="px-5 py-2.5 rounded-sm bg-curtain hover:bg-curtain-dim text-ivory font-medium transition-colors flex items-center gap-2"
                      aria-expanded={playingTrailer}
                    >
                      <Play size={15} />
                      {playingTrailer ? "Hide trailer" : "Watch trailer"}
                    </button>
                  )}
                  {providers?.flatrate?.length > 0 && (
                    <span className="text-xs text-smoke">
                      Streaming on {providers.flatrate.map((p) => p.provider_name).join(", ")}
                    </span>
                  )}
                </div>
              )}

              <div className="flex flex-wrap items-center gap-3">
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
        </div>
      </div>

      {playingTrailer && trailerUrl && (
        <div className="px-6 md:px-12 max-w-4xl mx-auto mb-10" aria-live="polite">
          <div className="aspect-video rounded-lg overflow-hidden border border-white/10 bg-void">
            <iframe
              src={`https://www.youtube-nocookie.com/embed/${trailer.key}?autoplay=1`}
              title={trailer.name || "Movie trailer"}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
              className="w-full h-full"
            />
          </div>
        </div>
      )}

      {providers && (
        <section className="px-6 md:px-12 max-w-4xl mx-auto mb-10">
          <h2 className="font-display text-xl text-ivory mb-5">Where to watch</h2>
          <div className="space-y-4">
            <WatchProviderGroup label="Streaming" providers={providers.flatrate} />
            <WatchProviderGroup label="Rent" providers={providers.rent} />
            <WatchProviderGroup label="Buy" providers={providers.buy} />
            {(providers.flatrate?.length || 0) + (providers.rent?.length || 0) +
              (providers.buy?.length || 0) ===
              0 && (
              <p className="text-smoke text-sm">
                No streaming availability data for this title in your region yet.
              </p>
            )}
          </div>
        </section>
      )}

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