import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "../services/api";
import MovieCard from "../components/movie/MovieCard";

const EXAMPLE_QUERIES = [
  "A mind-bending sci-fi movie with emotional depth",
  "Something dark and psychological with a twist ending",
  "Light and funny, good to watch with friends",
];

export default function AIDiscovery() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const runSearch = async (q) => {
    const searchQuery = q ?? query;
    if (!searchQuery.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await api.discoverByMood(searchQuery, 10);
      setResult(res);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen pt-16 pb-24 px-6 md:px-12">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-2xl mx-auto text-center mb-10"
      >
        <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase mb-4">
          AI Discovery
        </p>
        <h1 className="font-display text-3xl md:text-4xl text-ivory mb-4">
          Describe the movie you're in the mood for
        </h1>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runSearch()}
            placeholder="e.g. a dark psychological thriller with a surprising ending"
            className="flex-1 bg-panel border border-white/10 focus:border-gold/50 rounded-sm px-4 py-3 text-ivory placeholder:text-smoke outline-none transition-colors"
          />
          <button
            onClick={() => runSearch()}
            className="px-6 py-3 bg-curtain hover:bg-curtain-dim transition-colors rounded-sm font-body font-medium text-ivory shrink-0"
          >
            Search
          </button>
        </div>
        <div className="flex flex-wrap gap-2 justify-center mt-4">
          {EXAMPLE_QUERIES.map((eq) => (
            <button
              key={eq}
              onClick={() => {
                setQuery(eq);
                runSearch(eq);
              }}
              className="text-xs text-smoke hover:text-gold border border-white/10 rounded-full px-3 py-1.5 transition-colors"
            >
              {eq}
            </button>
          ))}
        </div>
      </motion.div>

      {loading && (
        <p className="text-center text-smoke text-sm">Interpreting your request…</p>
      )}

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-5xl mx-auto"
          >
            <div className="mb-6 text-center">
              {result.interpreted_genres?.length > 0 && (
                <p className="text-smoke text-sm mb-2">
                  Interpreted as:{" "}
                  {result.interpreted_genres.map((g) => (
                    <span
                      key={g}
                      className="inline-block mx-1 text-gold font-mono text-xs uppercase tracking-wider border border-gold/30 rounded-full px-2 py-0.5"
                    >
                      {g}
                    </span>
                  ))}
                </p>
              )}
              {result.mode === "fallback" && (
                <p className="text-smoke text-xs font-mono">
                  Matched by genre + keyword overlap (semantic model unavailable)
                </p>
              )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {result.results.map((r) => (
                <MovieCard
                  key={r.id}
                  movie={{ id: r.id, title: r.title, poster_path: null }}
                  matchPercentage={Math.round(r.match_score * 100)}
                />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
