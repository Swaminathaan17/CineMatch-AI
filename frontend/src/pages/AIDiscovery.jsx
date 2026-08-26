import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, SlidersHorizontal } from "lucide-react";
import { api } from "../services/api";
import MovieCard from "../components/movie/MovieCard";

const EXAMPLE_QUERIES = [
  "Something like Interstellar but darker and more emotional",
  "A mind-bending sci-fi movie about dreams, under 150 minutes",
  "A fun movie to watch with friends, highly rated and not too serious",
  "A dark psychological thriller with a crazy twist ending",
  "Recent movies about survival in space",
];

export default function AIDiscovery() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runSearch = async (q) => {
    const searchQuery = q ?? query;
    if (!searchQuery.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await api.discoverByMood(searchQuery, 12);
      setResult(res);
    } catch (e) {
      setError(e.message || "Discovery failed. Try another description.");
    } finally { setLoading(false); }
  };

  const intent = result?.intent;
  const interpreted = result?.explanation?.interpreted_as || [];

  return (
    <div className="min-h-screen pt-16 pb-24 px-6 md:px-12">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="max-w-3xl mx-auto text-center mb-10">
        <div className="flex items-center justify-center gap-2 mb-4">
          <Sparkles size={15} className="text-gold" />
          <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase">AI Discovery</p>
        </div>
        <h1 className="font-display text-3xl md:text-5xl text-ivory mb-4">Tell me what you want to watch.</h1>
        <p className="text-smoke text-sm mb-6">Titles, moods, themes, constraints, or comparisons — write it naturally.</p>
        <div className="flex gap-2">
          <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === "Enter" && runSearch()} placeholder="e.g. something like Interstellar but darker" className="flex-1 bg-panel border border-white/10 focus:border-gold/50 rounded-sm px-4 py-3 text-ivory outline-none" />
          <button onClick={() => runSearch()} disabled={loading} className="px-6 py-3 bg-curtain hover:bg-curtain-dim rounded-sm font-medium text-ivory shrink-0 disabled:opacity-50">{loading ? "Thinking…" : "Discover"}</button>
        </div>
        <div className="flex flex-wrap gap-2 justify-center mt-4">
          {EXAMPLE_QUERIES.map((eq) => <button key={eq} onClick={() => { setQuery(eq); runSearch(eq); }} className="text-xs text-smoke hover:text-gold border border-white/10 rounded-full px-3 py-1.5">{eq}</button>)}
        </div>
      </motion.div>

      {error && <p className="max-w-3xl mx-auto text-center text-curtain text-sm mb-6">{error}</p>}
      {loading && <p className="text-center text-smoke text-sm">Understanding your request and searching your library + TMDB…</p>}

      <AnimatePresence>
        {result && !loading && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="max-w-6xl mx-auto">
            <div className="bg-panel border border-white/5 rounded-md p-5 mb-8">
              <div className="flex items-center gap-2 mb-3"><SlidersHorizontal size={14} className="text-gold" /><span className="font-mono text-xs uppercase tracking-wider text-gold">I understood</span></div>
              <div className="flex flex-wrap gap-2">
                {interpreted.map((x) => <span key={x} className="text-xs text-ivory border border-white/10 rounded-full px-3 py-1">{x}</span>)}
                {intent?.runtime_max && <span className="text-xs text-ivory border border-white/10 rounded-full px-3 py-1">≤ {intent.runtime_max} min</span>}
                {intent?.min_rating && <span className="text-xs text-ivory border border-white/10 rounded-full px-3 py-1">≥ {intent.min_rating}/10</span>}
              </div>
              <p className="text-smoke text-xs mt-4">Semantic engine: <span className="text-ivory">{result.mode}</span> · TMDB: <span className="text-ivory">{result.sources?.tmdb ? "used" : "not needed"}</span></p>
            </div>

            {result.results?.length ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {result.results.map((r) => <MovieCard key={`${r.source}-${r.id}`} movie={r} matchPercentage={r.match_percentage} onFeedback={r.source === "local" ? undefined : undefined} />)}
              </div>
            ) : (
              <div className="text-center text-smoke py-16">I couldn't find a strong match. Try adding a genre, mood, actor, theme, year, or runtime.</div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
