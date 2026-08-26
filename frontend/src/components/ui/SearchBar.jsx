import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../services/api";

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [open, setOpen] = useState(false);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (!query.trim()) {
      setResult(null);
      setOpen(false);
      return;
    }
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await api.searchMovies(query, 6);
        setResult(res);
        setOpen(true);
      } catch (e) {
        console.error(e);
      }
    }, 300);
    return () => clearTimeout(debounceRef.current);
  }, [query]);

  return (
    <div className="relative w-full max-w-md">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => query.trim() && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Search movies…"
        className="w-full bg-panel border border-white/10 focus:border-gold/50 rounded-sm px-4 py-2.5 text-sm text-ivory placeholder:text-smoke outline-none transition-colors"
      />

      {open && result && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-panel-raised border border-white/10 rounded-md overflow-hidden z-50 shadow-xl">
          {result.found ? (
            <>
              {result.source === "tmdb" && (
                <div className="px-4 py-2 bg-void/50 border-b border-white/5">
                  <p className="text-smoke text-[11px] font-mono uppercase tracking-wider">
                    Found on TMDB — open it to preview, or add it to your library
                  </p>
                </div>
              )}
              {result.results.map((m) => (
                <Link
                  key={m.id}
                  to={m.source === "tmdb" ? `/movie/tmdb/${m.id}` : `/movie/${m.id}`}
                  className="block px-4 py-2.5 text-sm text-ivory hover:bg-panel hover:text-gold transition-colors"
                >
                  {m.title}
                </Link>
              ))}
            </>
          ) : (
            <div className="px-4 py-3">
              <p className="text-smoke text-xs leading-relaxed">{result.message}</p>
              <Link
                to="/discover"
                className="inline-block mt-2 text-gold text-xs font-mono uppercase tracking-wider hover:underline"
              >
                Try AI Discovery instead →
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
