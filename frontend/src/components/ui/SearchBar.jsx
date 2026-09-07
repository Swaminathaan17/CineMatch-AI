import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Search, Loader2 } from "lucide-react";
import { api } from "../../services/api";
import { thumbnailUrl, posterUrl } from "../../utils/tmdbImage";
import { formatReleaseYear } from "../../utils/format";
import PosterFallback from "../movie/PosterFallback";

export default function SearchBar() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);
  const debounceRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!query.trim()) {
      setResult(null);
      setOpen(false);
      setLoading(false);
      setError("");
      return;
    }
    clearTimeout(debounceRef.current);
    setLoading(true);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await api.searchMovies(query, 6);
        setResult(res);
        setError("");
        setActiveIndex(-1);
        setOpen(true);
      } catch {
        setResult(null);
        setError("Search is unavailable right now.");
        setOpen(true);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(debounceRef.current);
  }, [query]);

  const runSearch = () => {
    if (!query.trim()) return;
    navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    setOpen(false);
  };

  const submitTmdb = () => {
    setOpen(false);
    navigate("/search?q=" + encodeURIComponent(query.trim()) + "&live=1");
  };

  const onKeyDown = (e) => {
    if (!open || !result?.results?.length) {
      if (e.key === "Enter") runSearch();
      return;
    }
    const items = result.results;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Escape") {
      setOpen(false);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const focus = items[activeIndex];
      if (focus) {
        const path = focus.source === "tmdb" ? `/movie/tmdb/${focus.id}` : `/movie/${focus.id}`;
        navigate(path);
      } else {
        runSearch();
      }
      setOpen(false);
    }
  };

  const currentResult = result?.results?.[activeIndex];

  return (
    <div className="relative w-full max-w-md">
      <div className="relative">
        <Search
          size={15}
          className="absolute left-3.5 top-1/2 -translate-y-1/2 text-smoke"
          aria-hidden="true"
        />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => query.trim() && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKeyDown}
          placeholder="Search movies…"
          role="combobox"
          aria-expanded={open}
          aria-controls="search-results"
          aria-activedescendant={
            currentResult ? `search-result-${currentResult.id}` : undefined
          }
          className="w-full bg-panel border border-white/10 focus:border-gold/50 rounded-sm pl-9 pr-9 py-2.5 text-sm text-ivory placeholder:text-smoke outline-none transition-colors"
        />
        {loading && (
          <Loader2
            size={14}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gold animate-spin"
            aria-hidden="true"
          />
        )}
      </div>

      {open && (
        <div
          id="search-results"
          className="absolute top-full left-0 right-0 mt-2 bg-panel-raised border border-white/10 rounded-md overflow-hidden z-50 shadow-xl"
        >
          {error ? (
            <div className="px-4 py-3">
              <p className="text-smoke text-xs leading-relaxed">{error}</p>
            </div>
          ) : !result ? (
            <div className="px-4 py-3">
              <p className="text-smoke text-xs">Searching…</p>
            </div>
          ) : result.found ? (
            <>
              {result.source === "tmdb" && (
                <div className="px-4 py-2 bg-void/50 border-b border-white/5">
                  <p className="text-smoke text-[11px] font-mono uppercase tracking-wider">
                    Found on TMDB — open it to preview, or add it to your library
                  </p>
                  <button
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={submitTmdb}
                    className="inline-block mt-1.5 text-gold text-[11px] font-mono uppercase tracking-wider hover:underline"
                  >
                    Show all live results →
                  </button>
                </div>
              )}
              <ul role="listbox">
                {result.results.map((m, i) => {
                  const poster = posterUrl(m.poster_path) || thumbnailUrl(m.poster_path);
                  const year = formatReleaseYear(m.release_date);
                  return (
                    <li key={m.id} role="option" aria-selected={i === activeIndex}>
                      <Link
                        id={`search-result-${m.id}`}
                        to={m.source === "tmdb" ? `/movie/tmdb/${m.id}` : `/movie/${m.id}`}
                        className={`flex items-center gap-3 px-3 py-2 text-sm text-ivory transition-colors ${
                          i === activeIndex ? "bg-panel text-gold" : "hover:bg-panel"
                        }`}
                      >
                        {poster ? (
                          <img
                            src={poster}
                            alt=""
                            loading="lazy"
                            className="w-9 h-[52px] object-cover rounded-sm bg-panel-raised"
                          />
                        ) : (
                          <div className="relative w-9 h-[52px] rounded-sm overflow-hidden">
                            <PosterFallback />
                          </div>
                        )}
                        <span className="flex flex-col min-w-0">
                          <span className="truncate">{m.title}</span>
                          {year && (
                            <span className="font-mono text-[10px] uppercase tracking-wider text-smoke">
                              {year}
                            </span>
                          )}
                        </span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
              <div className="px-3 py-2 border-t border-white/5">
                <button
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={runSearch}
                  className="text-smoke text-[11px] font-mono uppercase tracking-wider hover:text-gold"
                >
                  View all results →
                </button>
              </div>
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