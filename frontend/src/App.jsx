import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { useState } from "react";
import { Bookmark, Menu, X } from "lucide-react";
import Landing from "./pages/Landing";
import Home from "./pages/Home";
import MovieDetail from "./pages/MovieDetail";
import AIDiscovery from "./pages/AIDiscovery";
import Watchlist from "./pages/Watchlist";
import SearchResults from "./pages/SearchResults";
import SearchBar from "./components/ui/SearchBar";

function NavLink({ to, children, onClick }) {
  const location = useLocation();
  const active = location.pathname === to;
  return (
    <Link
      to={to}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={`text-smoke hover:text-gold transition-colors ${
        active ? "text-gold" : ""
      }`}
    >
      {children}
    </Link>
  );
}

function Nav() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  if (location.pathname === "/") return null;

  const closeMenu = () => setMenuOpen(false);

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-md bg-void/70 border-b border-white/5">
      <div className="px-4 sm:px-6 md:px-12 py-3.5 flex items-center justify-between gap-3">
        <Link to="/home" className="font-display text-lg text-ivory shrink-0">
          Reel
        </Link>
        <div className="flex-1 min-w-0 flex justify-end px-2 sm:px-4">
          <SearchBar />
        </div>
        <div className="hidden md:flex items-center gap-6 font-mono text-xs uppercase tracking-wider text-smoke shrink-0">
          <NavLink to="/home">Home</NavLink>
          <NavLink to="/discover">AI Discovery</NavLink>
          <NavLink to="/watchlist">
            <span className="flex items-center gap-1.5">
              <Bookmark size={13} />
              Watchlist
            </span>
          </NavLink>
        </div>
        <button
          onClick={() => setMenuOpen((v) => !v)}
          className="md:hidden shrink-0 p-2 rounded-sm border border-white/10 text-ivory"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          aria-expanded={menuOpen}
          aria-controls="mobile-nav"
        >
          {menuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>

      {menuOpen && (
        <div
          id="mobile-nav"
          className="md:hidden border-t border-white/5 bg-void/95 backdrop-blur-md px-4 py-3 flex flex-col gap-3 font-mono text-xs uppercase tracking-wider"
        >
          <NavLink to="/home" onClick={closeMenu}>
            Home
          </NavLink>
          <NavLink to="/discover" onClick={closeMenu}>
            AI Discovery
          </NavLink>
          <NavLink to="/watchlist" onClick={closeMenu}>
            <span className="flex items-center gap-1.5">
              <Bookmark size={13} />
              Watchlist
            </span>
          </NavLink>
        </div>
      )}
    </nav>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/home" element={<Home />} />
        <Route path="/search" element={<SearchResults />} />
        <Route path="/movie/tmdb/:tmdbId" element={<MovieDetail />} />
        <Route path="/movie/:id" element={<MovieDetail />} />
        <Route path="/discover" element={<AIDiscovery />} />
        <Route path="/watchlist" element={<Watchlist />} />
      </Routes>
    </BrowserRouter>
  );
}
