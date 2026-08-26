import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import { Bookmark } from "lucide-react";
import Landing from "./pages/Landing";
import Home from "./pages/Home";
import MovieDetail from "./pages/MovieDetail";
import AIDiscovery from "./pages/AIDiscovery";
import Watchlist from "./pages/Watchlist";
import SearchBar from "./components/ui/SearchBar";

function Nav() {
  const location = useLocation();
  if (location.pathname === "/") return null;

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-md bg-void/70 border-b border-white/5 px-6 md:px-12 py-4 flex items-center justify-between gap-6">
      <Link to="/home" className="font-display text-lg text-ivory shrink-0">
        Reel
      </Link>
      <SearchBar />
      <div className="flex items-center gap-6 font-mono text-xs uppercase tracking-wider text-smoke shrink-0">
        <Link to="/home" className="hover:text-gold transition-colors">
          Home
        </Link>
        <Link to="/discover" className="hover:text-gold transition-colors">
          AI Discovery
        </Link>
        <Link to="/watchlist" className="hover:text-gold transition-colors flex items-center gap-1.5">
          <Bookmark size={13} />
          Watchlist
        </Link>
      </div>
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
        <Route path="/movie/tmdb/:tmdbId" element={<MovieDetail />} />
        <Route path="/movie/:id" element={<MovieDetail />} />
        <Route path="/discover" element={<AIDiscovery />} />
        <Route path="/watchlist" element={<Watchlist />} />
      </Routes>
    </BrowserRouter>
  );
}
