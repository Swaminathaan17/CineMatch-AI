import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Bookmark } from "lucide-react";
import { api } from "../services/api";
import LoadingSkeleton from "../components/ui/LoadingSkeleton";

export default function Watchlist() {
  const [items, setItems] = useState(null);

  useEffect(() => {
    api.getWatchlist().then((r) => setItems(r.results)).catch(() => setItems([]));
  }, []);

  return (
    <div className="min-h-screen pt-16 pb-24 px-6 md:px-12">
      <div className="mb-10">
        <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase mb-2">
          Your List
        </p>
        <h1 className="font-display text-3xl text-ivory">Watchlist</h1>
      </div>

      {items === null ? (
        <LoadingSkeleton />
      ) : items.length === 0 ? (
        <div className="text-center py-20">
          <Bookmark className="mx-auto text-smoke mb-4" size={32} />
          <p className="text-smoke text-sm">
            Nothing saved yet. Tap the bookmark icon on a movie page to add it here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          {items.map((m, i) => (
            <motion.div
              key={m.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <Link to={`/movie/${m.id}`}>
                <div className="aspect-[2/3] rounded-md bg-panel border border-white/5 hover:border-gold/40 transition-colors flex items-center justify-center p-4">
                  <span className="font-display text-sm text-ivory text-center">{m.title}</span>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
