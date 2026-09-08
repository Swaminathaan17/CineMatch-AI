import { motion } from "framer-motion";
import { Link } from "react-router-dom";

export default function Landing() {
  return (
    <div className="min-h-screen relative flex flex-col items-center justify-center overflow-hidden">
      {/* ambient backdrop - radial wine glow, not a stock image, since we
          don't want to depend on a specific licensed still */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 60% at 50% 20%, rgba(139,41,66,0.35), transparent 60%), radial-gradient(ellipse 60% 50% at 80% 80%, rgba(201,162,39,0.08), transparent 60%)",
        }}
      />
      <div className="absolute inset-0 bg-void/40" />

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10 text-center px-6 max-w-2xl"
      >
        <p className="font-mono text-xs tracking-[0.3em] text-gold uppercase mb-6">
          Reel — AI Movie Discovery
        </p>
        <h1 className="font-display text-5xl md:text-7xl text-ivory leading-[1.05] mb-6">
          Find your next
          <br />
          <span className="italic text-gold-soft">favorite</span> film
        </h1>
        <p className="text-smoke text-lg mb-10 max-w-lg mx-auto">
          Recommendations built from what audiences actually felt watching
          it — not just what's similar on paper.
        </p>
        <div className="flex items-center justify-center gap-4">
          <Link
            to="/home"
            className="px-6 py-3 bg-curtain hover:bg-curtain-dim transition-colors rounded-sm font-body font-medium text-ivory"
          >
            Explore movies
          </Link>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
        className="absolute bottom-8 font-mono text-xs text-smoke tracking-wider"
      >
        SCROLL TO EXPLORE
      </motion.div>
    </div>
  );
}
