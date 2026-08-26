import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { useState } from "react";

const CONFIDENCE_COLOR = {
  High: "text-gold",
  Medium: "text-smoke",
  Low: "text-curtain",
};

export default function MovieCard({ movie, matchPercentage, confidence, onFeedback }) {
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  const handleFeedback = (e, value) => {
    e.preventDefault();
    e.stopPropagation();
    setFeedbackGiven(value);
    onFeedback?.(movie.id, value);
  };

  return (
    <Link to={`/movie/${movie.id}`} className="shrink-0 w-48 md:w-56 group">
      <motion.div
        whileHover={{ y: -6 }}
        transition={{ duration: 0.25, ease: "easeOut" }}
        className="rounded-md overflow-hidden bg-panel border border-white/5 hover:border-gold/40 transition-colors"
      >
        <div className="relative aspect-[2/3] bg-panel-raised overflow-hidden">
          {movie.poster_path ? (
            <img
              src={`https://image.tmdb.org/t/p/w400${movie.poster_path}`}
              alt={movie.title}
              className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
              loading="lazy"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <span className="font-display text-smoke text-sm px-4 text-center">
                {movie.title}
              </span>
            </div>
          )}
          {matchPercentage != null && (
            <div className="absolute top-2 right-2 bg-void/85 backdrop-blur-sm px-2 py-0.5 rounded-full">
              <span className="font-mono text-xs text-gold">{matchPercentage}%</span>
            </div>
          )}

          {onFeedback && (
            <div className="absolute bottom-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={(e) => handleFeedback(e, "up")}
                className={`p-1.5 rounded-full backdrop-blur-sm transition-colors ${
                  feedbackGiven === "up" ? "bg-gold text-void" : "bg-void/85 text-smoke hover:text-gold"
                }`}
                aria-label="Good recommendation"
              >
                <ThumbsUp size={12} />
              </button>
              <button
                onClick={(e) => handleFeedback(e, "down")}
                className={`p-1.5 rounded-full backdrop-blur-sm transition-colors ${
                  feedbackGiven === "down" ? "bg-curtain text-ivory" : "bg-void/85 text-smoke hover:text-curtain"
                }`}
                aria-label="Not relevant"
              >
                <ThumbsDown size={12} />
              </button>
            </div>
          )}
        </div>
        <div className="p-3">
          <p className="font-display text-sm text-ivory truncate">{movie.title}</p>
          {confidence && (
            <p className={`font-mono text-[10px] uppercase tracking-wider mt-1 ${CONFIDENCE_COLOR[confidence.label] || "text-smoke"}`}>
              {confidence.label} confidence
            </p>
          )}
          {feedbackGiven === "down" && (
            <p className="text-smoke text-[10px] mt-1">Won't suggest this again</p>
          )}
        </div>
      </motion.div>
    </Link>
  );
}
