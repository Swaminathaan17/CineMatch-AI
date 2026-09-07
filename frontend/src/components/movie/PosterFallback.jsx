import { Clapperboard } from "lucide-react";

export default function PosterFallback({ title = "" }) {
  return (
    <div
      aria-hidden="true"
      className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-3 pb-2 bg-panel-raised overflow-hidden"
    >
      <div className="poster-grain absolute inset-0" />
      <div className="absolute inset-0 bg-gradient-to-b from-panel-raised via-transparent to-void/80" />
      <div className="relative flex flex-col items-center gap-2">
        <span className="w-9 h-9 rounded-full border border-gold/30 flex items-center justify-center">
          <Clapperboard size={16} className="text-gold/60" />
        </span>
        <span className="font-mono text-[9px] uppercase tracking-[0.25em] text-smoke/80 text-center">
          No poster available
        </span>
        {title && (
          <span className="font-display text-xs text-ivory/90 text-center leading-snug line-clamp-2 max-w-[80%]">
            {title}
          </span>
        )}
      </div>
    </div>
  );
}