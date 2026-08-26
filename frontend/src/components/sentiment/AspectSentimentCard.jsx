const LABEL_COLOR = {
  "Very Positive": "bg-gold",
  Positive: "bg-gold/70",
  Mixed: "bg-smoke",
  Negative: "bg-curtain/70",
  "Very Negative": "bg-curtain",
};

export default function AspectSentimentCard({ aspect, data }) {
  const capitalized = aspect.charAt(0).toUpperCase() + aspect.slice(1);

  if (data.status === "insufficient_data") {
    return (
      <div className="rounded-md border border-white/5 bg-panel p-4">
        <p className="font-display text-sm text-ivory mb-2">{capitalized}</p>
        <p className="text-smoke text-xs">Not enough review data yet</p>
      </div>
    );
  }

  return (
    <div className="rounded-md border border-white/5 bg-panel p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="font-display text-sm text-ivory">{capitalized}</p>
        <span className="font-mono text-xs text-smoke">{data.positive_pct}%</span>
      </div>
      <div className="h-1.5 w-full bg-panel-raised rounded-full overflow-hidden mb-2">
        <div
          className={`h-full ${LABEL_COLOR[data.label] || "bg-smoke"} rounded-full transition-all duration-700`}
          style={{ width: `${data.positive_pct}%` }}
        />
      </div>
      <p className="text-smoke text-xs">{data.label}</p>
    </div>
  );
}
