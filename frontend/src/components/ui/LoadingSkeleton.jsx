export default function LoadingSkeleton({
  count = 5,
  variant = "grid",
  className = "",
}) {
  if (variant === "row") {
    return (
      <div className="flex gap-4 px-6 md:px-12 pb-4">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="shrink-0 w-48 md:w-56 skeleton-card">
            <div className="aspect-[2/3] rounded-md bg-panel-raised relative overflow-hidden">
              <div className="skeleton-shimmer absolute inset-0" />
            </div>
            <div className="h-3 w-3/4 mt-3 rounded bg-panel-raised relative overflow-hidden">
              <div className="skeleton-shimmer absolute inset-0" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={`grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 ${className}`}>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="skeleton-card">
          <div className="aspect-[2/3] rounded-lg bg-panel-raised relative overflow-hidden">
            <div className="skeleton-shimmer absolute inset-0" />
          </div>
          <div className="h-3 w-4/5 mt-3 rounded bg-panel-raised relative overflow-hidden">
            <div className="skeleton-shimmer absolute inset-0" />
          </div>
          <div className="h-2.5 w-3/5 mt-2 rounded bg-panel-raised relative overflow-hidden">
            <div className="skeleton-shimmer absolute inset-0" />
          </div>
        </div>
      ))}
    </div>
  );
}