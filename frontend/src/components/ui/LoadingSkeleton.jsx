export default function LoadingSkeleton({ count = 5 }) {
  return (
    <div className="flex gap-4 px-6 md:px-12 pb-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="shrink-0 w-48 md:w-56">
          <div className="aspect-[2/3] rounded-md bg-panel animate-pulse" />
          <div className="h-4 w-3/4 mt-3 rounded bg-panel animate-pulse" />
        </div>
      ))}
    </div>
  );
}
