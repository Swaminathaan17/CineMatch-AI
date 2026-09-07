import { SearchX } from "lucide-react";

export default function EmptyState({ icon, title, message, action }) {
  const Icon = icon || SearchX;

  return (
    <div className="py-16 flex flex-col items-center text-center px-4">
      <div className="w-12 h-12 rounded-full border border-white/10 bg-panel flex items-center justify-center mb-4">
        <Icon size={20} className="text-smoke" />
      </div>
      <p className="font-display text-lg text-ivory mb-1.5">{title}</p>
      {message && <p className="text-smoke text-sm max-w-sm leading-relaxed">{message}</p>}
      {action}
    </div>
  );
}