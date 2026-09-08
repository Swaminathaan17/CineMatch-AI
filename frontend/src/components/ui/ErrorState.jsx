import { AlertTriangle } from "lucide-react";

export default function ErrorState({ title = "Something went wrong", message, onRetry }) {
  const safeMessage =
    typeof message === "string" && message.trim() && !message.startsWith("API error")
      ? message
      : "We couldn't load this right now.";

  return (
    <div className="py-16 flex flex-col items-center text-center px-4">
      <div className="w-12 h-12 rounded-full border border-curtain/40 bg-curtain/10 flex items-center justify-center mb-4">
        <AlertTriangle size={20} className="text-curtain" />
      </div>
      <p className="font-display text-lg text-ivory mb-1.5">{title}</p>
      <p className="text-smoke text-sm max-w-sm leading-relaxed">{safeMessage}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-5 px-5 py-2.5 rounded-sm bg-curtain hover:bg-curtain-dim text-ivory font-medium transition-colors"
        >
          Try again
        </button>
      )}
    </div>
  );
}