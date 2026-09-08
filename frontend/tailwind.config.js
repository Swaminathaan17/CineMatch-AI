/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0B0B0E",
        curtain: "#8B2942",
        "curtain-dim": "#5C1B2C",
        gold: "#C9A227",
        "gold-soft": "#E4C766",
        ivory: "#EDEAE4",
        smoke: "#9A968E",
        panel: "#151417",
        "panel-raised": "#1D1B1F",
      },
      fontFamily: {
        display: ["'Fraunces'", "serif"],
        body: ["'Inter'", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
