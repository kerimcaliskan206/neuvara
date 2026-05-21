import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // ── Canvas / background layers (dark) ────────────────────────────────
        canvas: "hsl(222 45% 4%)",

        // ── Surface layers ───────────────────────────────────────────────────
        surface: {
          DEFAULT: "hsl(222 35% 7%)",
          raised: "hsl(222 30% 10%)",
          overlay: "hsl(222 28% 13%)",
        },

        // ── Borders ──────────────────────────────────────────────────────────
        border: {
          DEFAULT: "hsl(222 22% 17%)",
          subtle: "hsl(222 22% 12%)",
          strong: "hsl(222 22% 26%)",
        },

        // ── Text hierarchy (light on dark) ───────────────────────────────────
        foreground: {
          DEFAULT: "hsl(220 15% 92%)",
          secondary: "hsl(220 12% 62%)",
          muted: "hsl(220 10% 42%)",
          placeholder: "hsl(220 8% 30%)",
        },

        // ── Brand — clinical blue ────────────────────────────────────────────
        brand: {
          50:  "hsl(217 55% 11%)",   // dark tinted bg
          100: "hsl(216 50% 17%)",   // elevated dark bg
          200: "hsl(215 48% 26%)",   // border / separator
          300: "hsl(214 65% 48%)",   // icon / accent
          400: "hsl(213 78% 60%)",   // interactive hover
          500: "hsl(217 88% 65%)",   // vibrant accent
          600: "hsl(221 83% 53%)",   // primary action
          700: "hsl(224 88% 72%)",   // text on dark bg
          800: "hsl(226 90% 80%)",   // lighter text
          900: "hsl(228 92% 87%)",   // lightest tint
          DEFAULT: "hsl(221 83% 53%)",
          foreground: "hsl(0 0% 100%)",
        },

        // ── Success ──────────────────────────────────────────────────────────
        success: {
          50:  "hsl(152 35% 8%)",
          100: "hsl(149 35% 13%)",
          200: "hsl(149 35% 20%)",
          500: "hsl(152 65% 48%)",
          600: "hsl(153 60% 40%)",
          700: "hsl(152 65% 58%)",   // text on dark
          DEFAULT: "hsl(152 60% 40%)",
          foreground: "hsl(0 0% 100%)",
        },

        // ── Warning / caution ────────────────────────────────────────────────
        warning: {
          50:  "hsl(45 55% 8%)",
          100: "hsl(43 50% 14%)",
          200: "hsl(43 48% 22%)",
          500: "hsl(38 90% 58%)",
          600: "hsl(35 85% 50%)",
          700: "hsl(38 90% 70%)",    // text on dark
          DEFAULT: "hsl(38 90% 55%)",
          foreground: "hsl(30 40% 10%)",
        },

        // ── Danger ───────────────────────────────────────────────────────────
        danger: {
          50:  "hsl(0 45% 8%)",
          100: "hsl(0 45% 14%)",
          200: "hsl(0 45% 22%)",
          500: "hsl(0 80% 62%)",
          600: "hsl(0 75% 55%)",
          700: "hsl(0 78% 72%)",     // text on dark
          DEFAULT: "hsl(0 75% 58%)",
          foreground: "hsl(0 0% 100%)",
        },

        // ── Legacy aliases ───────────────────────────────────────────────────
        background: "hsl(222 45% 4%)",
        muted: {
          DEFAULT: "hsl(222 30% 11%)",
          foreground: "hsl(220 12% 62%)",
        },
        primary: {
          DEFAULT: "hsl(221 83% 53%)",
          foreground: "hsl(0 0% 100%)",
        },
      },

      borderRadius: {
        "2xl": "1rem",
        "3xl": "1.25rem",
      },

      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "ui-sans-serif", "sans-serif"],
      },

      fontSize: {
        "2xs": ["0.65rem", { lineHeight: "1rem" }],
      },

      boxShadow: {
        card:        "0 1px 4px 0 rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.04)",
        "card-hover":"0 4px 20px 0 rgba(0,0,0,.6), 0 2px 8px -2px rgba(0,0,0,.4)",
        "card-raised":"0 8px 32px 0 rgba(0,0,0,.7), 0 2px 12px -2px rgba(0,0,0,.5)",
        soft:        "0 2px 12px 0 rgba(0,0,0,.4)",
        brand:       "0 4px 20px 0 hsl(221 83% 53% / 0.45)",
        "glow-brand":  "0 0 60px -10px hsl(221 83% 53% / 0.6)",
        "glow-success":"0 0 60px -10px hsl(152 65% 48% / 0.5)",
        "glow-danger": "0 0 60px -10px hsl(0 78% 62% / 0.5)",
        "glow-warning":"0 0 60px -10px hsl(38 90% 58% / 0.4)",
      },

      transitionTimingFunction: {
        "swift-out": "cubic-bezier(0.2, 0, 0, 1)",
        "swift-in":  "cubic-bezier(0.4, 0, 1, 1)",
        "smooth":    "cubic-bezier(0.32, 0.72, 0, 1)",
      },

      animation: {
        "fade-up":        "fadeUp 0.4s cubic-bezier(0.2,0,0,1) both",
        "fade-in":        "fadeIn 0.3s ease-out both",
        "slide-in-left":  "slideInLeft 0.35s cubic-bezier(0.2,0,0,1) both",
        "scale-in":       "scaleIn 0.25s cubic-bezier(0.2,0,0,1) both",
        "pulse-slow":     "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "blink-dot":      "blinkDot 1.2s cubic-bezier(0.4,0,0.6,1) infinite",
        "spin-slow":      "spin 3s linear infinite",
        "shimmer-dark":   "shimmerDark 1.8s linear infinite",
      },

      keyframes: {
        fadeUp: {
          from: { opacity: "0", transform: "translateY(12px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to:   { opacity: "1" },
        },
        slideInLeft: {
          from: { opacity: "0", transform: "translateX(-10px)" },
          to:   { opacity: "1", transform: "translateX(0)" },
        },
        scaleIn: {
          from: { opacity: "0", transform: "scale(0.96)" },
          to:   { opacity: "1", transform: "scale(1)" },
        },
        blinkDot: {
          "0%, 80%, 100%": { opacity: "0.2",  transform: "translateY(0)" },
          "40%":           { opacity: "1",    transform: "translateY(-3px)" },
        },
        shimmerDark: {
          "0%":   { backgroundPosition: "-400% 0" },
          "100%": { backgroundPosition: "400% 0" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
