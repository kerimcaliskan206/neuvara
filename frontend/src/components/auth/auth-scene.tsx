"use client";

import { motion } from "framer-motion";
import { ShieldCheck } from "lucide-react";
import { useEffect, useState } from "react";

// ── Phase ─────────────────────────────────────────────────────────────────────
type Phase = "intro" | "forming" | "sweep" | "settle" | "login";

// ── Neural field ──────────────────────────────────────────────────────────────
const NN_NODES: [number, number][] = [
  [10, 14], [23,  9], [38, 18], [53, 11], [67, 20], [82, 10], [94, 17],
  [ 6, 35], [19, 42], [34, 49], [50, 43], [64, 33], [79, 42], [92, 30],
  [13, 60], [27, 67], [43, 62], [58, 70], [73, 57], [88, 64],
  [18, 83], [36, 79], [52, 87], [68, 80], [85, 85],
  [46, 26], [30, 31], [62, 52], [76, 27],
];

const NN_THRESHOLD = 21;
const NN_EDGES: [number, number, number, number][] = (() => {
  const edges: [number, number, number, number][] = [];
  for (let i = 0; i < NN_NODES.length; i++) {
    for (let j = i + 1; j < NN_NODES.length; j++) {
      const [ax, ay] = NN_NODES[i]!;
      const [bx, by] = NN_NODES[j]!;
      const dx = ax - bx; const dy = ay - by;
      if (dx * dx + dy * dy < NN_THRESHOLD * NN_THRESHOLD) edges.push([ax, ay, bx, by]);
    }
  }
  return edges;
})();

const NN_PULSE: [number, number, number][] = [
  [34, 49, 0], [50, 43, 2.4], [64, 33, 4.8], [46, 26, 1.2], [62, 52, 3.6],
];

// ── Global convergence particles ──────────────────────────────────────────────
// [startOffsetX px, startOffsetY px, delay s, color]
const CONV: [number, number, number, string][] = [
  [-380, -180, 0.00, "hsl(196 100% 68%)"], [ 320, -210, 0.05, "hsl(258 84% 74%)"],
  [-300,  145, 0.10, "hsl(196 100% 68%)"], [ 360,   90, 0.03, "hsl(221 83% 70%)"],
  [-195,  265, 0.15, "hsl(258 84% 74%)"], [ 230,  245, 0.08, "hsl(196 100% 68%)"],
  [-420,  -30, 0.12, "hsl(221 83% 70%)"], [ 400,  -55, 0.18, "hsl(196 100% 68%)"],
  [   0, -295, 0.02, "hsl(258 84% 74%)"], [  40,  290, 0.14, "hsl(196 100% 68%)"],
  [-260, -280, 0.20, "hsl(221 83% 70%)"], [ 280, -260, 0.06, "hsl(196 100% 68%)"],
  [-160,  320, 0.16, "hsl(258 84% 74%)"], [ 180,  310, 0.04, "hsl(196 100% 68%)"],
  [-340, -100, 0.22, "hsl(221 83% 70%)"], [ 310,  175, 0.09, "hsl(196 100% 68%)"],
  [-100, -345, 0.07, "hsl(258 84% 74%)"], [ 120, -330, 0.19, "hsl(196 100% 68%)"],
  [-240,  190, 0.11, "hsl(221 83% 70%)"], [ 260, -170, 0.24, "hsl(258 84% 74%)"],
  [ -60,  360, 0.13, "hsl(196 100% 68%)"], [  80,  340, 0.01, "hsl(221 83% 70%)"],
  [-450,   60, 0.17, "hsl(196 100% 68%)"], [ 440,  -80, 0.21, "hsl(258 84% 74%)"],
];

// ── Background ambient particles ──────────────────────────────────────────────
// [left%, top%, radius, peak-opacity, delay-s]
const PARTICLES: [number, number, number, number, number][] = [
  [ 8, 15, 1.5, 0.20, 0.0], [22, 72, 1.2, 0.15, 2.1], [35, 38, 1.8, 0.18, 4.5],
  [18, 55, 1.0, 0.13, 1.3], [44, 88, 1.4, 0.17, 3.7], [12, 45, 1.1, 0.14, 6.2],
  [30, 22, 1.6, 0.19, 0.8], [42, 65, 1.3, 0.16, 5.0], [ 6, 80, 1.2, 0.13, 2.8],
  [28, 10, 1.5, 0.18, 1.5], [48, 50, 1.0, 0.12, 7.1], [15, 92, 1.3, 0.15, 3.3],
  [38, 78, 1.1, 0.13, 4.8], [25, 33, 1.4, 0.17, 0.3], [45, 18, 1.2, 0.14, 6.8],
  [10, 62, 1.6, 0.18, 2.5], [55, 82, 1.3, 0.14, 5.5], [70, 40, 1.1, 0.13, 8.0],
  [80, 20, 1.4, 0.16, 1.0], [92, 60, 1.2, 0.12, 4.0],
];

// ── Per-letter assembly sparks ─────────────────────────────────────────────────
// [offsetX, offsetY, relativeDelay (negative = arrives before letter), color]
const SPARK_OFFSETS: [number, number, number, string][] = [
  [-46, -28, -0.30, "hsl(196 100% 72%)"],
  [ 42, -38, -0.22, "hsl(258 84% 78%)"],
  [-34,  42, -0.16, "hsl(196 100% 72%)"],
  [ 52,  26, -0.10, "hsl(221 83% 74%)"],
  [-22, -54, -0.06, "hsl(258 84% 78%)"],
];

// ── Burst scatter sparks (polar pre-computed: [targetX, targetY, delay, color]) ─
const BURST_SPARKS: [number, number, number, string][] = [
  [ 180,    0, 0.00, "hsl(196 100% 74%)"],
  [ 130,   75, 0.04, "hsl(258 84% 80%)"],
  [ 105,  182, 0.02, "hsl(200 100% 86%)"],
  [   0,  165, 0.06, "hsl(196 100% 74%)"],
  [ -98,  169, 0.01, "hsl(258 84% 80%)"],
  [-137,   79, 0.05, "hsl(42  90% 72%)"],
  [-188,    0, 0.03, "hsl(196 100% 74%)"],
  [-150,  -87, 0.07, "hsl(258 84% 80%)"],
  [-102, -176, 0.02, "hsl(200 100% 86%)"],
  [   0, -162, 0.04, "hsl(196 100% 74%)"],
  [  96, -166, 0.06, "hsl(258 84% 80%)"],
  [ 153,  -89, 0.03, "hsl(42  90% 72%)"],
  [ 225,   60, 0.08, "hsl(196 100% 74%)"],
  [-215,  -57, 0.05, "hsl(258 84% 80%)"],
  [  63,  235, 0.01, "hsl(200 100% 86%)"],
  [ -61, -229, 0.07, "hsl(196 100% 74%)"],
];

// ── Left info panel bullets ────────────────────────────────────────────────────
const BULLETS: [string, string][] = [
  ["hsl(196 100% 68%)", "XRay tabanlı pulmoner risk skorlama"],
  ["hsl(258 84% 74%)", "GradCAM destekli görsel odak analizi"],
  ["hsl(196 100% 68%)", "Klinik verilerle birleşik AI değerlendirme"],
  ["hsl(258 84% 74%)", "Türkçe klinik açıklama ve AI asistan desteği"],
];

// ── Wordmark ──────────────────────────────────────────────────────────────────
const LETTERS = "NEURAVA".split("");
const GRAD =
  "linear-gradient(145deg, hsl(200 100% 93%) 0%, hsl(210 90% 97%) 28%, hsl(232 72% 94%) 55%, hsl(258 84% 88%) 100%)";

function NeuravaWordmark({ phase }: { phase: Phase }) {
  const isBig    = phase === "forming" || phase === "sweep";
  const isSmall  = phase === "settle"  || phase === "login";
  const isActive = phase !== "intro";

  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 flex items-center justify-center"
      style={{ zIndex: 5 }}
    >
      {/* Ambient halo bloom */}
      <motion.div
        className="absolute"
        style={{
          width: "82vw", height: "58vh",
          background:
            "radial-gradient(ellipse at center, hsl(221 83% 44% / 0.28) 0%, hsl(258 84% 54% / 0.14) 45%, transparent 72%)",
          filter: "blur(64px)",
        }}
        animate={
          !isActive ? { opacity: 0, scale: 0.8 } :
          isBig     ? { opacity: 1, scale: 1 } :
          { opacity: [0.38, 0.58, 0.38] }
        }
        transition={
          phase === "forming" ? { opacity: { duration: 2.2 }, scale: { duration: 1.8 } } :
          phase === "settle"  ? { opacity: { duration: 1.0 }, scale: { duration: 1.0 } } :
          phase === "login"   ? { opacity: { duration: 5.5, repeat: Infinity, ease: "easeInOut" as const } } :
          { duration: 0.4 }
        }
      />

      {/* Global convergence particles — forming phase only */}
      {phase === "forming" && CONV.map(([ox, oy, delay, color], i) => (
        <motion.div
          key={i}
          style={{
            position: "absolute", top: "50%", left: "50%",
            marginTop: -2, marginLeft: -2,
            width: 4, height: 4, borderRadius: "50%",
            background: color, boxShadow: `0 0 5px 1px ${color}`,
          }}
          initial={{ x: ox, y: oy, opacity: 0 }}
          animate={{ x: 0, y: 0, opacity: [0, 0.85, 0] }}
          transition={{ duration: 1.30, delay, ease: "easeIn" as const }}
        />
      ))}

      {/* Wordmark container — scales down to background watermark on settle/login */}
      <motion.div
        style={{ position: "relative" }}
        animate={isSmall ? { scale: 0.55 } : { scale: 1 }}
        transition={{ duration: 0.90, ease: [0.22, 1, 0.36, 1] as const }}
      >
        {/* Letters */}
        <div style={{ display: "flex", alignItems: "center" }}>
          {LETTERS.map((letter, i) => {
            const d = 0.08 + i * 0.22;

            return (
              <span
                key={i}
                style={{
                  position: "relative",
                  display: "inline-block",
                  marginRight: i < LETTERS.length - 1 ? "0.28em" : 0,
                }}
              >
                {/* Node activation ring */}
                <motion.span
                  aria-hidden
                  style={{
                    position: "absolute", top: "50%", left: "50%",
                    marginTop: -15, marginLeft: -15,
                    width: 30, height: 30, borderRadius: "50%",
                    border: "1.5px solid hsl(196 100% 68%)",
                    display: "block", pointerEvents: "none",
                  }}
                  initial={{ scale: 0.3, opacity: 0 }}
                  animate={isBig
                    ? { scale: [0.3, 2.5, 2.5], opacity: [0, 0.55, 0] }
                    : { scale: 0.3, opacity: 0 }
                  }
                  transition={{ duration: 0.85, delay: d, ease: "easeOut" as const }}
                />

                {/* Per-letter assembly sparks */}
                {phase === "forming" && SPARK_OFFSETS.map(([ox, oy, relD, color], j) => (
                  <motion.span
                    key={j}
                    aria-hidden
                    style={{
                      position: "absolute", top: "50%", left: "50%",
                      marginTop: -1.5, marginLeft: -1.5,
                      width: 3, height: 3, borderRadius: "50%",
                      background: color, boxShadow: `0 0 4px 1.5px ${color}`,
                      display: "block", pointerEvents: "none",
                    }}
                    initial={{ x: ox, y: oy, opacity: 0 }}
                    animate={{ x: 0, y: 0, opacity: [0, 0.82, 0] }}
                    transition={{ duration: 0.52, delay: Math.max(0, d + relD), ease: "easeIn" as const }}
                  />
                ))}

                {/* Letter */}
                <motion.span
                  style={{
                    display: "inline-block",
                    fontSize: "clamp(3.5rem, 11vw, 9.5rem)",
                    fontWeight: 300, lineHeight: 1, userSelect: "none",
                    background: GRAD,
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                  initial={{ opacity: 0, filter: "blur(20px)", y: 10, scale: 0.82 }}
                  animate={
                    !isActive
                      ? { opacity: 0, filter: "blur(20px)", y: 10, scale: 0.82 }
                      : isBig
                        ? { opacity: 1, filter: "blur(0px)", y: 0, scale: 1 }
                        : { opacity: [0.12, 0.18, 0.12], filter: "blur(0px)", y: 0, scale: 1 }
                  }
                  transition={
                    isBig ? {
                      opacity: { duration: 1.65, delay: d, ease: "easeOut" as const },
                      filter:  { duration: 1.45, delay: d, ease: "easeOut" as const },
                      y:       { duration: 1.45, delay: d, ease: [0.22, 1, 0.36, 1] as const },
                      scale:   { duration: 1.45, delay: d, ease: [0.22, 1, 0.36, 1] as const },
                    } :
                    isSmall ? {
                      opacity: { duration: 6.5, repeat: Infinity, ease: "easeInOut" as const, delay: i * 0.10 },
                      filter:  { duration: 0.80, ease: "easeOut" as const },
                      y:       { duration: 0.80, ease: "easeOut" as const },
                      scale:   { duration: 0.80, ease: "easeOut" as const },
                    } : {}
                  }
                >
                  {letter}
                </motion.span>
              </span>
            );
          })}
        </div>

        {/* Tagline */}
        <motion.p
          style={{
            textAlign: "center", fontSize: "0.55rem", fontWeight: 500,
            letterSpacing: "0.42em", textTransform: "uppercase",
            color: "hsl(196 100% 72% / 0.40)", marginTop: "0.60em", userSelect: "none",
          }}
          animate={{ opacity: isBig ? 1 : 0 }}
          transition={{ opacity: { duration: 1.0, delay: isBig ? 1.6 : 0.15 } }}
        >
          Hanta AI Platform
        </motion.p>

        {/* ── Burst / sweep effects — only during "sweep" phase ── */}
        {phase === "sweep" && (
          <>
            {/* 1. Central radial burst — expands from wordmark center */}
            <motion.div
              style={{
                position: "absolute", top: "50%", left: "50%",
                width: "110%", height: "260%",
                marginTop: "-130%", marginLeft: "-55%",
                background:
                  "radial-gradient(ellipse at center, hsl(196 100% 84% / 0.62) 0%, hsl(258 84% 74% / 0.38) 28%, hsl(221 83% 65% / 0.16) 58%, transparent 78%)",
                filter: "blur(22px)", pointerEvents: "none",
              }}
              initial={{ scale: 0.12, opacity: 0 }}
              animate={{ scale: [0.12, 2.4, 1.6], opacity: [0, 1, 0] }}
              transition={{ duration: 0.72, ease: "easeOut" as const }}
            />

            {/* 2. Colorful shimmer sweep — brighter / more colorful than before */}
            <motion.div
              style={{
                position: "absolute", top: 0, bottom: 0, left: 0,
                width: "48%",
                background:
                  "radial-gradient(ellipse 45% 100% at 50% 50%, hsl(200 100% 96% / 0.52) 0%, hsl(196 100% 82% / 0.32) 30%, hsl(258 84% 80% / 0.18) 62%, transparent 80%)",
                filter: "blur(12px)", pointerEvents: "none",
              }}
              initial={{ x: "-100%", opacity: 0 }}
              animate={{ x: "260%", opacity: [0, 0.95, 0.95, 0] }}
              transition={{ duration: 0.88, delay: 0.08, ease: "easeInOut" as const }}
            />

            {/* 3. Thin lens flare line — horizontal bright streak */}
            <motion.div
              style={{
                position: "absolute", top: "44%", left: 0, right: 0,
                height: 3, marginTop: -1.5,
                background:
                  "linear-gradient(to right, transparent 0%, hsl(196 100% 88% / 0.72) 18%, hsl(210 100% 97% / 0.96) 50%, hsl(258 84% 88% / 0.72) 82%, transparent 100%)",
                filter: "blur(1.5px)", pointerEvents: "none",
              }}
              initial={{ x: "-110%", opacity: 0 }}
              animate={{ x: "110%", opacity: [0, 1, 1, 0] }}
              transition={{ duration: 0.55, delay: 0.06, ease: "easeOut" as const }}
            />

            {/* 4. Primary shockwave ring — cyan */}
            <motion.div
              style={{
                position: "absolute", top: "50%", left: "50%",
                marginTop: -18, marginLeft: -18,
                width: 36, height: 36, borderRadius: "50%",
                border: "1.5px solid hsl(196 100% 80%)",
                filter: "blur(1px)", pointerEvents: "none",
              }}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: [0.4, 5.0, 5.0], opacity: [0, 0.75, 0] }}
              transition={{ duration: 0.82, ease: "easeOut" as const }}
            />

            {/* 5. Secondary shockwave ring — violet, slightly delayed */}
            <motion.div
              style={{
                position: "absolute", top: "50%", left: "50%",
                marginTop: -14, marginLeft: -14,
                width: 28, height: 28, borderRadius: "50%",
                border: "1px solid hsl(258 84% 80%)",
                filter: "blur(1px)", pointerEvents: "none",
              }}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: [0.4, 4.2, 4.2], opacity: [0, 0.55, 0] }}
              transition={{ duration: 0.78, delay: 0.14, ease: "easeOut" as const }}
            />

            {/* 6. Large echo ring — outermost corona */}
            <motion.div
              style={{
                position: "absolute", top: "50%", left: "50%",
                marginTop: -22, marginLeft: -22,
                width: 44, height: 44, borderRadius: "50%",
                border: "1px solid hsl(196 100% 72% / 0.45)",
                filter: "blur(2px)", pointerEvents: "none",
              }}
              initial={{ scale: 0.3, opacity: 0 }}
              animate={{ scale: [0.3, 7.5, 7.5], opacity: [0, 0.40, 0] }}
              transition={{ duration: 1.05, delay: 0.18, ease: "easeOut" as const }}
            />

            {/* 7. Scatter sparks — fly outward from wordmark center */}
            {BURST_SPARKS.map(([tx, ty, delay, color], i) => (
              <motion.div
                key={i}
                style={{
                  position: "absolute", top: "50%", left: "50%",
                  marginTop: -1.5, marginLeft: -1.5,
                  width: 3, height: 3, borderRadius: "50%",
                  background: color, boxShadow: `0 0 5px 2px ${color}`,
                  pointerEvents: "none",
                }}
                initial={{ x: 0, y: 0, scale: 1, opacity: 0 }}
                animate={{ x: tx, y: ty, scale: [1, 0.4, 0], opacity: [0, 0.92, 0] }}
                transition={{ duration: 0.90, delay, ease: "easeOut" as const }}
              />
            ))}
          </>
        )}
      </motion.div>
    </div>
  );
}

// ── Auth scene ─────────────────────────────────────────────────────────────────

export function AuthScene({ children }: { children: React.ReactNode }) {
  const [phase, setPhase] = useState<Phase>("intro");

  useEffect(() => {
    const timers = [
      setTimeout(() => setPhase("forming"), 500),
      setTimeout(() => setPhase("sweep"),   3200),
      setTimeout(() => setPhase("settle"),  4400),
      setTimeout(() => setPhase("login"),   5000),
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  const showLogin = phase === "login";

  return (
    <div
      className="relative min-h-screen overflow-hidden"
      style={{ background: "hsl(222 50% 4%)" }}
    >

      {/* ── BG-1: Atmosphere orbs ── */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div className="atmosphere-orb" style={{
          top: "-14%", right: "-8%", width: 760, height: 760,
          background: "radial-gradient(circle at 38% 38%, hsl(258 84% 62% / 0.44) 0%, transparent 60%)",
          opacity: 0.80,
        }} />
        <div className="atmosphere-orb" style={{
          bottom: "-18%", left: "-10%", width: 820, height: 820,
          background: "radial-gradient(circle at 34% 34%, hsl(221 83% 55% / 0.32) 0%, transparent 60%)",
          opacity: 0.72, animationDelay: "7s",
        }} />
        <div className="atmosphere-orb" style={{
          top: "18%", left: "18%", width: 560, height: 560,
          background: "radial-gradient(circle at 50% 50%, hsl(196 100% 65% / 0.10) 0%, transparent 68%)",
          opacity: 1, animationDelay: "3.5s",
        }} />
        <div className="atmosphere-orb" style={{
          bottom: "5%", right: "12%", width: 400, height: 400,
          background: "radial-gradient(circle at 50% 50%, hsl(258 84% 60% / 0.12) 0%, transparent 70%)",
          opacity: 0.9, animationDelay: "11s",
        }} />
      </div>

      {/* ── BG-2: Central bridge glow ── */}
      <div aria-hidden className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <motion.div
          style={{
            width: "88vw", height: "72vh",
            background:
              "radial-gradient(ellipse 70% 60% at 50% 50%, hsl(221 83% 38% / 0.14) 0%, hsl(258 84% 48% / 0.08) 42%, transparent 70%)",
            filter: "blur(50px)",
          }}
          animate={{ scale: [1, 1.05, 1], opacity: [0.70, 1, 0.70] }}
          transition={{ duration: 16, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {/* ── BG-3: Right panel backing glow ── */}
      <div aria-hidden style={{
        position: "absolute", top: "50%", right: "3%",
        transform: "translateY(-50%)",
        width: 600, height: 740, pointerEvents: "none",
        background:
          "radial-gradient(ellipse 80% 70% at 55% 50%, hsl(258 84% 56% / 0.14) 0%, hsl(221 83% 48% / 0.08) 48%, transparent 72%)",
        filter: "blur(64px)",
      }} />

      {/* ── BG-4: Waveforms ── */}
      <motion.svg
        aria-hidden viewBox="0 0 1440 900" preserveAspectRatio="xMidYMid slice"
        className="pointer-events-none absolute inset-0 h-full w-full"
        animate={{ opacity: [0.55, 0.80, 0.55] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
      >
        <defs>
          <linearGradient id="wf-g1" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%"   stopColor="transparent" />
            <stop offset="16%"  stopColor="hsl(196 100% 68%)" stopOpacity={0.12} />
            <stop offset="48%"  stopColor="hsl(196 100% 68%)" stopOpacity={0.22} />
            <stop offset="80%"  stopColor="hsl(258 84% 72%)"  stopOpacity={0.12} />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
          <linearGradient id="wf-g2" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%"   stopColor="transparent" />
            <stop offset="20%"  stopColor="hsl(258 84% 72%)"  stopOpacity={0.07} />
            <stop offset="58%"  stopColor="hsl(196 100% 68%)" stopOpacity={0.11} />
            <stop offset="100%" stopColor="transparent" />
          </linearGradient>
        </defs>
        <path d="M -80,438 C 80,410 180,466 360,440 C 540,414 640,470 820,446 C 1000,422 1100,476 1280,450 C 1380,437 1520,446 1520,446"
          fill="none" stroke="url(#wf-g1)" strokeWidth="1.5" />
        <path d="M -80,548 C 120,526 240,564 420,540 C 600,516 700,558 880,534 C 1060,510 1160,556 1340,534 C 1420,523 1560,534 1560,534"
          fill="none" stroke="url(#wf-g2)" strokeWidth="0.9" />
      </motion.svg>

      {/* ── BG-5: Neural field ── */}
      <motion.svg
        aria-hidden viewBox="0 0 100 100" preserveAspectRatio="xMidYMid slice"
        className="pointer-events-none absolute inset-0 h-full w-full"
        style={{ filter: "blur(0.3px)" }}
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        transition={{ duration: 3.0, delay: 0.4 }}
      >
        {NN_EDGES.map(([x1, y1, x2, y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={i % 3 === 0 ? "hsl(196 100% 68%)" : i % 3 === 1 ? "hsl(258 84% 74%)" : "hsl(221 83% 70%)"}
            strokeWidth="0.07" opacity={0.14} />
        ))}
        {NN_NODES.map(([cx, cy], i) => (
          <circle key={i} cx={cx} cy={cy}
            r={i % 5 === 0 ? 0.26 : 0.16}
            fill={i % 3 === 0 ? "hsl(196 100% 72%)" : i % 3 === 1 ? "hsl(258 84% 78%)" : "hsl(221 83% 72%)"}
            opacity={0.28} />
        ))}
        {NN_PULSE.map(([cx, cy, delay], i) => (
          <motion.circle key={i} cx={cx} cy={cy} fill="hsl(196 100% 74%)"
            animate={{ r: [0.18, 0.36, 0.18], opacity: [0.24, 0.52, 0.24] }}
            transition={{ duration: 3.8 + i * 0.9, delay, repeat: Infinity, ease: "easeInOut" }} />
        ))}
      </motion.svg>

      {/* ── BG-6: Vignette + dot lattice ── */}
      <div aria-hidden className="pointer-events-none absolute inset-0">
        <div style={{
          position: "absolute", inset: 0,
          background: "radial-gradient(ellipse 90% 88% at 50% 50%, transparent 26%, hsl(222 50% 2% / 0.72) 100%)",
        }} />
        <div className="atmosphere-grid" style={{ color: "hsl(0 0% 100% / 0.035)" }} />
      </div>

      {/* ── BG-7: Ambient particles ── */}
      <div aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
        {PARTICLES.map(([left, top, r, op, delay], i) => (
          <motion.div key={i}
            style={{
              position: "absolute", left: `${left}%`, top: `${top}%`,
              width: r * 5, height: r * 5, borderRadius: "50%",
              background:
                i % 3 === 0 ? "hsl(196 100% 68%)" :
                i % 3 === 1 ? "hsl(258 84% 75%)" : "hsl(221 83% 70%)",
              filter: `blur(${r * 0.9}px)`,
            }}
            animate={{ y: [0, -52, 0], opacity: [0, op, 0] }}
            transition={{ duration: 9 + delay * 1.3, delay, repeat: Infinity, ease: "easeInOut" }}
          />
        ))}
      </div>

      {/* ── Right-edge light leak ── */}
      <div aria-hidden style={{
        position: "absolute", top: 0, right: 0, width: 1, height: "100%",
        pointerEvents: "none", zIndex: 2,
        background:
          "linear-gradient(to bottom, transparent 8%, hsl(221 83% 53% / 0.30) 36%, hsl(258 84% 65% / 0.18) 68%, transparent 92%)",
      }} />

      {/* ── NEURAVA cinematic wordmark ── */}
      <NeuravaWordmark phase={phase} />

      {/* ── Logo — always visible top-left ── */}
      <div className="absolute top-10 left-10 z-20 hidden items-center gap-3.5 lg:flex">
        <div
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl"
          style={{
            background: "linear-gradient(135deg, hsl(221 83% 53%), hsl(258 84% 65%))",
            boxShadow: "0 0 24px -4px hsl(221 83% 53% / 0.55)",
          }}
        >
          <ShieldCheck className="h-4.5 w-4.5 text-white" />
        </div>
        <div>
          <p className="text-lg font-bold tracking-[0.26em] text-white">NEURAVA</p>
          <p className="text-[10px] font-semibold uppercase tracking-[0.20em] text-white/36">
            Hanta AI Platform
          </p>
        </div>
      </div>

      {/* ── Mobile logo ── */}
      <div className="absolute top-8 left-6 z-20 flex items-center gap-2.5 lg:hidden">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg"
          style={{ background: "linear-gradient(135deg, hsl(221 83% 53%), hsl(258 84% 65%))" }}>
          <ShieldCheck className="h-4 w-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold tracking-[0.22em] text-white">NEURAVA</p>
          <p className="text-[10px] uppercase tracking-widest text-white/35">Hanta AI Platform</p>
        </div>
      </div>

      {/* ── Two-column layout — NOT mounted until intro completes ── */}
      {/* Password/email inputs must not exist in the DOM during the cinematic  */}
      {/* intro. Conditional mount (not opacity:0) is the only reliable way to  */}
      {/* prevent Safari/Chrome from triggering the autofill popup.             */}
      {showLogin && (
        <div className="relative z-10 flex min-h-screen items-stretch">

          {/* ── Left: Hanta info panel ── */}
          <div className="hidden lg:flex w-[52%] items-center px-16 xl:px-20">
            <motion.div
              initial={{ opacity: 0, x: -22, filter: "blur(8px)" }}
              animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
              transition={{ duration: 0.90, ease: [0.22, 1, 0.36, 1] as const }}
              style={{ maxWidth: 448 }}
            >
              {/* Top accent rule */}
              <div style={{
                width: 36, height: 2, borderRadius: 1, marginBottom: "1.4rem",
                background: "linear-gradient(to right, hsl(196 100% 68%), hsl(258 84% 74%))",
              }} />

              {/* Title */}
              <h2 style={{
                fontSize: "clamp(1.30rem, 2.0vw, 1.72rem)",
                fontWeight: 600, letterSpacing: "0.005em",
                color: "hsl(220 30% 96%)", lineHeight: 1.28,
                marginBottom: "0.85rem",
              }}>
                Hanta Virüs Karar Destek Sistemi
              </h2>

              {/* Subtitle */}
              <p style={{
                fontSize: "0.865rem",
                color: "hsl(220 18% 62%)",
                lineHeight: 1.70, marginBottom: "1.85rem",
              }}>
                Hantavirüs ve pulmoner bulgular için yapay zeka destekli klinik analiz platformu.
              </p>

              {/* Bullets */}
              <ul style={{ display: "flex", flexDirection: "column", gap: "0.82rem" }}>
                {BULLETS.map(([dotColor, text], i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.55, delay: 0.16 + i * 0.08, ease: "easeOut" as const }}
                    style={{ display: "flex", alignItems: "flex-start", gap: "0.72rem" }}
                  >
                    <span style={{
                      marginTop: "0.36em", flexShrink: 0,
                      width: 5, height: 5, borderRadius: "50%",
                      background: dotColor,
                      boxShadow: `0 0 5px 1px ${dotColor.replace(")", " / 0.45)")}`,
                      display: "inline-block",
                    }} />
                    <span style={{
                      fontSize: "0.820rem",
                      color: "hsl(220 18% 60%)",
                      lineHeight: 1.62,
                    }}>
                      {text}
                    </span>
                  </motion.li>
                ))}
              </ul>

              {/* Bottom accent line */}
              <div style={{
                marginTop: "2.2rem", height: 1,
                background:
                  "linear-gradient(to right, hsl(196 100% 68% / 0.22), hsl(258 84% 74% / 0.14), transparent)",
              }} />
            </motion.div>
          </div>

          {/* ── Right: auth card ── */}
          <div className="flex flex-1 items-center justify-center px-6 py-12 lg:px-12">
            <motion.div
              className="w-full max-w-[420px]"
              initial={{ opacity: 0, y: 28, scale: 0.96, filter: "blur(10px)" }}
              animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
              transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] as const }}
            >
              {/* Glass card */}
              <div
                className="relative w-full rounded-2xl px-8 py-9"
                style={{
                  background: "rgba(6, 10, 26, 0.80)",
                  border: "1px solid rgba(255, 255, 255, 0.09)",
                  backdropFilter: "blur(36px)",
                  WebkitBackdropFilter: "blur(36px)",
                  boxShadow: [
                    "0 0 0 1px rgba(255,255,255,0.04) inset",
                    "0 0 80px -20px hsl(221 83% 53% / 0.24)",
                    "0 0 48px -14px hsl(258 84% 65% / 0.18)",
                    "0 40px 80px -28px rgba(0,0,0,0.75)",
                  ].join(", "),
                }}
              >
                {/* Top-edge shimmer */}
                <div aria-hidden style={{
                  position: "absolute", inset: 0, borderRadius: "inherit", pointerEvents: "none",
                  background: "linear-gradient(180deg, rgba(255,255,255,0.055) 0%, transparent 12%)",
                }} />
                {/* Left cyan inner glow */}
                <div aria-hidden style={{
                  position: "absolute", inset: 0, borderRadius: "inherit", pointerEvents: "none",
                  background:
                    "radial-gradient(ellipse 140% 100% at -10% 50%, hsl(196 100% 68% / 0.050) 0%, transparent 52%)",
                }} />

                {/* Card header */}
                <div className="mb-7 flex items-center gap-2.5">
                  <div
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg"
                    style={{
                      background: "linear-gradient(135deg, hsl(221 83% 53%), hsl(258 84% 65%))",
                      boxShadow: "0 0 16px -2px hsl(221 83% 53% / 0.50)",
                    }}
                  >
                    <ShieldCheck className="h-3.5 w-3.5 text-white" />
                  </div>
                  <div>
                    <p className="text-[11px] font-bold tracking-[0.18em] text-white/90">NEURAVA</p>
                    <p className="text-[9px] uppercase tracking-[0.16em] text-white/30">
                      Güvenli Klinik Erişim
                    </p>
                  </div>
                </div>

                {children}

                {/* Secure label */}
                <div className="mt-7 flex items-center justify-center gap-1.5 border-t border-white/[0.06] pt-5">
                  <div className="h-1 w-1 rounded-full bg-emerald-400 opacity-70" />
                  <p className="text-[10px] tracking-wide text-white/22">
                    Şifreli · Güvenli · Klinik Amaçlı
                  </p>
                </div>
              </div>
            </motion.div>
          </div>

        </div>
      )}
    </div>
  );
}
