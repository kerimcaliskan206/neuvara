"use client";

import { useEffect, useRef, useState } from "react";

interface UseCountUpOptions {
  duration?: number;        // ms
  decimals?: number;        // displayed decimals
  startOnMount?: boolean;
}

/**
 * Smoothly counts from 0 (or current value) toward `target`.
 * Uses requestAnimationFrame and an easeOutCubic curve.
 * Falls back to the immediate target value under prefers-reduced-motion.
 */
export function useCountUp(
  target: number,
  { duration = 700, decimals = 0, startOnMount = true }: UseCountUpOptions = {},
): number {
  const [value, setValue] = useState(startOnMount ? 0 : target);
  const fromRef = useRef(startOnMount ? 0 : target);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    if (reduceMotion) {
      setValue(target);
      fromRef.current = target;
      return;
    }

    const from = fromRef.current;
    const delta = target - from;
    if (Math.abs(delta) < 1e-6) {
      setValue(target);
      return;
    }

    const start = performance.now();
    const ease = (t: number) => 1 - Math.pow(1 - t, 3);

    function tick(now: number) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / duration);
      const next = from + delta * ease(t);
      setValue(next);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
      }
    }

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [target, duration]);

  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
