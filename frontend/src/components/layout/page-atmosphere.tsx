import { cn } from "@/lib/utils";

export type AtmosphereVariant =
  | "neural"
  | "biotech"
  | "cinematic"
  | "node"
  | "medical"
  | "minimal";

interface OrbSpec {
  hue: string;
  size: number;
  top?: string;
  left?: string;
  right?: string;
  bottom?: string;
  delay?: number;
  opacity?: number;
}

interface VariantConfig {
  baseColor: string;
  gridColor: string;
  orbs: OrbSpec[];
}

const VARIANTS: Record<AtmosphereVariant, VariantConfig> = {
  medical: {
    baseColor: "hsl(222 45% 4%)",
    gridColor: "hsl(221 60% 55% / 0.06)",
    orbs: [
      { hue: "221 90% 65%", size: 600, top: "-15%", left: "-10%",  delay: 0,  opacity: 0.18 },
      { hue: "195 85% 60%", size: 500, top: "25%",  right: "-12%", delay: 8,  opacity: 0.14 },
      { hue: "258 70% 70%", size: 420, bottom: "-10%", left: "30%",delay: 16, opacity: 0.12 },
    ],
  },
  neural: {
    baseColor: "hsl(222 45% 3.5%)",
    gridColor: "hsl(221 60% 55% / 0.08)",
    orbs: [
      { hue: "221 90% 65%", size: 680, top: "-15%", left: "-10%",  delay: 0,  opacity: 0.22 },
      { hue: "258 84% 72%", size: 560, top: "28%",  right: "-12%", delay: 6,  opacity: 0.17 },
      { hue: "199 89% 70%", size: 500, bottom: "-14%", left: "28%",delay: 12, opacity: 0.15 },
      { hue: "221 70% 50%", size: 380, bottom: "8%", right: "8%",  delay: 18, opacity: 0.10 },
    ],
  },
  biotech: {
    baseColor: "hsl(220 45% 4%)",
    gridColor: "hsl(189 65% 50% / 0.06)",
    orbs: [
      { hue: "189 94% 60%", size: 520, top: "-10%", left: "-5%",  delay: 0,  opacity: 0.18 },
      { hue: "172 80% 60%", size: 440, top: "35%",  right: "-6%", delay: 5,  opacity: 0.15 },
      { hue: "155 70% 65%", size: 380, bottom: "-15%", left: "25%",delay: 11, opacity: 0.13 },
    ],
  },
  cinematic: {
    baseColor: "hsl(225 45% 4%)",
    gridColor: "hsl(245 60% 55% / 0.07)",
    orbs: [
      { hue: "245 85% 68%", size: 580, top: "-15%", left: "-10%", delay: 0,  opacity: 0.20 },
      { hue: "180 75% 62%", size: 500, top: "25%",  right: "-8%", delay: 4,  opacity: 0.17 },
      { hue: "280 75% 72%", size: 420, bottom: "-10%", left: "35%",delay: 9, opacity: 0.14 },
    ],
  },
  node: {
    baseColor: "hsl(223 45% 4%)",
    gridColor: "hsl(221 65% 55% / 0.08)",
    orbs: [
      { hue: "221 90% 68%", size: 500, top: "-10%", left: "-5%",  delay: 0,  opacity: 0.18 },
      { hue: "258 84% 72%", size: 420, top: "40%",  right: "-8%", delay: 7,  opacity: 0.15 },
    ],
  },
  minimal: {
    baseColor: "hsl(222 45% 4%)",
    gridColor: "hsl(214 50% 55% / 0.05)",
    orbs: [
      { hue: "214 90% 68%", size: 480, top: "-10%", left: "-5%",  delay: 0,  opacity: 0.14 },
      { hue: "221 80% 72%", size: 400, bottom: "-15%", right: "-5%",delay: 6, opacity: 0.12 },
    ],
  },
};

export interface PageAtmosphereProps {
  variant?: AtmosphereVariant;
  className?: string;
}

export function PageAtmosphere({
  variant = "minimal",
  className,
}: PageAtmosphereProps) {
  const cfg = VARIANTS[variant];

  return (
    <div aria-hidden className={cn("atmosphere-layer", className)}>
      {/* Deep dark base */}
      <div
        className="absolute inset-0"
        style={{ backgroundColor: cfg.baseColor }}
      />

      {/* Subtle dot grid */}
      <div
        className="atmosphere-grid"
        style={{ color: cfg.gridColor }}
      />

      {/* Ambient color orbs */}
      {cfg.orbs.map((orb, i) => (
        <div
          key={i}
          className="atmosphere-orb"
          style={{
            width: orb.size,
            height: orb.size,
            top: orb.top,
            left: orb.left,
            right: orb.right,
            bottom: orb.bottom,
            opacity: orb.opacity ?? 0.15,
            background: `radial-gradient(circle at 30% 30%, hsl(${orb.hue}) 0%, transparent 70%)`,
            animationDelay: `${orb.delay ?? 0}s`,
          }}
        />
      ))}

      {/* Depth vignette */}
      <div className="atmosphere-vignette" />
    </div>
  );
}

export function atmosphereVariantForPath(pathname: string): AtmosphereVariant {
  if (pathname.startsWith("/medical"))   return "medical";
  if (pathname.startsWith("/dashboard")) return "neural";
  if (pathname.startsWith("/vision"))    return "biotech";
  if (pathname.startsWith("/fusion"))    return "cinematic";
  if (pathname.startsWith("/chat"))      return "node";
  return "minimal";
}
