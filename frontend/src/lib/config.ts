/**
 * Centralised, typed access to NEXT_PUBLIC_* environment variables.
 *
 * Every URL, limit, and feature flag the app uses must come from here —
 * no `process.env.NEXT_PUBLIC_*` reads anywhere else. That gives us a single
 * place to validate values, apply defaults, and surface misconfiguration.
 */

function readString(name: string, fallback: string): string {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  return raw;
}

function readNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readBoolean(name: string, fallback: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  return raw === "1" || raw.toLowerCase() === "true";
}

function readList(name: string, fallback: string[]): string[] {
  const raw = process.env[name];
  if (!raw) return fallback;
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

const API_BASE = readString("NEXT_PUBLIC_API_BASE_URL", "http://localhost:8000").replace(/\/+$/, "");
const API_V1 = readString("NEXT_PUBLIC_API_V1_PREFIX", "/api/v1");
const AI_BASE_OVERRIDE = readString("NEXT_PUBLIC_AI_BASE_URL", "");

export const config = {
  api: {
    baseUrl: API_BASE,
    v1Prefix: API_V1,
    /** Fully-qualified base for the v1 REST API. */
    v1: `${API_BASE}${API_V1}`,
    /** Fully-qualified base for AI endpoints — separate so it can be pointed at another host. */
    ai: AI_BASE_OVERRIDE || `${API_BASE}${API_V1}/ai`,
    timeoutMs: readNumber("NEXT_PUBLIC_API_TIMEOUT_MS", 20_000),
    aiTimeoutMs: readNumber("NEXT_PUBLIC_AI_TIMEOUT_MS", 60_000),
  },
  upload: {
    maxMb: readNumber("NEXT_PUBLIC_MAX_UPLOAD_MB", 10),
    allowedMimeTypes: readList("NEXT_PUBLIC_ALLOWED_MIME_TYPES", [
      "image/jpeg",
      "image/png",
      "image/webp",
    ]),
  },
  features: {
    gradcam: readBoolean("NEXT_PUBLIC_ENABLE_GRADCAM", true),
    aiChat: readBoolean("NEXT_PUBLIC_ENABLE_AI_CHAT", true),
    devLogs: readBoolean("NEXT_PUBLIC_ENABLE_DEV_LOGS", false),
  },
} as const;

export type AppConfig = typeof config;
