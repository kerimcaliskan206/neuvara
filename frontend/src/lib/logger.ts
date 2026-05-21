import { config } from "@/lib/config";

type Level = "debug" | "info" | "warn" | "error";

function emit(level: Level, message: string, meta?: unknown) {
  if (level === "debug" && !config.features.devLogs) return;
  const ts = new Date().toISOString();
  const line = `[${ts}] [${level.toUpperCase()}] ${message}`;
  switch (level) {
    case "debug":
    case "info":
      console.log(line, meta ?? "");
      break;
    case "warn":
      console.warn(line, meta ?? "");
      break;
    case "error":
      console.error(line, meta ?? "");
      break;
  }
}

export const logger = {
  debug: (message: string, meta?: unknown) => emit("debug", message, meta),
  info: (message: string, meta?: unknown) => emit("info", message, meta),
  warn: (message: string, meta?: unknown) => emit("warn", message, meta),
  error: (message: string, meta?: unknown) => emit("error", message, meta),
};
