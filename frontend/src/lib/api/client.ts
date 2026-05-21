/**
 * Centralised Axios clients.
 *
 * Two instances:
 *   • `api`   — the standard v1 client (short timeout, JSON).
 *   • `aiApi` — pointed at the AI sub-tree with a longer timeout because
 *               Ollama can be slow on cold-start.
 *
 * Both share:
 *   - Bearer token injection from the Zustand auth store (read lazily so we
 *     never bundle the store on the server unnecessarily).
 *   - Response interceptor that normalises errors into ApiError.
 *   - 401 handling: clear the auth store + emit a synthetic event so any
 *     UI surface can react (e.g. redirect to /login).
 */
import axios, {
  AxiosError,
  AxiosHeaders,
  type AxiosInstance,
  type InternalAxiosRequestConfig,
} from "axios";

import { config } from "@/lib/config";
import { dispatchAppEvent } from "@/lib/events";
import { logger } from "@/lib/logger";
import type { ApiErrorBody, ValidationErrorItem } from "@/lib/api/types";

export class ApiError extends Error {
  public readonly validationErrors: ValidationErrorItem[];

  constructor(
    message: string,
    public readonly status: number,
    public readonly body: ApiErrorBody | string | null,
    public override readonly cause?: unknown,
    validationErrors?: ValidationErrorItem[],
  ) {
    super(message);
    this.name = "ApiError";
    this.validationErrors = validationErrors ?? [];
  }

  static fromAxios(err: AxiosError): ApiError {
    const status = err.response?.status ?? 0;
    const body = (err.response?.data ?? null) as ApiErrorBody | string | null;

    // For 422 responses, extract per-field validation errors
    let validationErrors: ValidationErrorItem[] = [];
    if (
      status === 422 &&
      typeof body === "object" &&
      body !== null &&
      Array.isArray((body as ApiErrorBody).detail)
    ) {
      validationErrors = (body as ApiErrorBody).detail as ValidationErrorItem[];
    }

    // Pick the human-readable top-level message
    let message: string;
    if (validationErrors.length > 0) {
      // Use the first field error as the primary message
      message = validationErrors[0]!.message;
    } else if (typeof body === "object" && body !== null) {
      const b = body as ApiErrorBody;
      const raw = b.error || (typeof b.detail === "string" ? b.detail : null);
      message = typeof raw === "string" ? raw : err.message || "Bilinmeyen ağ hatası.";
    } else {
      message = err.message || "Bilinmeyen ağ hatası.";
    }

    return new ApiError(message, status, body, err, validationErrors);
  }
}

/**
 * Hook for the auth store to register itself with the API client.
 *
 * The store calls `registerAuthAccessor` at module init.  We hold a function
 * reference so the axios layer doesn't need to import the store directly
 * (avoiding a circular dependency).
 */
type AuthAccessor = {
  getToken: () => string | null;
  onUnauthorized: () => void;
};

let authAccessor: AuthAccessor = {
  getToken: () => null,
  onUnauthorized: () => {
    dispatchAppEvent("auth:unauthorized", {});
  },
};

export function registerAuthAccessor(accessor: AuthAccessor): void {
  authAccessor = accessor;
}

export function getAuthToken(): string | null {
  return authAccessor.getToken();
}

function attachAuthInterceptor(instance: AxiosInstance): void {
  instance.interceptors.request.use((cfg: InternalAxiosRequestConfig) => {
    const token = authAccessor.getToken();
    if (token) {
      const headers = AxiosHeaders.from(cfg.headers);
      headers.set("Authorization", `Bearer ${token}`);
      cfg.headers = headers;
    }
    return cfg;
  });
}

function attachErrorInterceptor(instance: AxiosInstance, label: string): void {
  instance.interceptors.response.use(
    (response) => response,
    (error: AxiosError) => {
      const apiError = ApiError.fromAxios(error);
      const url = error.config?.url;
      logger.warn(`${label} request failed`, {
        url,
        status: apiError.status,
        code: error.code,
        message: apiError.message,
      });

      // Don't broadcast errors from background health probes — they would
      // re-trigger the offline banner endlessly during normal outages.
      const silent = error.config?.headers?.get?.("X-Silent-Errors") === "1";

      if (apiError.status === 401) {
        authAccessor.onUnauthorized();
      } else if (!silent) {
        if (error.code === "ECONNABORTED" || error.code === "ETIMEDOUT") {
          dispatchAppEvent("api:timeout", { url, message: apiError.message });
        } else if (apiError.status === 0) {
          dispatchAppEvent("api:offline", { url, message: apiError.message });
        } else if (apiError.status >= 500) {
          dispatchAppEvent("api:error", {
            url,
            status: apiError.status,
            message: apiError.message,
          });
        }
      }

      return Promise.reject(apiError);
    },
  );
}

function buildInstance(baseURL: string, timeout: number, label: string): AxiosInstance {
  const instance = axios.create({
    baseURL,
    timeout,
    headers: {
      Accept: "application/json",
    },
  });
  attachAuthInterceptor(instance);
  attachErrorInterceptor(instance, label);
  return instance;
}

export const api = buildInstance(config.api.v1, config.api.timeoutMs, "api");
export const aiApi = buildInstance(config.api.ai, config.api.aiTimeoutMs, "ai");
