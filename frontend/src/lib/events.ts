/**
 * Lightweight typed event bus for cross-cutting UI signals.
 *
 * Why an event bus instead of context?  Because the producer (axios
 * interceptor) lives outside the React tree and the consumer (toast/banner)
 * lives at the root.  A bare `window.dispatchEvent` works fine here and
 * stays under 30 lines.  Each event has a string discriminator + payload.
 */

export type AppEventMap = {
  "auth:unauthorized": { reason?: string };
  "api:offline": { url?: string; message: string };
  "api:timeout": { url?: string; message: string };
  "api:error": { url?: string; status: number; message: string };
};

export type AppEventName = keyof AppEventMap;

export function dispatchAppEvent<K extends AppEventName>(
  name: K,
  detail: AppEventMap[K],
): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(name, { detail }));
}

export function onAppEvent<K extends AppEventName>(
  name: K,
  handler: (detail: AppEventMap[K]) => void,
): () => void {
  if (typeof window === "undefined") return () => undefined;
  const listener = (event: Event) => {
    const custom = event as CustomEvent<AppEventMap[K]>;
    handler(custom.detail);
  };
  window.addEventListener(name, listener);
  return () => window.removeEventListener(name, listener);
}
