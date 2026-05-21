"use client";

import { useCallback, useState } from "react";

import { aiAssistant } from "@/lib/api/ai";
import { logger } from "@/lib/logger";
import { useChatStore } from "@/stores/chat-store";
import type { ChatBubbleMessage } from "@/components/chat/message-bubble";

function uid(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Chat orchestration hook.
 *
 * Reads/writes the persisted chat store so refresh keeps history alive.
 * Designed to be drop-in compatible with a future streaming backend:
 * `sendMessage` first appends a user bubble, then a placeholder assistant
 * bubble, then patches the placeholder once the response arrives.  Swap
 * to SSE later by streaming patches into the same placeholder.
 */
export function useAiChat() {
  const messages = useChatStore((s) => s.messages);
  const sessionId = useChatStore((s) => s.sessionId);
  const hasHydrated = useChatStore((s) => s.hasHydrated);
  const appendMessage = useChatStore((s) => s.appendMessage);
  const reset = useChatStore((s) => s.reset);

  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      setError(null);

      const userMsg: ChatBubbleMessage = {
        id: uid(),
        role: "user",
        content: trimmed,
        timestamp: new Date().toISOString(),
      };
      appendMessage(userMsg);

      setIsSending(true);
      try {
        const reply = await aiAssistant.chat({
          message: trimmed,
          session_id: sessionId,
        });
        appendMessage({
          id: uid(),
          role: "assistant",
          content: reply.content,
          refused: reply.refused,
          refusalReason: reply.refusal_reason,
          timestamp: reply.timestamp,
        });
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "AI servisine ulaşılamadı.";
        logger.warn("ai chat failed", err);
        setError(message);
      } finally {
        setIsSending(false);
      }
    },
    [appendMessage, sessionId],
  );

  return {
    messages,
    isSending,
    error,
    hasHydrated,
    sendMessage,
    reset,
    sessionId,
  };
}
