"use client";

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

import type { ChatBubbleMessage } from "@/components/chat/message-bubble";

export interface ChatState {
  messages: ChatBubbleMessage[];
  sessionId: string;
  hasHydrated: boolean;

  appendMessage: (message: ChatBubbleMessage) => void;
  reset: () => void;
  setHasHydrated: (value: boolean) => void;
}

function newSessionId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Persisted chat memory.
 *
 * Why persist? So a reload (or an accidental navigation) keeps the
 * conversation the user was having.  We persist `messages` AND `sessionId`
 * so the backend's per-session memory continues to track this user.
 *
 * The buffer is capped at 50 turns to keep localStorage payloads sensible.
 */
const MAX_BUFFER = 50;

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      messages: [],
      sessionId: newSessionId(),
      hasHydrated: false,

      appendMessage(message) {
        set((state) => {
          const next = [...state.messages, message];
          if (next.length > MAX_BUFFER) next.splice(0, next.length - MAX_BUFFER);
          return { messages: next };
        });
      },

      reset() {
        set({ messages: [], sessionId: newSessionId() });
      },

      setHasHydrated(value) {
        set({ hasHydrated: value });
      },
    }),
    {
      name: "hanta-chat",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        messages: state.messages,
        sessionId: state.sessionId,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true);
      },
    },
  ),
);
