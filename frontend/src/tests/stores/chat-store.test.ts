import { beforeEach, describe, expect, it } from "vitest";

import { useChatStore } from "@/stores/chat-store";

const sampleAssistant = {
  id: "a-1",
  role: "assistant" as const,
  content: "Merhaba, ben HantaProject asistanı.",
  refused: false,
  refusalReason: null,
  timestamp: "2026-05-14T00:00:00Z",
};

describe("useChatStore", () => {
  beforeEach(() => {
    useChatStore.setState({ messages: [], sessionId: "session-test" });
  });

  it("appends messages and preserves order", () => {
    useChatStore.getState().appendMessage({
      id: "u-1",
      role: "user",
      content: "Hantavirüs nedir?",
      timestamp: "2026-05-14T00:00:01Z",
    });
    useChatStore.getState().appendMessage(sampleAssistant);

    const state = useChatStore.getState();
    expect(state.messages).toHaveLength(2);
    expect(state.messages[0]?.role).toBe("user");
    expect(state.messages[1]?.role).toBe("assistant");
  });

  it("caps the buffer at 50 entries", () => {
    for (let i = 0; i < 60; i += 1) {
      useChatStore.getState().appendMessage({
        id: `m-${i}`,
        role: "user",
        content: `mesaj ${i}`,
      });
    }
    const state = useChatStore.getState();
    expect(state.messages).toHaveLength(50);
    // Oldest entries (indices 0..9) dropped.
    expect(state.messages[0]?.id).toBe("m-10");
    expect(state.messages[49]?.id).toBe("m-59");
  });

  it("reset clears messages and rotates the session id", () => {
    useChatStore.getState().appendMessage(sampleAssistant);
    const prevSession = useChatStore.getState().sessionId;

    useChatStore.getState().reset();

    const state = useChatStore.getState();
    expect(state.messages).toHaveLength(0);
    expect(state.sessionId).not.toBe(prevSession);
  });
});
