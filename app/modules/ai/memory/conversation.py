"""
Conversation memory.

In-memory, per-session ring buffer of recent turns. Keeps the model's
context window predictable — older turns drop off automatically.

The implementation is intentionally simple and thread-safe enough for
single-process use. The class exposes the same shape a future DB-backed
store will use (``append`` / ``recent`` / ``clear``), so the chat service
does not need to change when persistence is added.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock


@dataclass
class ConversationTurn:
    role: str           # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    """
    Per-session bounded conversation buffer.

    Parameters
    ----------
    max_turns : int
        Maximum number of user+assistant pairs to retain.  Older turns are
        dropped FIFO once the cap is reached.
    """

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        # Each turn is one message. A "turn" pair is two messages, so the
        # buffer length is 2*max_turns.
        self._buffer: dict[str, deque[ConversationTurn]] = {}
        self._lock = RLock()

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            buf = self._buffer.get(session_id)
            if buf is None:
                buf = deque(maxlen=self.max_turns * 2)
                self._buffer[session_id] = buf
            buf.append(ConversationTurn(role=role, content=content))

    def recent(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            buf = self._buffer.get(session_id)
            return list(buf) if buf else []

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._buffer.pop(session_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._buffer.clear()


# ── Future DB-backed store hook ──────────────────────────────────────────────
#
# When persistence is needed, implement a ConversationStore protocol with the
# same three methods (append, recent, clear) backed by SQLAlchemy and swap
# it in via dependency injection. AIChatService only depends on the duck
# interface, not the in-memory class directly.