from __future__ import annotations

import time
from collections import deque

from sources.base import SourcePacket
from sources.text_bus import publish_text


class ChatSource:
    """Queue-backed text source for desktop chat/UI integration.

    Text is normalized into SourcePacket and published onto the shared text
    input bus. The current desktop runtime lazily attaches a terminal transport,
    but the agent-facing contract stays the same for future GUI/web/socket
    transports.
    """

    def __init__(
        self,
        *,
        source_name: str = "chat",
        enable_terminal: bool = True,
    ):
        self.source_name = source_name
        self.enable_terminal = bool(enable_terminal)
        self._queue: deque[SourcePacket] = deque()
        self._terminal = None

    def submit(
        self,
        text: str,
        *,
        entity_id: str | None = "desktop_user",
        timestamp: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        text = str(text).strip()
        if not text:
            return

        packet = SourcePacket(
            source=self.source_name,
            modality="text",
            timestamp=time.monotonic() if timestamp is None else float(timestamp),
            entity_id=entity_id,
            payload={"text": text},
            confidence=1.0,
            metadata=dict(metadata or {}),
        )

        self._queue.append(packet)
        publish_text(packet)

    def update(self) -> list[SourcePacket]:
        self._ensure_terminal()

        packets = list(self._queue)
        self._queue.clear()
        return packets

    def close(self) -> None:
        if self._terminal is not None:
            self._terminal.close()

    def _ensure_terminal(self) -> None:
        if not self.enable_terminal or self._terminal is not None:
            return

        # Lazy import avoids coupling the generic chat source to a terminal at
        # module import time. Future transports can feed submit() directly.
        from sources.terminal_chat import TerminalChatInput

        self._terminal = TerminalChatInput(self)
        self._terminal.start()
