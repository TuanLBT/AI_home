from __future__ import annotations

import time
from collections import deque

from sources.base import SourcePacket


class ChatSource:
    """Queue-backed text source for desktop chat/UI integration.

    The current runtime does not own a chat UI yet. A UI, socket, terminal, or
    external client can push text here without the agent core depending on how
    that text arrived.
    """

    def __init__(self, *, source_name: str = "chat"):
        self.source_name = source_name
        self._queue: deque[SourcePacket] = deque()

    def submit(
        self,
        text: str,
        *,
        entity_id: str | None = "user",
        timestamp: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        text = str(text).strip()
        if not text:
            return

        self._queue.append(SourcePacket(
            source=self.source_name,
            modality="text",
            timestamp=time.monotonic() if timestamp is None else float(timestamp),
            entity_id=entity_id,
            payload={"text": text},
            confidence=1.0,
            metadata=dict(metadata or {}),
        ))

    def update(self) -> list[SourcePacket]:
        packets = list(self._queue)
        self._queue.clear()
        return packets
