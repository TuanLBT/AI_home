from __future__ import annotations

import time
from collections import deque

from sources.base import SourcePacket
from sources.text_bus import publish_text


class ChatSource:
    """Queue-backed text source for desktop chat/UI integration.

    Text is normalized into SourcePacket and published onto the shared text
    input bus. This source does not decide which sensors/evidence are needed;
    semantic observation routing happens later in the agent pipeline.
    """

    def __init__(
        self,
        *,
        source_name: str = "chat",
        enable_terminal: bool = False,
        enable_ipc: bool = True,
        ipc_host: str = "127.0.0.1",
        ipc_port: int = 8765,
    ):
        self.source_name = source_name
        self.enable_terminal = bool(enable_terminal)
        self.enable_ipc = bool(enable_ipc)
        self.ipc_host = ipc_host
        self.ipc_port = int(ipc_port)
        self._queue: deque[SourcePacket] = deque()
        self._terminal = None
        self._ipc = None

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
        self._ensure_transports()

        packets = list(self._queue)
        self._queue.clear()
        return packets

    def close(self) -> None:
        if self._terminal is not None:
            self._terminal.close()

        if self._ipc is not None:
            from sources.chat_ipc import set_active_chat_server

            self._ipc.close()
            set_active_chat_server(None)

    def _ensure_transports(self) -> None:
        if self.enable_ipc and self._ipc is None:
            from sources.chat_ipc import ChatIPCServer, set_active_chat_server

            self._ipc = ChatIPCServer(
                self,
                host=self.ipc_host,
                port=self.ipc_port,
            )
            set_active_chat_server(self._ipc)
            self._ipc.start()

        if self.enable_terminal and self._terminal is None:
            from sources.terminal_chat import TerminalChatInput

            self._terminal = TerminalChatInput(self)
            self._terminal.start()
