from __future__ import annotations

import queue
import threading

from sources.chat import ChatSource


class TerminalChatInput:
    """Non-blocking terminal adapter that feeds a ChatSource.

    stdin reading lives on a daemon thread, so camera/audio/perception loops are
    never blocked by input(). The agent core still only sees ChatSource packets,
    which means this adapter can later be replaced by a GUI, web UI, socket, or
    another client without changing the text-input contract.
    """

    def __init__(
        self,
        chat_source: ChatSource,
        *,
        prompt: str = "you> ",
        entity_id: str = "desktop_user",
    ):
        self.chat_source = chat_source
        self.prompt = prompt
        self.entity_id = entity_id
        self._closed = threading.Event()
        self._status: queue.Queue[dict] = queue.Queue(maxsize=8)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="terminal-chat-input",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed.set()

    def update(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self._status.get_nowait())
            except queue.Empty:
                break
        return events

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                text = input(self.prompt)
            except EOFError:
                self._push_status({"type": "TERMINAL_CHAT_EOF"})
                return
            except Exception as exc:
                self._push_status({
                    "type": "TERMINAL_CHAT_ERROR",
                    "error": str(exc),
                })
                return

            if self._closed.is_set():
                return

            text = text.strip()
            if not text:
                continue

            self.chat_source.submit(
                text,
                entity_id=self.entity_id,
                metadata={"transport": "terminal"},
            )

    def _push_status(self, event: dict) -> None:
        try:
            self._status.put_nowait(event)
        except queue.Full:
            pass
