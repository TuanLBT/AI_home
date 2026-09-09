from __future__ import annotations

import json
import socket
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sources.chat import ChatSource


class ChatIPCServer:
    """Small localhost-only JSON-lines transport for ChatSource.

    The agent-facing side remains ChatSource/SourcePacket. This server is only a
    desktop transport, so it can later be replaced by a GUI or web frontend
    without changing the agent core.
    """

    def __init__(
        self,
        chat_source: "ChatSource",
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        self.chat_source = chat_source
        self.host = host
        self.port = int(port)
        self._closed = threading.Event()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, socket.socket] = {}
        self._clients_lock = threading.Lock()
        self._send_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._serve,
            name="chat-ipc-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._closed.set()

        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass

        with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()

        for client in clients:
            try:
                client.close()
            except OSError:
                pass

    def publish_reply(
        self,
        entity_id: str,
        text: str,
        *,
        error: bool = False,
    ) -> bool:
        with self._clients_lock:
            client = self._clients.get(entity_id)

        if client is None:
            return False

        payload = {
            "type": "error" if error else "reply",
            "entity_id": entity_id,
            "text": str(text),
        }
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

        try:
            with self._send_lock:
                client.sendall(encoded)
            return True
        except OSError:
            self._drop_client(entity_id, client)
            return False

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(4)
        server.settimeout(0.5)
        self._server = server

        while not self._closed.is_set():
            try:
                client, _addr = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handle_client,
                args=(client,),
                name="chat-ipc-client",
                daemon=True,
            ).start()

    def _handle_client(self, client: socket.socket) -> None:
        entity_id: str | None = None
        try:
            with client.makefile("r", encoding="utf-8", newline="\n") as reader:
                for line in reader:
                    if self._closed.is_set():
                        return

                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = str(request.get("text") or "").strip()
                    if not text:
                        continue

                    entity_id = str(
                        request.get("entity_id") or "desktop_user"
                    ).strip() or "desktop_user"

                    with self._clients_lock:
                        old = self._clients.get(entity_id)
                        self._clients[entity_id] = client

                    if old is not None and old is not client:
                        try:
                            old.close()
                        except OSError:
                            pass

                    self.chat_source.submit(
                        text,
                        entity_id=entity_id,
                        metadata={"transport": "localhost_ipc"},
                    )
        except OSError:
            pass
        finally:
            if entity_id is not None:
                self._drop_client(entity_id, client)
            try:
                client.close()
            except OSError:
                pass

    def _drop_client(self, entity_id: str, client: socket.socket) -> None:
        with self._clients_lock:
            if self._clients.get(entity_id) is client:
                self._clients.pop(entity_id, None)


_ACTIVE_SERVER: ChatIPCServer | None = None
_ACTIVE_LOCK = threading.Lock()


def set_active_chat_server(server: ChatIPCServer | None) -> None:
    global _ACTIVE_SERVER
    with _ACTIVE_LOCK:
        _ACTIVE_SERVER = server


def publish_chat_reply(
    entity_id: str,
    text: str,
    *,
    error: bool = False,
) -> bool:
    with _ACTIVE_LOCK:
        server = _ACTIVE_SERVER

    if server is None:
        return False

    return server.publish_reply(
        entity_id,
        text,
        error=error,
    )
