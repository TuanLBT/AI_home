from __future__ import annotations

import threading

from sources.base import SourcePacket


_lock = threading.Lock()
_latest: SourcePacket | None = None


def publish_screen(packet: SourcePacket) -> None:
    """Publish the newest normalized screen observation."""
    global _latest
    with _lock:
        _latest = packet


def latest_screen() -> SourcePacket | None:
    with _lock:
        return _latest


def latest_screen_context() -> dict:
    packet = latest_screen()
    if packet is None:
        return {"available": False}

    metadata = dict(packet.metadata or {})
    return {
        "available": True,
        "timestamp": packet.timestamp,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "changed": metadata.get("changed"),
        "change_score": metadata.get("change_score"),
        "capture_backend": metadata.get("capture_backend"),
    }
