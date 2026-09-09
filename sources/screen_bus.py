from __future__ import annotations

import threading

from sources.base import SourcePacket


_lock = threading.Lock()
_latest: SourcePacket | None = None
_capture_requested = False


def publish_screen(packet: SourcePacket) -> None:
    """Publish the newest normalized screen observation."""
    global _latest
    with _lock:
        _latest = packet


def request_screen_capture() -> None:
    """Ask the active ScreenSource to capture one fresh frame."""
    global _capture_requested
    with _lock:
        _capture_requested = True


def consume_screen_capture_request() -> bool:
    """Consume one pending capture request."""
    global _capture_requested
    with _lock:
        requested = _capture_requested
        _capture_requested = False
        return requested


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
