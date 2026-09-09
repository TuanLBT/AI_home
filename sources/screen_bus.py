from __future__ import annotations

import threading

from sources.base import SourcePacket


_lock = threading.Lock()
_latest: SourcePacket | None = None
_latest_representation: dict | None = None
_capture_requested = False


def publish_screen(packet: SourcePacket) -> None:
    """Publish the newest normalized raw screen observation."""
    global _latest
    with _lock:
        _latest = packet


def publish_screen_representation(representation: dict) -> None:
    """Publish derived screen facts separately from raw screenshot evidence."""
    global _latest_representation
    with _lock:
        _latest_representation = dict(representation or {})


def request_screen_capture() -> None:
    """Ask the active ScreenSource to capture one fresh frame.

    Clear the previous derived representation immediately so a new screen
    question can never accidentally reuse facts from an older screenshot if
    fresh capture/perception fails.
    """
    global _capture_requested, _latest_representation
    with _lock:
        _capture_requested = True
        _latest_representation = None


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


def latest_screen_representation() -> dict | None:
    with _lock:
        if _latest_representation is None:
            return None
        return dict(_latest_representation)


def latest_screen_context() -> dict:
    with _lock:
        packet = _latest
        representation = (
            dict(_latest_representation)
            if _latest_representation is not None
            else None
        )

    if packet is None:
        return {
            "available": False,
            "representation": representation,
        }

    metadata = dict(packet.metadata or {})
    return {
        "available": True,
        "timestamp": packet.timestamp,
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "changed": metadata.get("changed"),
        "change_score": metadata.get("change_score"),
        "capture_backend": metadata.get("capture_backend"),
        "representation": representation,
    }
