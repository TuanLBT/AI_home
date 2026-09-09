from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from sources.base import SourcePacket
from sources.screen_bus import consume_screen_capture_request, publish_screen

try:
    from PIL import ImageGrab
except ImportError:
    ImageGrab = None


@contextmanager
def _quiet_stderr(enabled: bool = True) -> Iterator[None]:
    """Temporarily silence OS-level stderr inherited by capture helpers.

    On KDE Wayland, Pillow may spawn Spectacle. Spectacle writes informational
    Tesseract/icon-theme messages directly to stderr, so Qt logging rules alone
    do not reliably silence them. Redirecting file descriptor 2 during grab()
    also silences the helper process while preserving Python exceptions.
    """
    if not enabled:
        yield
        return

    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        os.dup2(null_fd, stderr_fd)
        yield
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        finally:
            os.close(saved_fd)
            os.close(null_fd)


class ScreenSource:
    """Desktop screen source with cached, on-demand capture.

    Continuous polling is intentionally avoided on KDE Wayland because Pillow
    may spawn Spectacle for every grab. The source captures only when requested
    (for example, by a chat turn) or when capture_now() is called explicitly.
    """

    def __init__(
        self,
        *,
        source_name: str = "screen",
        change_threshold: float = 0.015,
        analysis_size: tuple[int, int] = (160, 90),
        all_screens: bool = False,
        suppress_backend_stderr: bool = True,
    ):
        if ImageGrab is None:
            raise RuntimeError(
                "Pillow ImageGrab is unavailable. Install Pillow or replace "
                "ScreenSource with another desktop capture adapter."
            )

        self.source_name = source_name
        self.change_threshold = max(0.0, float(change_threshold))
        self.analysis_size = analysis_size
        self.all_screens = bool(all_screens)
        self.suppress_backend_stderr = bool(suppress_backend_stderr)

        self._previous_analysis: np.ndarray | None = None
        self._latest_packet: SourcePacket | None = None
        self._latest_change_score: float | None = None
        self._capture_error: str | None = None

        existing = os.environ.get("QT_LOGGING_RULES", "").strip()
        quiet_rules = "kf.iconthemes=false;spectacle.debug=false"
        if quiet_rules not in existing:
            os.environ["QT_LOGGING_RULES"] = (
                f"{existing};{quiet_rules}" if existing else quiet_rules
            )

    def packet_from_frame(
        self,
        frame: Any,
        *,
        timestamp: float | None = None,
        active_window: str | None = None,
        metadata: dict | None = None,
    ) -> SourcePacket:
        extra = dict(metadata or {})
        if active_window is not None:
            extra["active_window"] = active_window

        return SourcePacket(
            source=self.source_name,
            modality="screen_frame",
            timestamp=time.monotonic() if timestamp is None else float(timestamp),
            payload={"frame": frame},
            metadata=extra,
        )

    def update(self) -> list[SourcePacket]:
        if not consume_screen_capture_request():
            return []

        packet = self.capture_now()
        if packet is None:
            return []

        # A requested fresh frame is useful to perception even when its cheap
        # change score is below threshold. `changed` remains evidence metadata.
        return [packet]

    def capture_now(self) -> SourcePacket | None:
        now = time.monotonic()

        try:
            with _quiet_stderr(self.suppress_backend_stderr):
                image = ImageGrab.grab(all_screens=self.all_screens)
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            self._capture_error = None
        except Exception as exc:
            self._capture_error = str(exc)
            return None

        analysis = self._analysis_frame(frame)
        change_score = self._change_score(analysis)
        changed = (
            self._previous_analysis is None
            or change_score >= self.change_threshold
        )

        self._previous_analysis = analysis
        self._latest_change_score = change_score

        packet = self.packet_from_frame(
            frame,
            timestamp=now,
            metadata={
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "channels": int(frame.shape[2]),
                "change_score": float(change_score),
                "changed": bool(changed),
                "capture_backend": "pillow_imagegrab_on_demand",
            },
        )
        self._latest_packet = packet
        publish_screen(packet)
        return packet

    def latest(self) -> SourcePacket | None:
        return self._latest_packet

    def latest_context(self) -> dict:
        packet = self._latest_packet
        if packet is None:
            return {
                "available": False,
                "capture_error": self._capture_error,
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
        }

    def metadata(self) -> dict:
        return {
            "source_type": "desktop_screen",
            "capabilities": ["rgb", "change_detection", "on_demand_capture"],
            "change_threshold": self.change_threshold,
            "backend": "pillow_imagegrab_on_demand",
            "suppress_backend_stderr": self.suppress_backend_stderr,
        }

    def _analysis_frame(self, frame: np.ndarray) -> np.ndarray:
        target_w, target_h = self.analysis_size
        h, w = frame.shape[:2]

        xs = np.linspace(0, max(0, w - 1), target_w).astype(np.int32)
        ys = np.linspace(0, max(0, h - 1), target_h).astype(np.int32)
        sampled = frame[np.ix_(ys, xs)]

        gray = (
            sampled[..., 0].astype(np.float32) * 0.299
            + sampled[..., 1].astype(np.float32) * 0.587
            + sampled[..., 2].astype(np.float32) * 0.114
        ) / 255.0
        return gray

    def _change_score(self, current: np.ndarray) -> float:
        previous = self._previous_analysis
        if previous is None or previous.shape != current.shape:
            return 1.0

        return float(np.mean(np.abs(current - previous)))
