from __future__ import annotations

import time
from typing import Any

from sources.base import SourcePacket


class ScreenSource:
    """Hardware/desktop-neutral screen observation source skeleton.

    A later backend can use Wayland portals, PipeWire, X11, OS APIs, or a
    remote desktop feed. The agent core should only receive SourcePacket data.
    """

    def __init__(self, *, source_name: str = "screen"):
        self.source_name = source_name

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
        # No capture backend is selected yet. This intentionally stays empty
        # until the desktop capture method is chosen for the host OS.
        return []
