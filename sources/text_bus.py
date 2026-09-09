from __future__ import annotations

import queue

from sources.base import SourcePacket


_TEXT_INPUTS: queue.Queue[SourcePacket] = queue.Queue(maxsize=64)


def publish_text(packet: SourcePacket) -> None:
    """Publish one normalized text observation for the agent text pipeline."""
    try:
        _TEXT_INPUTS.put_nowait(packet)
    except queue.Full:
        try:
            _TEXT_INPUTS.get_nowait()
        except queue.Empty:
            pass
        try:
            _TEXT_INPUTS.put_nowait(packet)
        except queue.Full:
            pass


def drain_text_inputs() -> list[SourcePacket]:
    packets: list[SourcePacket] = []
    while True:
        try:
            packets.append(_TEXT_INPUTS.get_nowait())
        except queue.Empty:
            break
    return packets
