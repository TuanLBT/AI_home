from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SourcePacket:
    """Hardware-neutral payload emitted by an input source.

    Sources may attach device-specific details inside metadata, but consumers
    should primarily depend on source/modality/payload.
    """

    source: str
    modality: str
    timestamp: float
    payload: Any
    confidence: float | None = None
    entity_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
