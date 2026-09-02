from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(slots=True)
class Observation:
    source: str
    type: str
    entity_id: str
    confidence: float
    timestamp: float = field(default_factory=time.monotonic)
    data: dict[str, Any] = field(default_factory=dict)
