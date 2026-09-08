from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import time
import uuid


EPISODE_SCHEMA_VERSION = 1
VALID_SOURCES = {"camera", "image", "video", "audio", "text", "mixed", "unknown"}
VALID_LABEL_ORIGINS = {"human", "self", "external_model", "rule", "import", "unknown"}


@dataclass(slots=True)
class LearningEpisode:
    """Source-agnostic unit of experience used for learning and retrieval.

    Keep observations as close to the raw perception output as practical.
    Derived features belong in ``representations`` so future learners can be
    changed without losing the original evidence.
    """

    source: str
    observations: dict[str, Any]
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    entity_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    representations: dict[str, Any] = field(default_factory=dict)
    proposed_meaning: str | None = None
    confidence: float | None = None
    label_origin: str = "unknown"
    verified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.source = (self.source or "unknown").strip().lower()
        if self.source not in VALID_SOURCES:
            self.metadata.setdefault("original_source", self.source)
            self.source = "unknown"

        self.label_origin = (self.label_origin or "unknown").strip().lower()
        if self.label_origin not in VALID_LABEL_ORIGINS:
            self.metadata.setdefault("original_label_origin", self.label_origin)
            self.label_origin = "unknown"

        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))

        if self.ended_at is None:
            self.ended_at = self.started_at

        if self.ended_at < self.started_at:
            raise ValueError("ended_at must be >= started_at")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "LearningEpisode":
        return cls(
            episode_id=str(record.get("episode_id") or uuid.uuid4()),
            schema_version=int(record.get("schema_version") or EPISODE_SCHEMA_VERSION),
            source=str(record.get("source") or "unknown"),
            started_at=float(record.get("started_at") or time.time()),
            ended_at=(
                float(record["ended_at"])
                if record.get("ended_at") is not None
                else None
            ),
            entity_id=record.get("entity_id"),
            observations=dict(record.get("observations") or {}),
            context=dict(record.get("context") or {}),
            representations=dict(record.get("representations") or {}),
            proposed_meaning=record.get("proposed_meaning"),
            confidence=(
                float(record["confidence"])
                if record.get("confidence") is not None
                else None
            ),
            label_origin=str(record.get("label_origin") or "unknown"),
            verified=bool(record.get("verified", False)),
            metadata=dict(record.get("metadata") or {}),
        )


def legacy_pose_record_to_episode(record: dict[str, Any]) -> LearningEpisode | None:
    """Convert the existing teaching_examples.jsonl pose format on read.

    No migration is required: old examples remain usable while new data can be
    written in the generic episode format.
    """

    if record.get("modality") != "pose":
        return None

    label = str(record.get("label") or "").strip() or None
    samples = record.get("samples") or []
    if not samples:
        return None

    started_at = record.get("started_at") or record.get("timestamp") or time.time()
    ended_at = record.get("ended_at") or started_at

    return LearningEpisode(
        source=str(record.get("source") or "camera"),
        started_at=float(started_at),
        ended_at=float(ended_at),
        entity_id=record.get("entity_id"),
        observations={"pose": {"samples": samples}},
        context=dict(record.get("context") or {}),
        proposed_meaning=label,
        confidence=1.0 if label else None,
        label_origin=str(record.get("label_origin") or "human"),
        verified=bool(record.get("verified", True)),
        metadata={
            "legacy_format": "pose_teaching_example",
            **dict(record.get("metadata") or {}),
        },
    )
