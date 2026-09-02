from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MemoryItem:
    timestamp: float
    category: str
    type: str
    data: dict[str, Any]


class InteractionMemory:
    """
    Short-term interaction memory.

    Stores recent perception/events/actions/speech per person without depending
    on YOLO, ByteTrack, or TTS internals.

    This is intentionally short-term memory only. Old items expire after
    retention_s.
    """

    def __init__(
        self,
        retention_s: float = 60.0,
        max_items_per_person: int = 100,
    ):
        self.retention_s = retention_s
        self.max_items_per_person = max_items_per_person
        self._items: dict[str, deque[MemoryItem]] = {}

    def record(
        self,
        entity_id: str,
        category: str,
        type: str,
        now: float,
        data: dict[str, Any] | None = None,
    ) -> None:
        queue = self._items.setdefault(
            entity_id,
            deque(maxlen=self.max_items_per_person),
        )

        queue.append(
            MemoryItem(
                timestamp=now,
                category=category,
                type=type,
                data=dict(data or {}),
            )
        )

        self._prune(entity_id, now)

    def record_low_level_events(
        self,
        events: list[dict],
        now: float,
    ) -> None:
        for event in events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            self.record(
                entity_id=entity_id,
                category="world_event",
                type=event["type"],
                now=now,
                data={
                    key: value
                    for key, value in event.items()
                    if key not in ("entity_id", "type", "timestamp")
                },
            )

    def record_high_level_events(
        self,
        events: list[dict],
        now: float,
    ) -> None:
        for event in events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            self.record(
                entity_id=entity_id,
                category="semantic_event",
                type=event["type"],
                now=now,
                data={
                    key: value
                    for key, value in event.items()
                    if key not in ("entity_id", "type", "timestamp")
                },
            )

    def record_behavior_events(
        self,
        events: list[dict],
        now: float,
    ) -> None:
        for event in events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            self.record(
                entity_id=entity_id,
                category="behavior",
                type=event["type"],
                now=now,
                data={
                    key: value
                    for key, value in event.items()
                    if key not in ("entity_id", "type", "timestamp")
                },
            )

    def record_executed_actions(
        self,
        events: list[dict],
        now: float,
    ) -> None:
        for event in events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            data = {
                "action": event.get("action"),
                "command": event.get("command"),
                "reason": event.get("reason"),
                "gesture": event.get("gesture"),
                "speech_text": event.get("speech_text"),
                "speech_ok": event.get("speech_ok"),
            }

            self.record(
                entity_id=entity_id,
                category="action",
                type="ACTION_EXECUTED",
                now=now,
                data=data,
            )

    def recent(
        self,
        entity_id: str,
        now: float,
        within_s: float = 30.0,
    ) -> list[MemoryItem]:
        self._prune(entity_id, now)

        return [
            item
            for item in self._items.get(entity_id, ())
            if now - item.timestamp <= within_s
        ]

    def has_recent(
        self,
        entity_id: str,
        type: str,
        now: float,
        within_s: float,
        category: str | None = None,
    ) -> bool:
        for item in reversed(
            self.recent(
                entity_id,
                now,
                within_s=within_s,
            )
        ):
            if item.type != type:
                continue

            if category is not None and item.category != category:
                continue

            return True

        return False

    def last(
        self,
        entity_id: str,
        now: float,
        category: str | None = None,
        type: str | None = None,
    ) -> MemoryItem | None:
        self._prune(entity_id, now)

        for item in reversed(self._items.get(entity_id, ())):
            if category is not None and item.category != category:
                continue

            if type is not None and item.type != type:
                continue

            return item

        return None

    def snapshot(
        self,
        entity_id: str,
        now: float,
        within_s: float = 30.0,
    ) -> dict[str, Any]:
        items = self.recent(
            entity_id,
            now,
            within_s=within_s,
        )

        last_action = self.last(
            entity_id,
            now,
            category="action",
        )

        last_semantic_event = self.last(
            entity_id,
            now,
            category="semantic_event",
        )

        recent_actions = [
            item
            for item in items
            if item.category == "action"
        ]

        return {
            "recent_event_types": [
                item.type
                for item in items
                if item.category in (
                    "world_event",
                    "semantic_event",
                )
            ],
            "last_action": (
                last_action.data.get("action")
                if last_action is not None
                else None
            ),
            "last_speech": (
                last_action.data.get("speech_text")
                if last_action is not None
                else None
            ),
            "last_gesture": (
                last_action.data.get("gesture")
                if last_action is not None
                else None
            ),
            "last_semantic_event": (
                last_semantic_event.type
                if last_semantic_event is not None
                else None
            ),
            "recent_actions": [
                {
                    "action": item.data.get("action"),
                    "gesture": item.data.get("gesture"),
                    "speech_text": item.data.get("speech_text"),
                    "age_s": max(0.0, now - item.timestamp),
                }
                for item in recent_actions[-10:]
            ],
        }

    def _prune(
        self,
        entity_id: str,
        now: float,
    ) -> None:
        queue = self._items.get(entity_id)

        if queue is None:
            return

        cutoff = now - self.retention_s

        while queue and queue[0].timestamp < cutoff:
            queue.popleft()
