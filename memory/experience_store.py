from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
import uuid


class ExperienceStore:
    """Append-only persistent records for future learning and evaluation."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path = "data/experiences.jsonl",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_brain_decision(self, event: dict[str, Any]) -> None:
        decision = {
            "brain_ok": event.get("brain_ok"),
            "proposal": event.get("proposal"),
            "normalized_action": event.get("normalized_action"),
            "approved": event.get("approved"),
            "final_action": event.get("final_action"),
            "executor_action": event.get("executor_action"),
            "reason": event.get("reason"),
            "violations": event.get("violations", []),
            "error": event.get("error"),
        }

        self._append({
            "kind": "brain_decision",
            "entity_id": event.get("entity_id"),
            "observed_at": event.get("timestamp"),
            "observation": event.get("world_state"),
            "decision": decision,
            "result": None,
        })

    def record_action_result(self, event: dict[str, Any]) -> None:
        decision = {
            "source": event.get("source"),
            "intent": event.get("intent"),
            "confidence": event.get("confidence"),
            "requested_action": event.get("action"),
            "command": event.get("command"),
            "reason": event.get("reason"),
            "target_id": event.get("target_id"),
            "parameters": event.get("parameters", {}),
        }

        result = {
            "request_id": event.get("request_id"),
            "status": event.get("status"),
            "actual": event.get("actual", {}),
            "observations": event.get("observations", {}),
            "goal_id": event.get("goal_id"),
            "goal_reached": event.get("goal_reached"),
            "speech_text": event.get("speech_text"),
            "speech_ok": event.get("speech_ok"),
            "speech_backend": event.get("speech_backend"),
            "speech_error": event.get("speech_error"),
            "finished_at": event.get("finished_at"),
        }

        self._append({
            "kind": "action_result",
            "entity_id": event.get("entity_id"),
            "observed_at": event.get("timestamp"),
            "observation": event.get("context", {}),
            "decision": decision,
            "result": result,
        })

    def record_action_results(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        for event in events:
            self.record_action_result(event)

    def _append(self, payload: dict[str, Any]) -> None:
        recorded_at = time.time()
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "experience_id": str(uuid.uuid4()),
            "recorded_at": recorded_at,
            "recorded_at_utc": datetime.fromtimestamp(
                recorded_at,
                tz=timezone.utc,
            ).isoformat(),
            **payload,
        }

        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            default=self._json_default,
        )

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)

        item = getattr(value, "item", None)
        if callable(item):
            return item()

        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return enum_value

        return repr(value)
