from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
import uuid


class ExperienceLabelStore:
    """Append-only human feedback linked to immutable experiences."""

    SCHEMA_VERSION = 1
    VALID_LABELS = {"positive", "negative", "neutral"}

    def __init__(
        self,
        path: str | Path = "data/experience_labels.jsonl",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        experience_id: str,
        label: str,
        *,
        note: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        normalized = label.strip().lower()

        if normalized not in self.VALID_LABELS:
            raise ValueError(
                f"label must be one of {sorted(self.VALID_LABELS)}"
            )

        if not experience_id.strip():
            raise ValueError("experience_id must not be empty")

        labeled_at = time.time()
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "label_id": str(uuid.uuid4()),
            "experience_id": experience_id,
            "label": normalized,
            "note": note.strip() if note and note.strip() else None,
            "source": source,
            "labeled_at": labeled_at,
            "labeled_at_utc": datetime.fromtimestamp(
                labeled_at,
                tz=timezone.utc,
            ).isoformat(),
        }

        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()

        return record

    def latest_by_experience(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}

        if not self.path.is_file():
            return latest

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.path}:{line_number}"
                    ) from exc

                experience_id = record.get("experience_id")

                if experience_id:
                    latest[experience_id] = record

        return latest
