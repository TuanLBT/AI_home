from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any
import uuid


class TeachingExampleStore:
    """Append-only, modality-agnostic teaching examples."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path = "data/teaching_examples.jsonl",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        label: str,
        modality: str,
        samples: list[dict[str, Any]],
        duration_s: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        label = label.strip()
        modality = modality.strip().lower()

        if not label:
            raise ValueError("label must not be empty")

        if not modality:
            raise ValueError("modality must not be empty")

        if not samples:
            raise ValueError("samples must not be empty")

        captured_at = time.time()
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "example_id": str(uuid.uuid4()),
            "captured_at": captured_at,
            "captured_at_utc": datetime.fromtimestamp(
                captured_at,
                tz=timezone.utc,
            ).isoformat(),
            "label": label,
            "modality": modality,
            "duration_s": float(duration_s),
            "sample_count": len(samples),
            "samples": samples,
            "metadata": dict(metadata or {}),
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

        return record

    def load(
        self,
        *,
        modality: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []

        records: list[dict[str, Any]] = []

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

                if modality is not None and (
                    record.get("modality") != modality
                ):
                    continue

                if label is not None and record.get("label") != label:
                    continue

                records.append(record)

        return records

    @staticmethod
    def _json_default(value: Any) -> Any:
        item = getattr(value, "item", None)

        if callable(item):
            return item()

        return repr(value)
