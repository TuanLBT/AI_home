from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


class EpisodeQualityStore:
    """Append-only training eligibility overrides keyed by episode_id."""

    def __init__(self, path: str | Path = "data/episode_quality.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        episode_id: str,
        excluded: bool,
        reason: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        episode_id = str(episode_id).strip()
        if not episode_id:
            raise ValueError("episode_id is required")

        record = {
            "episode_id": episode_id,
            "excluded": bool(excluded),
            "reason": reason,
            "note": note,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()

        return record

    def latest_by_episode(self) -> dict[str, dict[str, Any]]:
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
                        f"Invalid quality record at {self.path}:{line_number}"
                    ) from exc

                episode_id = str(record.get("episode_id") or "").strip()
                if episode_id:
                    latest[episode_id] = record

        return latest

    def excluded_ids(self) -> set[str]:
        return {
            episode_id
            for episode_id, record in self.latest_by_episode().items()
            if bool(record.get("excluded"))
        }
