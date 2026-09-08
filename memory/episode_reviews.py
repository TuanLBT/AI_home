from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


class EpisodeReviewStore:
    """Append-only human review log for episode labels.

    Reviews are kept separate from raw episodes so source evidence remains
    immutable. The latest review for an episode is considered authoritative.
    """

    def __init__(self, path: str | Path = "data/episode_reviews.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        episode_id: str,
        decision: str,
        label: str | None = None,
        note: str | None = None,
        reviewer: str = "human",
    ) -> dict[str, Any]:
        decision = decision.strip().lower()
        if decision not in {"accept", "correct", "reject"}:
            raise ValueError("decision must be accept, correct, or reject")

        if decision in {"accept", "correct"} and not (label or "").strip():
            raise ValueError("label is required for accept/correct")

        record = {
            "episode_id": episode_id,
            "decision": decision,
            "label": (label or "").strip() or None,
            "note": note,
            "reviewer": reviewer,
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
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
                        f"Invalid review at {self.path}:{line_number}"
                    ) from exc

                episode_id = str(record.get("episode_id") or "").strip()
                if episode_id:
                    latest[episode_id] = record

        return latest
