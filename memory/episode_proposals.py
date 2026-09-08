from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any


class EpisodeProposalStore:
    """Append-only automatic meaning proposals for raw episodes."""

    def __init__(self, path: str | Path = "data/episode_proposals.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, *, episode_id: str, proposal: dict[str, Any]) -> dict[str, Any]:
        record = {
            "episode_id": episode_id,
            "proposal": proposal,
            "proposed_at_utc": datetime.now(timezone.utc).isoformat(),
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
                        f"Invalid proposal at {self.path}:{line_number}"
                    ) from exc

                episode_id = str(record.get("episode_id") or "").strip()
                if episode_id:
                    latest[episode_id] = record

        return latest
