from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Iterable

from learning.episode import LearningEpisode


class EpisodeStore:
    """Append-only JSONL store for source-agnostic learning episodes."""

    def __init__(self, path: str | Path = "data/episodes.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, episode: LearningEpisode) -> str:
        record = episode.to_dict()
        record["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
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

        return episode.episode_id

    def append_many(self, episodes: Iterable[LearningEpisode]) -> list[str]:
        return [self.append(episode) for episode in episodes]

    def iter_episodes(
        self,
        *,
        verified_only: bool = False,
        proposed_meaning: str | None = None,
        source: str | None = None,
    ):
        if not self.path.is_file():
            return

        with self.path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record: dict[str, Any] = json.loads(line)
                    episode = LearningEpisode.from_dict(record)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid episode at {self.path}:{line_number}"
                    ) from exc

                if verified_only and not episode.verified:
                    continue
                if proposed_meaning is not None and episode.proposed_meaning != proposed_meaning:
                    continue
                if source is not None and episode.source != source:
                    continue

                yield episode

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
