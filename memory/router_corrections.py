from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import threading
from typing import Any


class RouterCorrectionStore:
    """Persistent human corrections for semantic observation routing.

    Corrections are data, not code. The router can retrieve relevant examples
    and generalize from them on later turns without editing its Python source or
    growing a keyword list.
    """

    def __init__(self, path: str | Path = "data/router_corrections.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        item = dict(record)
        item.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
        item.setdefault("label_origin", "human")
        item.setdefault("verified", True)
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))

        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
                file.flush()

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []

        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return []

        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            if len(out) >= max(1, int(limit)):
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("verified", True):
                out.append(item)
        out.reverse()
        return out

    def examples_for(self, text: str, limit: int = 6) -> list[dict[str, Any]]:
        candidates = self.recent(limit=80)
        if not candidates:
            return []

        query = self._normalize(text)
        query_tokens = set(query.split())
        scored: list[tuple[float, int, dict[str, Any]]] = []

        for recency, item in enumerate(candidates):
            prior = self._normalize(item.get("input_text", ""))
            feedback = self._normalize(item.get("feedback_text", ""))
            lesson = self._normalize(item.get("lesson", ""))
            haystack = " ".join(part for part in (prior, feedback, lesson) if part)
            if not haystack:
                continue

            hay_tokens = set(haystack.split())
            union = query_tokens | hay_tokens
            overlap = (
                len(query_tokens & hay_tokens) / len(union)
                if union
                else 0.0
            )
            sequence = SequenceMatcher(None, query, prior or haystack).ratio()
            score = max(overlap, sequence * 0.8)
            scored.append((score, recency, item))

        # Always let a few recent human lessons remain available even if the
        # new wording is different; the LLM performs the semantic generalizing.
        scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
        selected = [item for _, _, item in scored[: max(1, int(limit))]]

        return [
            {
                "input_text": item.get("input_text"),
                "previous_requests": item.get("previous_requests") or [],
                "corrected_requests": item.get("corrected_requests") or [],
                "lesson": item.get("lesson") or "",
                "feedback_text": item.get("feedback_text") or "",
            }
            for item in selected
        ]

    @staticmethod
    def _normalize(value: Any) -> str:
        return " ".join(str(value or "").casefold().split())
