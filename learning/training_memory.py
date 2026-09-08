from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from learning.episode import LearningEpisode, legacy_pose_record_to_episode
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


class TrainingMemory:
    """Resolve raw episodes plus human reviews into trainable episodes."""

    def __init__(
        self,
        episode_path: str | Path = "data/episodes.jsonl",
        review_path: str | Path = "data/episode_reviews.jsonl",
        legacy_path: str | Path = "data/teaching_examples.jsonl",
    ):
        self.episode_store = EpisodeStore(episode_path)
        self.review_store = EpisodeReviewStore(review_path)
        self.legacy_path = Path(legacy_path)

    def iter_trainable(
        self,
        *,
        include_verified_raw: bool = True,
        include_legacy: bool = True,
    ) -> Iterable[LearningEpisode]:
        reviews = self.review_store.latest_by_episode()

        if self.episode_store.path.is_file():
            for episode in self.episode_store.iter_episodes():
                review = reviews.get(episode.episode_id)

                if review is not None:
                    decision = review.get("decision")
                    if decision == "reject":
                        continue

                    label = str(review.get("label") or "").strip()
                    if label:
                        yield replace(
                            episode,
                            proposed_meaning=label,
                            confidence=1.0,
                            label_origin="human",
                            verified=True,
                        )
                    continue

                if (
                    include_verified_raw
                    and episode.verified
                    and (episode.proposed_meaning or "").strip()
                ):
                    yield episode

        if include_legacy and self.legacy_path.is_file():
            import json

            with self.legacy_path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    episode = legacy_pose_record_to_episode(record)
                    if episode is not None and episode.proposed_meaning:
                        yield episode
