from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from learning.episode import LearningEpisode, legacy_pose_record_to_episode
from memory.episode_proposals import EpisodeProposalStore
from memory.episode_quality import EpisodeQualityStore
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


class TrainingMemory:
    """Resolve raw episodes plus reviews/proposals/quality into training data."""

    def __init__(
        self,
        episode_path: str | Path = "data/episodes.jsonl",
        review_path: str | Path = "data/episode_reviews.jsonl",
        proposal_path: str | Path = "data/episode_proposals.jsonl",
        quality_path: str | Path = "data/episode_quality.jsonl",
        legacy_path: str | Path = "data/teaching_examples.jsonl",
    ):
        self.episode_store = EpisodeStore(episode_path)
        self.review_store = EpisodeReviewStore(review_path)
        self.proposal_store = EpisodeProposalStore(proposal_path)
        self.quality_store = EpisodeQualityStore(quality_path)
        self.legacy_path = Path(legacy_path)

    def iter_trainable(
        self,
        *,
        include_verified_raw: bool = True,
        include_legacy: bool = False,
        include_auto_trusted: bool = True,
    ) -> Iterable[LearningEpisode]:
        """Yield episodes eligible for learning.

        Legacy ``data/teaching_examples.jsonl`` data is intentionally disabled
        by default. It predates the generic episode/provenance pipeline and can
        contain low-quality or duplicate captures. Pass ``include_legacy=True``
        only for explicit audits or backwards-compatibility experiments.
        """
        reviews = self.review_store.latest_by_episode()
        proposals = self.proposal_store.latest_by_episode()
        excluded_ids = self.quality_store.excluded_ids()

        if self.episode_store.path.is_file():
            for episode in self.episode_store.iter_episodes():
                if episode.episode_id in excluded_ids:
                    continue

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
                    continue

                if include_auto_trusted:
                    proposal_record = proposals.get(episode.episode_id) or {}
                    proposal = proposal_record.get("proposal") or {}
                    trust = proposal.get("auto_trust") or {}

                    if trust.get("trusted"):
                        label = str(trust.get("label") or "").strip()
                        if label:
                            yield replace(
                                episode,
                                proposed_meaning=label,
                                confidence=float(proposal.get("confidence") or 0.0),
                                label_origin="self",
                                verified=False,
                            )

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
