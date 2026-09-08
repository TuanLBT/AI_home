from __future__ import annotations

from dataclasses import replace
from typing import Any

from learning.episode import LearningEpisode
from learning.pose_concept_learner import PoseConceptLearner


class MeaningProposer:
    """Propose meanings for unlabeled episodes using learned concepts."""

    def __init__(
        self,
        pose_learner: PoseConceptLearner,
        *,
        accept_threshold: float = 0.65,
    ):
        self.pose_learner = pose_learner
        self.accept_threshold = float(accept_threshold)

    def propose(self, episode: LearningEpisode) -> dict[str, Any]:
        prediction = self.pose_learner.predict_episode(episode)
        if prediction is None:
            return {
                "label": "UNKNOWN",
                "confidence": 0.0,
                "accepted": False,
                "reason": "no_usable_pose",
            }

        confidence = float(prediction["confidence"])
        label = str(prediction["label"])
        accepted = confidence >= self.accept_threshold

        return {
            "label": label if accepted else "UNKNOWN",
            "candidate_label": label,
            "confidence": confidence,
            "accepted": accepted,
            "reason": "learner_prediction",
            "distances": prediction.get("distances", {}),
        }

    def apply(self, episode: LearningEpisode) -> LearningEpisode:
        proposal = self.propose(episode)
        metadata = dict(episode.metadata)
        metadata["meaning_proposal"] = proposal

        return replace(
            episode,
            proposed_meaning=(
                proposal["candidate_label"]
                if proposal.get("accepted")
                else episode.proposed_meaning
            ),
            confidence=(
                proposal["confidence"]
                if proposal.get("accepted")
                else episode.confidence
            ),
            label_origin=(
                "self" if proposal.get("accepted") else episode.label_origin
            ),
            verified=False,
            metadata=metadata,
        )
