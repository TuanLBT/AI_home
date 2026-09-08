from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from learning.episode import LearningEpisode


@dataclass(frozen=True, slots=True)
class AutoTrustDecision:
    trusted: bool
    reason: str
    label: str | None
    confidence: float
    margin_ratio: float | None
    human_examples: int
    max_auto_examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trusted": self.trusted,
            "reason": self.reason,
            "label": self.label,
            "confidence": self.confidence,
            "margin_ratio": self.margin_ratio,
            "human_examples": self.human_examples,
            "max_auto_examples": self.max_auto_examples,
        }


class AutoTrustPolicy:
    """Conservative gate for using model-proposed labels without review.

    Auto labels are only trusted when the concept already has enough human
    anchors, the prediction is very confident, and the best distance is
    meaningfully separated from the runner-up. A per-label quota prevents
    self-labeled data from overwhelming human examples.
    """

    def __init__(
        self,
        *,
        min_human_examples: int = 5,
        min_confidence: float = 0.90,
        max_best_to_second_ratio: float = 0.72,
        max_auto_per_human_ratio: float = 0.25,
    ):
        self.min_human_examples = max(1, int(min_human_examples))
        self.min_confidence = float(min_confidence)
        self.max_best_to_second_ratio = float(max_best_to_second_ratio)
        self.max_auto_per_human_ratio = max(0.0, float(max_auto_per_human_ratio))

    @staticmethod
    def human_counts(episodes: Iterable[LearningEpisode]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for episode in episodes:
            if not episode.verified or episode.label_origin != "human":
                continue
            label = str(episode.proposed_meaning or "").strip()
            if label:
                counts[label] += 1
        return counts

    def evaluate(
        self,
        proposal: dict[str, Any],
        *,
        human_examples: int,
        existing_auto_examples: int = 0,
    ) -> AutoTrustDecision:
        label = str(
            proposal.get("candidate_label")
            or proposal.get("label")
            or ""
        ).strip() or None
        confidence = float(proposal.get("confidence") or 0.0)
        max_auto_examples = int(human_examples * self.max_auto_per_human_ratio)

        if label is None or label == "UNKNOWN":
            return AutoTrustDecision(
                False, "unknown_label", label, confidence, None,
                human_examples, max_auto_examples,
            )

        if human_examples < self.min_human_examples:
            return AutoTrustDecision(
                False, "not_enough_human_anchors", label, confidence, None,
                human_examples, max_auto_examples,
            )

        if confidence < self.min_confidence:
            return AutoTrustDecision(
                False, "confidence_too_low", label, confidence, None,
                human_examples, max_auto_examples,
            )

        distances = proposal.get("distances") or {}
        ordered = sorted(float(value) for value in distances.values())
        margin_ratio: float | None = None
        if len(ordered) >= 2 and ordered[1] > 0:
            margin_ratio = ordered[0] / ordered[1]

        if margin_ratio is None:
            return AutoTrustDecision(
                False, "insufficient_distance_margin", label, confidence, None,
                human_examples, max_auto_examples,
            )

        if margin_ratio > self.max_best_to_second_ratio:
            return AutoTrustDecision(
                False, "prediction_not_separated", label, confidence, margin_ratio,
                human_examples, max_auto_examples,
            )

        if existing_auto_examples >= max_auto_examples:
            return AutoTrustDecision(
                False, "auto_quota_reached", label, confidence, margin_ratio,
                human_examples, max_auto_examples,
            )

        return AutoTrustDecision(
            True, "trusted_by_policy", label, confidence, margin_ratio,
            human_examples, max_auto_examples,
        )
