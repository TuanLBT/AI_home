from __future__ import annotations

from collections import deque
from typing import Any

from learning.pose_concept_learner import PoseConceptLearner
from learning.pose_features import pose_frame_vector


class PoseConceptObserver:
    """Runs learned pose concepts in shadow mode without emitting actions."""

    def __init__(
        self,
        learner: PoseConceptLearner,
        *,
        window_s: float = 0.65,
        prediction_interval_s: float = 0.5,
        min_frames: int = 5,
    ):
        self.learner = learner
        self.window_s = window_s
        self.prediction_interval_s = prediction_interval_s
        self.min_frames = min_frames
        self._history: dict[
            str,
            deque[tuple[float, Any]],
        ] = {}
        self._last_prediction_at: dict[str, float] = {}

    def update(
        self,
        observations,
        now: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        for observation in observations:
            if observation.type != "person_pose":
                continue

            vector = pose_frame_vector(
                observation.data.get("keypoints", []),
                observation.data.get(
                    "keypoint_confidences",
                    [],
                ),
            )

            if vector is None:
                continue

            entity_id = observation.entity_id
            history = self._history.setdefault(
                entity_id,
                deque(),
            )
            history.append((now, vector))

            cutoff = now - self.window_s

            while history and history[0][0] < cutoff:
                history.popleft()

            last_prediction = self._last_prediction_at.get(
                entity_id,
                0.0,
            )

            if (
                now - last_prediction
                < self.prediction_interval_s
            ):
                continue

            if len(history) < self.min_frames:
                continue

            prediction = self.learner.predict([
                item[1]
                for item in history
            ])

            if prediction is None:
                continue

            self._last_prediction_at[entity_id] = now
            events.append({
                "type": "LEARNED_CONCEPT_SHADOW",
                "entity_id": entity_id,
                "timestamp": now,
                **prediction,
            })

        return events
