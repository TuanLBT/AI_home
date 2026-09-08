from __future__ import annotations

from collections import deque
from typing import Any

from learning.pose_concept_learner import PoseConceptLearner
from learning.pose_features import pose_frame_vector


class PoseConceptObserver:
    """Run learned pose concepts live and confirm temporal concept events."""

    def __init__(
        self,
        learner: PoseConceptLearner,
        *,
        window_s: float = 0.65,
        prediction_interval_s: float = 0.33,
        min_frames: int = 5,
        confirmation_windows: int = 3,
        release_windows: int = 2,
        min_confirmation_confidence: float = 0.50,
        ignored_event_labels: tuple[str, ...] = ("IDLE",),
    ):
        self.learner = learner
        self.window_s = window_s
        self.prediction_interval_s = prediction_interval_s
        self.min_frames = min_frames
        self.confirmation_windows = max(1, int(confirmation_windows))
        self.release_windows = max(1, int(release_windows))
        self.min_confirmation_confidence = float(min_confirmation_confidence)
        self.ignored_event_labels = set(ignored_event_labels)

        self._history: dict[str, deque[tuple[float, Any]]] = {}
        self._last_prediction_at: dict[str, float] = {}
        self._streak_label: dict[str, str] = {}
        self._streak_count: dict[str, int] = {}

        # A confirmed label stays latched until the observer sees a stable
        # ignored/background label (normally IDLE). This makes events
        # edge-triggered: one event per gesture bout, not one every cooldown.
        self._active_confirmed_label: dict[str, str] = {}
        self._release_count: dict[str, int] = {}

    def update(
        self,
        observations,
        now: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for observation in observations:
            if observation.type != "person_pose":
                continue

            entity_id = observation.entity_id
            seen_ids.add(entity_id)

            vector = pose_frame_vector(
                observation.data.get("keypoints", []),
                observation.data.get(
                    "keypoint_confidences",
                    [],
                ),
            )

            if vector is None:
                continue

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
            label = str(prediction["label"])
            confidence = float(prediction["confidence"])

            if self._streak_label.get(entity_id) == label:
                streak = self._streak_count.get(entity_id, 0) + 1
            else:
                streak = 1

            self._streak_label[entity_id] = label
            self._streak_count[entity_id] = streak

            events.append({
                "type": "LEARNED_CONCEPT_SHADOW",
                "entity_id": entity_id,
                "timestamp": now,
                "streak": streak,
                "active_confirmed_label": self._active_confirmed_label.get(
                    entity_id
                ),
                **prediction,
            })

            if label in self.ignored_event_labels:
                release_count = self._release_count.get(entity_id, 0) + 1
                self._release_count[entity_id] = release_count

                if release_count >= self.release_windows:
                    self._active_confirmed_label.pop(entity_id, None)
                continue

            self._release_count[entity_id] = 0

            if confidence < self.min_confirmation_confidence:
                continue

            if streak < self.confirmation_windows:
                continue

            if self._active_confirmed_label.get(entity_id) == label:
                continue

            self._active_confirmed_label[entity_id] = label
            events.append({
                "type": "LEARNED_CONCEPT_CONFIRMED",
                "entity_id": entity_id,
                "timestamp": now,
                "label": label,
                "confidence": confidence,
                "streak": streak,
                "distances": prediction.get("distances", {}),
            })

        # Drop stale per-person state when a tracked person disappears.
        stale_ids = set(self._history) - seen_ids
        for entity_id in stale_ids:
            self._history.pop(entity_id, None)
            self._last_prediction_at.pop(entity_id, None)
            self._streak_label.pop(entity_id, None)
            self._streak_count.pop(entity_id, None)
            self._active_confirmed_label.pop(entity_id, None)
            self._release_count.pop(entity_id, None)

        return events
