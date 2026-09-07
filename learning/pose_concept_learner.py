from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from learning.pose_features import (
    aggregate_pose_window,
    episode_window_vectors,
)


class PoseConceptLearner:
    """Small data-driven k-nearest concept learner for pose episodes."""

    def __init__(
        self,
        vectors: np.ndarray,
        labels: list[str],
        example_counts: dict[str, int],
        *,
        neighbors_per_label: int = 3,
    ):
        if vectors.ndim != 2 or len(vectors) != len(labels):
            raise ValueError("vectors and labels must align")

        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            raise ValueError(
                "Pose learner needs examples from at least two labels"
            )

        self.vectors = vectors.astype(np.float32)
        self.labels = np.asarray(labels)
        self.unique_labels = unique_labels
        self.example_counts = dict(example_counts)
        self.neighbors_per_label = max(1, neighbors_per_label)

        self.center = np.median(self.vectors, axis=0)
        mad = np.median(
            np.abs(self.vectors - self.center),
            axis=0,
        ) * 1.4826
        self.scale = np.where(mad < 0.08, 0.08, mad)
        self.standardized = (
            self.vectors - self.center
        ) / self.scale

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path = "data/teaching_examples.jsonl",
    ) -> "PoseConceptLearner":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        vectors: list[np.ndarray] = []
        labels: list[str] = []
        example_counts: Counter[str] = Counter()

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from exc

                if record.get("modality") != "pose":
                    continue

                label = str(record.get("label") or "").strip()

                if not label:
                    continue

                episode_vectors = episode_window_vectors(
                    record.get("samples") or []
                )

                if not episode_vectors:
                    continue

                vectors.extend(episode_vectors)
                labels.extend([label] * len(episode_vectors))
                example_counts[label] += 1

        if not vectors:
            raise ValueError("No usable pose teaching examples")

        return cls(
            np.stack(vectors),
            labels,
            dict(example_counts),
        )

    def predict(
        self,
        frame_vectors: list[np.ndarray],
    ) -> dict[str, Any] | None:
        query = aggregate_pose_window(frame_vectors)

        if query is None:
            return None

        standardized_query = (
            query - self.center
        ) / self.scale

        distances: dict[str, float] = {}

        for label in self.unique_labels:
            candidates = np.sqrt(np.mean(
                (
                    self.standardized[self.labels == label]
                    - standardized_query
                ) ** 2,
                axis=1,
            ))
            count = min(
                self.neighbors_per_label,
                len(candidates),
            )
            nearest = np.partition(
                candidates,
                count - 1,
            )[:count]
            distances[label] = float(np.mean(nearest))

        ordered = sorted(
            distances.items(),
            key=lambda item: item[1],
        )
        predicted_label = ordered[0][0]
        distance_values = np.asarray(
            [distance for _, distance in ordered],
            dtype=np.float32,
        )
        temperature = max(
            0.1,
            float(np.median(distance_values)),
        )
        scores = np.exp(-distance_values / temperature)
        confidence = float(scores[0] / np.sum(scores))

        return {
            "label": predicted_label,
            "confidence": confidence,
            "distances": distances,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "labels": self.unique_labels,
            "example_counts": self.example_counts,
            "training_windows": len(self.vectors),
        }
