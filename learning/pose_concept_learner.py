from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from learning.episode import LearningEpisode, legacy_pose_record_to_episode
from learning.pose_features import (
    aggregate_pose_window,
    episode_window_vectors,
)


class PoseConceptLearner:
    """Small data-driven k-nearest concept learner for pose episodes."""

    DEFAULT_SOURCES = (
        Path("data/episodes.jsonl"),
        Path("data/teaching_examples.jsonl"),
    )

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
        path: str | Path | None = None,
    ) -> "PoseConceptLearner":
        """Load pose learning data.

        With no explicit path, merge the generic episode store with the old
        teaching-example store. This keeps old samples alive while all new
        Teach Mode captures go to ``data/episodes.jsonl``.
        """

        if path is None:
            return cls.from_jsonl_sources(cls.DEFAULT_SOURCES)

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        return cls._from_records(cls._iter_episode_records(path))

    @classmethod
    def from_jsonl_sources(
        cls,
        paths: Iterable[str | Path],
    ) -> "PoseConceptLearner":
        existing_paths = [Path(path) for path in paths if Path(path).is_file()]
        if not existing_paths:
            raise FileNotFoundError("No teaching or episode JSONL source found")

        def records():
            for path in existing_paths:
                yield from cls._iter_episode_records(path)

        return cls._from_records(records())

    @classmethod
    def _from_records(
        cls,
        episodes: Iterable[LearningEpisode],
    ) -> "PoseConceptLearner":
        vectors: list[np.ndarray] = []
        labels: list[str] = []
        example_counts: Counter[str] = Counter()

        for episode in episodes:
            label = str(episode.proposed_meaning or "").strip()
            if not label:
                continue

            pose = episode.observations.get("pose") or {}
            samples = pose.get("samples") or []
            episode_vectors = episode_window_vectors(samples)

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

    @staticmethod
    def _iter_episode_records(path: Path):
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

                if "observations" in record and "source" in record:
                    try:
                        yield LearningEpisode.from_dict(record)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid episode at {path}:{line_number}"
                        ) from exc
                    continue

                episode = legacy_pose_record_to_episode(record)
                if episode is not None:
                    yield episode

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
