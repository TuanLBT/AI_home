from __future__ import annotations

from typing import Any

import numpy as np


UPPER_BODY_JOINTS = tuple(range(11))
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6


def pose_frame_vector(
    points,
    confidences,
    *,
    min_confidence: float = 0.25,
) -> np.ndarray | None:
    """Normalize upper-body pose around shoulder center and width."""
    points_array = np.asarray(points, dtype=np.float32)
    confidence_array = np.asarray(confidences, dtype=np.float32)

    if points_array.shape != (17, 2):
        return None

    if confidence_array.shape != (17,):
        return None

    if min(
        confidence_array[LEFT_SHOULDER],
        confidence_array[RIGHT_SHOULDER],
    ) < min_confidence:
        return None

    left_shoulder = points_array[LEFT_SHOULDER]
    right_shoulder = points_array[RIGHT_SHOULDER]
    origin = (left_shoulder + right_shoulder) / 2.0
    scale = float(np.linalg.norm(right_shoulder - left_shoulder))

    if scale < 10.0:
        return None

    values: list[float] = []

    for joint in UPPER_BODY_JOINTS:
        point = points_array[joint]
        confidence = float(confidence_array[joint])

        if (
            confidence >= min_confidence
            and point[0] > 0
            and point[1] > 0
        ):
            normalized = (point - origin) / scale
            values.extend([
                float(normalized[0]),
                float(normalized[1]),
                confidence,
            ])
        else:
            values.extend([0.0, 0.0, 0.0])

    return np.asarray(values, dtype=np.float32)


def aggregate_pose_window(
    vectors: list[np.ndarray],
) -> np.ndarray | None:
    if not vectors:
        return None

    return np.median(
        np.stack(vectors),
        axis=0,
    ).astype(np.float32)


def episode_window_vectors(
    samples: list[dict[str, Any]],
    *,
    window_s: float = 0.65,
    min_frames: int = 5,
) -> list[np.ndarray]:
    rows: list[tuple[float, np.ndarray]] = []

    for sample in samples:
        vector = pose_frame_vector(
            sample.get("keypoints", []),
            sample.get("keypoint_confidences", []),
        )

        if vector is None:
            continue

        rows.append((
            float(sample.get("relative_time_s", 0.0)),
            vector,
        ))

    if not rows:
        return []

    start = rows[0][0]
    end = rows[-1][0]
    stride = window_s / 2.0
    windows: list[np.ndarray] = []
    cursor = start

    while cursor <= end:
        vectors = [
            vector
            for timestamp, vector in rows
            if cursor <= timestamp < cursor + window_s
        ]

        if len(vectors) >= min_frames:
            aggregate = aggregate_pose_window(vectors)

            if aggregate is not None:
                windows.append(aggregate)

        cursor += stride

    if windows:
        return windows

    aggregate = aggregate_pose_window([
        vector
        for _, vector in rows
    ])
    return [aggregate] if aggregate is not None else []
