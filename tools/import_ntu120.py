from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.episode import LearningEpisode
from memory.episode_store import EpisodeStore


NTU_HAND_WAVING_LABEL = 22  # A23 in zero-based MMAction2 labels


def _annotations(payload):
    if isinstance(payload, dict):
        if isinstance(payload.get("annotations"), list):
            return payload["annotations"]
        if isinstance(payload.get("data_list"), list):
            return payload["data_list"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported NTU pickle format: expected list or dict with annotations/data_list")


def _primary_person(keypoint: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    # Expected MMAction2 shape: [M, T, V, 2], score [M, T, V].
    if keypoint.ndim == 3:
        keypoint = keypoint[None, ...]
    if score.ndim == 2:
        score = score[None, ...]

    if keypoint.ndim != 4 or score.ndim != 3:
        return None
    if keypoint.shape[0] != score.shape[0] or keypoint.shape[1:3] != score.shape[1:3]:
        return None

    valid = score > 0
    person_quality = np.where(valid, score, 0.0).sum(axis=(1, 2)) / np.maximum(valid.sum(axis=(1, 2)), 1)
    index = int(np.argmax(person_quality))
    return keypoint[index].astype(np.float32), score[index].astype(np.float32)


def _motion_score(points: np.ndarray, scores: np.ndarray) -> float:
    if len(points) < 2:
        return float("inf")

    # Normalize motion by shoulder width when COCO-17 joints are available.
    if points.shape[1] >= 7:
        shoulder_width = np.linalg.norm(points[:, 6] - points[:, 5], axis=1)
        scale = float(np.median(shoulder_width[shoulder_width > 1.0])) if np.any(shoulder_width > 1.0) else 1.0
    else:
        span = np.ptp(points.reshape(-1, 2), axis=0)
        scale = max(float(np.linalg.norm(span)), 1.0)

    delta = np.linalg.norm(np.diff(points, axis=0), axis=2)
    valid = (scores[1:] > 0.2) & (scores[:-1] > 0.2)
    if not np.any(valid):
        return float("inf")
    return float(np.median(delta[valid]) / scale)


def _to_samples(points: np.ndarray, scores: np.ndarray, *, target_fps: float, max_frames: int) -> list[dict]:
    frame_count = len(points)
    if frame_count == 0:
        return []

    if frame_count > max_frames:
        indices = np.linspace(0, frame_count - 1, max_frames).round().astype(int)
    else:
        indices = np.arange(frame_count)

    samples: list[dict] = []
    for out_idx, frame_idx in enumerate(indices):
        frame_points = points[frame_idx]
        frame_scores = scores[frame_idx]
        if frame_points.shape != (17, 2) or frame_scores.shape != (17,):
            continue
        samples.append({
            "relative_time_s": out_idx / target_fps,
            "entity_id": "ntu_primary_person",
            "confidence": float(np.mean(frame_scores)),
            "bbox": [],
            "keypoints": frame_points.tolist(),
            "keypoint_confidences": frame_scores.tolist(),
        })
    return samples


def _episode(annotation: dict, *, label: str, source_label: int, points: np.ndarray, scores: np.ndarray, target_fps: float, max_frames: int, motion_score: float | None) -> LearningEpisode | None:
    samples = _to_samples(points, scores, target_fps=target_fps, max_frames=max_frames)
    if not samples:
        return None

    started_at = time.time()
    duration = samples[-1]["relative_time_s"] if samples else 0.0
    frame_dir = annotation.get("frame_dir") or annotation.get("filename") or annotation.get("sample_name")

    return LearningEpisode(
        source="video",
        started_at=started_at,
        ended_at=started_at + duration,
        entity_id="ntu_primary_person",
        observations={"pose": {"samples": samples, "sample_count": len(samples)}},
        context={"capture_mode": "public_dataset", "dataset": "NTU RGB+D 120 2D"},
        proposed_meaning=label,
        confidence=1.0,
        label_origin="import",
        verified=True,
        metadata={
            "dataset": "ntu120_2d",
            "source_action_label_zero_based": source_label,
            "source_sample": frame_dir,
            "motion_score": motion_score,
            "target_fps": target_fps,
            "original_frames": int(len(points)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a balanced CALL_ATTENTION/IDLE subset from MMAction2 ntu120_2d.pkl")
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/public/ntu120_2d.pkl"))
    parser.add_argument("--per-label", type=int, default=200, help="Maximum imported episodes per project label")
    parser.add_argument("--idle-pool", type=int, default=3000, help="How many non-A23 samples to inspect for low-motion IDLE negatives")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/episodes.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Dataset not found: {args.input}")
    if args.per_label < 1 or args.max_frames < 5 or args.fps <= 0:
        parser.error("Invalid import limits")

    print(f"Loading {args.input} ...")
    with args.input.open("rb") as file:
        payload = pickle.load(file)
    annotations = _annotations(payload)
    print(f"Annotations: {len(annotations)}")

    rng = random.Random(args.seed)
    positives: list[tuple[dict, np.ndarray, np.ndarray]] = []
    negative_candidates: list[tuple[float, dict, int, np.ndarray, np.ndarray]] = []

    order = list(range(len(annotations)))
    rng.shuffle(order)
    scanned_idle = 0

    for idx in order:
        ann = annotations[idx]
        if not isinstance(ann, dict):
            continue
        try:
            source_label = int(ann.get("label"))
        except (TypeError, ValueError):
            continue

        keypoint = np.asarray(ann.get("keypoint"), dtype=np.float32)
        keypoint_score = np.asarray(ann.get("keypoint_score"), dtype=np.float32)
        selected = _primary_person(keypoint, keypoint_score)
        if selected is None:
            continue
        points, scores = selected
        if points.shape[1:] != (17, 2):
            continue

        if source_label == NTU_HAND_WAVING_LABEL:
            if len(positives) < args.per_label:
                positives.append((ann, points, scores))
            continue

        if scanned_idle >= args.idle_pool:
            continue
        scanned_idle += 1
        motion = _motion_score(points, scores)
        if np.isfinite(motion):
            negative_candidates.append((motion, ann, source_label, points, scores))

        if len(positives) >= args.per_label and scanned_idle >= args.idle_pool:
            break

    negative_candidates.sort(key=lambda item: item[0])
    negatives = negative_candidates[: args.per_label]

    print(f"CALL_ATTENTION candidates: {len(positives)}")
    print(f"IDLE low-motion candidates: {len(negatives)} / scanned {scanned_idle}")
    if negatives:
        print(f"IDLE motion range: {negatives[0][0]:.5f} .. {negatives[-1][0]:.5f}")

    if args.dry_run:
        print("Dry run: nothing written.")
        return

    store = EpisodeStore(args.output)
    counts: Counter[str] = Counter()

    for ann, points, scores in positives:
        episode = _episode(
            ann,
            label="CALL_ATTENTION",
            source_label=NTU_HAND_WAVING_LABEL,
            points=points,
            scores=scores,
            target_fps=args.fps,
            max_frames=args.max_frames,
            motion_score=_motion_score(points, scores),
        )
        if episode is not None:
            store.append(episode)
            counts["CALL_ATTENTION"] += 1

    for motion, ann, source_label, points, scores in negatives:
        episode = _episode(
            ann,
            label="IDLE",
            source_label=source_label,
            points=points,
            scores=scores,
            target_fps=args.fps,
            max_frames=args.max_frames,
            motion_score=motion,
        )
        if episode is not None:
            store.append(episode)
            counts["IDLE"] += 1

    print(f"Imported: {dict(counts)} -> {args.output}")


if __name__ == "__main__":
    main()
