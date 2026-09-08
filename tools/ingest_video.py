from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from learning.episode import LearningEpisode
from memory.episode_store import EpisodeStore
from perception.static_pose import StaticPoseExtractor


def select_primary_pose(poses):
    if not poses:
        return None
    return max(poses, key=lambda pose: pose["confidence"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a local video into a labeled learning episode."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--label",
        default=None,
        help="Optional proposed meaning for the video episode.",
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Mark the supplied label as human-verified.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Pose extraction rate. Default: 5 FPS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/episodes.jsonl"),
    )
    args = parser.parse_args()

    if args.verified and not args.label:
        parser.error("--verified requires --label")
    if args.sample_fps <= 0:
        parser.error("--sample-fps must be greater than 0")
    if not args.input.is_file():
        parser.error(f"Video not found: {args.input}")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0
    frame_step = max(1, round(source_fps / args.sample_fps))

    cfg = Config()
    extractor = StaticPoseExtractor(
        model_name=cfg.model_name,
        confidence=cfg.confidence,
        imgsz=cfg.imgsz,
        device=cfg.device,
    )
    store = EpisodeStore(args.output)

    samples: list[dict] = []
    frame_index = 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            relative_time_s = frame_index / source_fps
            pose = select_primary_pose(extractor.process(frame))
            if pose is not None:
                samples.append({
                    "relative_time_s": relative_time_s,
                    "entity_id": "video_primary_person",
                    "confidence": pose["confidence"],
                    "bbox": pose["bbox"],
                    "keypoints": pose["keypoints"],
                    "keypoint_confidences": pose["keypoint_confidences"],
                })

            frame_index += 1
    finally:
        cap.release()

    if not samples:
        raise RuntimeError("No usable person pose found in video")

    started_at = time.time()
    duration_s = samples[-1]["relative_time_s"]
    episode = LearningEpisode(
        source="video",
        started_at=started_at,
        ended_at=started_at + duration_s,
        entity_id="video_primary_person",
        observations={
            "pose": {"samples": samples},
            "video": {
                "path": str(args.input),
                "width": width,
                "height": height,
                "source_fps": float(source_fps),
                "sample_fps": float(args.sample_fps),
                "frame_count": frame_index,
            },
        },
        proposed_meaning=args.label,
        confidence=1.0 if args.label else None,
        label_origin="human" if args.label else "unknown",
        verified=bool(args.verified),
        metadata={
            "ingestor": "tools/ingest_video.py",
            "model": cfg.model_name,
            "imgsz": cfg.imgsz,
        },
    )

    episode_id = store.append(episode)
    print(
        f"Saved {episode_id} with {len(samples)} pose samples "
        f"from {args.input} to {args.output}."
    )


if __name__ == "__main__":
    main()
