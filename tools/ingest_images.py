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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_paths(inputs: list[Path]):
    for input_path in inputs:
        if input_path.is_file():
            if input_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield input_path
            continue

        if input_path.is_dir():
            for path in sorted(input_path.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local images into Indoor AI learning episodes."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--label",
        default=None,
        help="Optional proposed meaning applied to every ingested image.",
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Mark the supplied label as human-verified.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/episodes.jsonl"),
    )
    args = parser.parse_args()

    if args.verified and not args.label:
        parser.error("--verified requires --label")

    cfg = Config()
    extractor = StaticPoseExtractor(
        model_name=cfg.model_name,
        confidence=cfg.confidence,
        imgsz=cfg.imgsz,
        device=cfg.device,
    )
    store = EpisodeStore(args.output)

    image_paths = list(iter_image_paths(args.inputs))
    if not image_paths:
        parser.error("No supported image files found")

    saved = 0
    skipped = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"SKIP unreadable: {image_path}")
            skipped += 1
            continue

        poses = extractor.process(image)
        if not poses:
            print(f"SKIP no person pose: {image_path}")
            skipped += 1
            continue

        for person_index, pose in enumerate(poses):
            now = time.time()
            sample = {
                "relative_time_s": 0.0,
                "entity_id": f"image_person_{person_index}",
                "confidence": pose["confidence"],
                "bbox": pose["bbox"],
                "keypoints": pose["keypoints"],
                "keypoint_confidences": pose["keypoint_confidences"],
            }

            episode = LearningEpisode(
                source="image",
                started_at=now,
                ended_at=now,
                entity_id=sample["entity_id"],
                observations={
                    "pose": {"samples": [sample]},
                    "image": {
                        "path": str(image_path),
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    },
                },
                proposed_meaning=args.label,
                confidence=1.0 if args.label else None,
                label_origin="human" if args.label else "unknown",
                verified=bool(args.verified),
                metadata={
                    "ingestor": "tools/ingest_images.py",
                    "model": cfg.model_name,
                    "imgsz": cfg.imgsz,
                    "person_index": person_index,
                },
            )
            episode_id = store.append(episode)
            saved += 1
            print(
                f"SAVED {episode_id} label={args.label!r} "
                f"person={person_index} file={image_path}"
            )

    print(
        f"Done. Saved {saved} episode(s), skipped {skipped} image(s) "
        f"to {args.output}."
    )


if __name__ == "__main__":
    main()
