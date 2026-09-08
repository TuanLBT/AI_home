from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.pose_tracker import PersonPoseTracker
from config import Config
from learning.episode import LearningEpisode
from memory.episode_store import EpisodeStore


def select_primary_person(observations):
    people = [
        observation
        for observation in observations
        if observation.type == "person_pose"
    ]

    if not people:
        return None

    return max(people, key=lambda observation: observation.confidence)


def draw_status(frame, text: str, color=(255, 255, 255)) -> None:
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def countdown(cap, seconds: float, label: str) -> bool:
    deadline = time.monotonic() + seconds

    while time.monotonic() < deadline:
        ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Camera frame read failed during countdown")

        remaining = max(0.0, deadline - time.monotonic())
        draw_status(
            frame,
            f"GET READY: {label} ({remaining:.1f}s)",
            (0, 255, 255),
        )
        cv2.imshow("Indoor AI Teach Mode", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False

    return True


def capture_pose_episode(
    cap,
    tracker: PersonPoseTracker,
    *,
    label: str,
    duration_s: float,
) -> tuple[list[dict], str | None, float]:
    started_monotonic = time.monotonic()
    started_wall = time.time()
    samples: list[dict] = []
    primary_entity_id: str | None = None

    while True:
        now = time.monotonic()

        if now - started_monotonic >= duration_s:
            break

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Camera frame read failed while recording")

        observation = select_primary_person(
            tracker.process(frame)
        )

        if observation is not None:
            if primary_entity_id is None:
                primary_entity_id = observation.entity_id

            samples.append({
                "relative_time_s": now - started_monotonic,
                "entity_id": observation.entity_id,
                "confidence": observation.confidence,
                "bbox": observation.data.get("bbox", []),
                "keypoints": observation.data.get("keypoints", []),
                "keypoint_confidences": observation.data.get(
                    "keypoint_confidences",
                    [],
                ),
            })

        draw_status(
            frame,
            f"RECORDING: {label} ({now - started_monotonic:.1f}s)",
            (0, 0, 255),
        )
        cv2.imshow("Indoor AI Teach Mode", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    return samples, primary_entity_id, started_wall


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record labeled learning episodes for Indoor AI."
    )
    parser.add_argument(
        "--modality",
        default="pose",
        choices=["pose"],
        help="Input adapter to record. More modalities will be added later.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Arbitrary concept label, for example IDLE or CALL_ATTENTION.",
    )
    parser.add_argument("--takes", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--countdown", type=float, default=2.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/episodes.jsonl"),
        help="Generic episode JSONL output.",
    )
    args = parser.parse_args()

    if args.takes < 1:
        parser.error("--takes must be at least 1")

    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")

    cfg = Config()
    cap = cv2.VideoCapture(cfg.camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {cfg.camera_index}"
        )

    tracker = PersonPoseTracker(
        model_name=cfg.model_name,
        confidence=cfg.confidence,
        imgsz=cfg.imgsz,
        device=cfg.device,
    )
    store = EpisodeStore(args.output)

    print(
        f"Teach Mode: label={args.label!r}, "
        f"modality={args.modality}, takes={args.takes}"
    )
    print("Press Q or ESC to stop.")

    saved = 0

    try:
        for take in range(1, args.takes + 1):
            print(f"Take {take}/{args.takes}: get ready...")

            if not countdown(cap, args.countdown, args.label):
                break

            samples, entity_id, started_at = capture_pose_episode(
                cap,
                tracker,
                label=args.label,
                duration_s=args.seconds,
            )

            if not samples:
                print("No person pose captured; take was not saved.")
                continue

            actual_duration_s = float(samples[-1]["relative_time_s"])
            episode = LearningEpisode(
                source="camera",
                started_at=started_at,
                ended_at=started_at + actual_duration_s,
                entity_id=entity_id,
                observations={
                    "pose": {
                        "samples": samples,
                        "sample_count": len(samples),
                    }
                },
                context={
                    "capture_mode": "teach",
                    "modality": args.modality,
                },
                proposed_meaning=args.label,
                confidence=1.0,
                label_origin="human",
                verified=True,
                metadata={
                    "take": take,
                    "requested_duration_s": args.seconds,
                    "actual_duration_s": actual_duration_s,
                    "model": cfg.model_name,
                    "imgsz": cfg.imgsz,
                    "keypoint_confidence": cfg.keypoint_confidence,
                },
            )
            episode_id = store.append(episode)
            saved += 1
            print(
                f"Saved episode {episode_id} "
                f"with {len(samples)} pose samples."
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved} learning episode(s) to {args.output}.")


if __name__ == "__main__":
    main()
