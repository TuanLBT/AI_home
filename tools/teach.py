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
from teaching.example_store import TeachingExampleStore


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
) -> list[dict]:
    started_at = time.monotonic()
    samples: list[dict] = []

    while True:
        now = time.monotonic()

        if now - started_at >= duration_s:
            break

        ok, frame = cap.read()

        if not ok:
            raise RuntimeError("Camera frame read failed while recording")

        observation = select_primary_person(
            tracker.process(frame)
        )

        if observation is not None:
            samples.append({
                "relative_time_s": now - started_at,
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
            f"RECORDING: {label} ({now - started_at:.1f}s)",
            (0, 0, 255),
        )
        cv2.imshow("Indoor AI Teach Mode", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record labeled observation episodes for Indoor AI."
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
        default=Path("data/teaching_examples.jsonl"),
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
    store = TeachingExampleStore(args.output)

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

            samples = capture_pose_episode(
                cap,
                tracker,
                label=args.label,
                duration_s=args.seconds,
            )

            if not samples:
                print("No person pose captured; take was not saved.")
                continue

            record = store.record(
                label=args.label,
                modality=args.modality,
                samples=samples,
                duration_s=args.seconds,
                metadata={
                    "take": take,
                    "model": cfg.model_name,
                    "imgsz": cfg.imgsz,
                    "keypoint_confidence": cfg.keypoint_confidence,
                },
            )
            saved += 1
            print(
                f"Saved {record['example_id']} "
                f"with {record['sample_count']} samples."
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved {saved} teaching example(s) to {args.output}.")


if __name__ == "__main__":
    main()
