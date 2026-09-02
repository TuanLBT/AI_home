# Indoor AI — M1: Pose / Gesture

This is the next step after M0.

## What changed

Instead of a plain object detector, this version uses:

`yolo11n-pose.pt`

That single model provides:

- person detection
- ByteTrack tracking ID
- 17 body keypoints
- simple gesture inference

Current gesture rules:

- `LEFT_HAND_RAISED`
- `RIGHT_HAND_RAISED`
- `BOTH_HANDS_RAISED`

Example:

```text
person_1 ENTERED
person_1 GESTURE_STARTED: RIGHT_HAND_RAISED
person_1 GESTURE_ENDED: RIGHT_HAND_RAISED
person_1 LEFT
```

## Why gesture confirmation exists

At 2 FPS, a single bad keypoint can otherwise create a false gesture.

Default:

```python
gesture_confirm_frames = 2
```

So the same gesture must be seen twice in a row (~1 second at 2 FPS) before
`GESTURE_STARTED` is emitted.

Set it to `1` if you want faster reaction.

## Run

You can reuse the existing virtual environment if it already contains
Ultralytics and OpenCV.

```bash
source .venv/bin/activate
python main.py
```

The pose weights will download automatically on first run.

## Main config

Edit `config.py`:

```python
perception_fps = 2.0
imgsz = 320
gesture_confirm_frames = 2
keypoint_confidence = 0.35
```

## Architecture

```text
Camera
  ↓
YOLO11n-pose
  ├─ person detection
  ├─ ByteTrack ID
  └─ body keypoints
        ↓
Observation(person_pose)
        ↓
WorldState
  ├─ ENTERED / PRESENT / LEFT
  └─ gesture temporal state
        ↓
GESTURE_STARTED / GESTURE_ENDED
```

Next step after this works reliably:
event detection such as "person sat down", "person stood up", and "person
approached camera".
