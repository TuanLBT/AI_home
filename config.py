from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    camera_index: int = 0

    # One pose model does: person detection + tracking + body keypoints.
    model_name: str = "yolo11n-pose.pt"

    device: str = "cpu"
    confidence: float = 0.45
    keypoint_confidence: float = 0.35
    imgsz: int = 320
    perception_fps: float = 30

    left_timeout_s: float = 1.5
    present_log_interval_s: float = 2.0

    # At 30 FPS this is about 0.33 s. Two frames was only 0.07 s and
    # allowed momentary keypoint jumps to become real gesture events.
    gesture_confirm_frames: int = 10

    show_window: bool = True
    window_name: str = "Indoor AI - M1 Pose/Gesture"
