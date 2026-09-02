from dataclasses import dataclass, field


@dataclass(slots=True)
class PersonEntity:
    entity_id: str
    present: bool
    first_seen: float
    last_seen: float
    last_present_log: float
    confidence: float
    bbox: list[float]

    keypoints: list[list[float]] = field(default_factory=list)
    keypoint_confidences: list[float] = field(default_factory=list)

    active_gestures: set[str] = field(default_factory=set)
    pending_gesture_counts: dict[str, int] = field(default_factory=dict)
    pending_gesture_end_counts: dict[str, int] = field(default_factory=dict)

    posture: str = "unknown"
    pending_posture: str = "unknown"
    pending_posture_count: int = 0

    scale_history: list[tuple[float, float]] = field(default_factory=list)
    depth_measurement_history: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    movement_state: str = "stable"
    pending_movement: str = "stable"
    pending_movement_count: int = 0

    # (timestamp, center_x, bbox_width)
    x_history: list[tuple[float, float, float]] = field(default_factory=list)
    horizontal_state: str = "stable"
    pending_horizontal: str = "stable"
    pending_horizontal_count: int = 0

    fused_motion: str = "stationary"

    @property
    def seen_duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)
