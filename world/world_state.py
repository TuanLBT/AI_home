from __future__ import annotations

from typing import Iterable
import time
from statistics import median

from perception.observation import Observation
from world.entities import PersonEntity
from world.motion_fusion import fuse_motion
from gesture.pose_rules import infer_gestures, infer_posture


class WorldState:
    def __init__(
        self,
        left_timeout_s: float = 1.5,
        present_log_interval_s: float = 2.0,
        keypoint_confidence: float = 0.35,
        gesture_confirm_frames: int = 6,
        gesture_end_confirm_frames: int = 6,
        posture_confirm_frames: int = 2,

        # Depth detector
        movement_window_s: float = 0.8,
        movement_start_threshold: float = 0.10,
        movement_stop_threshold: float = 0.02,
        movement_confirm_frames: int = 5,

        # Horizontal detector
        horizontal_window_s: float = 0.6,
        horizontal_start_threshold: float = 0.10,
        horizontal_stop_threshold: float = 0.03,
        horizontal_confirm_frames: int = 5,
    ):
        self.people: dict[str, PersonEntity] = {}
        self.track_to_entity: dict[int, str] = {}
        self.next_person_id = 1

        self.action_feedback: dict[str, dict] = {}
        self.last_action_result: dict | None = None

        self.left_timeout_s = left_timeout_s
        self.present_log_interval_s = present_log_interval_s
        self.keypoint_confidence = keypoint_confidence
        self.gesture_confirm_frames = max(1, gesture_confirm_frames)
        self.gesture_end_confirm_frames = max(1, gesture_end_confirm_frames)
        self.posture_confirm_frames = max(1, posture_confirm_frames)

        self.movement_window_s = movement_window_s
        self.movement_start_threshold = movement_start_threshold
        self.movement_stop_threshold = movement_stop_threshold
        self.movement_confirm_frames = max(1, movement_confirm_frames)

        self.horizontal_window_s = horizontal_window_s
        self.horizontal_start_threshold = horizontal_start_threshold
        self.horizontal_stop_threshold = horizontal_stop_threshold
        self.horizontal_confirm_frames = max(1, horizontal_confirm_frames)

    def record_action_result(self, result: dict) -> None:
        if result.get("type") != "ACTION_EXECUTED":
            return

        entity_id = result.get("entity_id")
        if entity_id is None:
            return

        snapshot = {
            "request_id": result.get("request_id"),
            "entity_id": entity_id,
            "action": result.get("action"),
            "command": result.get("command"),
            "status": result.get("status"),
            "reason": result.get("reason"),
            "actual": dict(result.get("actual") or {}),
            "observations": dict(result.get("observations") or {}),
            "goal_id": result.get("goal_id"),
            "goal_reached": result.get("goal_reached"),
            "timestamp": result.get("timestamp"),
            "finished_at": result.get("finished_at"),
            "speech_ok": result.get("speech_ok"),
            "speech_error": result.get("speech_error"),
        }

        self.action_feedback[entity_id] = snapshot
        self.last_action_result = snapshot

    def get_action_feedback(self, entity_id: str | None = None) -> dict | None:
        if entity_id is None:
            if self.last_action_result is None:
                return None
            return dict(self.last_action_result)

        result = self.action_feedback.get(entity_id)
        return dict(result) if result is not None else None

    def update(
        self,
        observations: Iterable[Observation],
        now: float | None = None,
    ) -> list[dict]:
        now = time.monotonic() if now is None else now
        events: list[dict] = []
        seen_this_cycle: set[str] = set()

        for obs in observations:
            if obs.type != "person_pose":
                continue

            entity_id = self._resolve_entity_id(obs)
            seen_this_cycle.add(entity_id)
            person = self.people.get(entity_id)

            if person is None:
                person = PersonEntity(
                    entity_id=entity_id,
                    present=True,
                    first_seen=now,
                    last_seen=now,
                    last_present_log=now,
                    confidence=obs.confidence,
                    bbox=obs.data["bbox"],
                )
                self.people[entity_id] = person
                events.append(self._event("ENTERED", person, now))
            else:
                if not person.present:
                    person.present = True
                    person.first_seen = now
                    person.last_present_log = now
                    events.append(self._event("ENTERED", person, now))

                person.last_seen = now
                person.confidence = obs.confidence
                person.bbox = obs.data["bbox"]

                if now - person.last_present_log >= self.present_log_interval_s:
                    person.last_present_log = now
                    events.append(self._event("PRESENT", person, now))

            person.keypoints = obs.data.get("keypoints", [])
            person.keypoint_confidences = obs.data.get(
                "keypoint_confidences", []
            )

            events.extend(self._update_gestures(person, now))
            events.extend(self._update_posture(person, now))

            # Independent low-level motion detectors.
            events.extend(self._update_depth_movement(person, now))
            events.extend(self._update_horizontal_movement(person, now))

            # Higher-level fusion layer.
            events.extend(self._update_motion_fusion(person, now))

        for person in self.people.values():
            if person.present and person.entity_id not in seen_this_cycle:
                if now - person.last_seen >= self.left_timeout_s:
                    person.present = False
                    events.append(self._event("LEFT", person, now))

                    for gesture in sorted(person.active_gestures):
                        events.append(
                            self._gesture_event(
                                "GESTURE_ENDED",
                                person,
                                gesture,
                                now,
                            )
                        )

                    person.active_gestures.clear()
                    person.pending_gesture_counts.clear()
                    person.pending_gesture_end_counts.clear()

                    person.posture = "unknown"
                    person.pending_posture = "unknown"
                    person.pending_posture_count = 0

                    person.scale_history.clear()
                    person.depth_measurement_history.clear()
                    person.movement_state = "stable"
                    person.pending_movement = "stable"
                    person.pending_movement_count = 0

                    person.x_history.clear()
                    person.horizontal_state = "stable"
                    person.pending_horizontal = "stable"
                    person.pending_horizontal_count = 0

                    person.fused_motion = "stationary"

        return events

    def _update_gestures(self, person: PersonEntity, now: float) -> list[dict]:
        events: list[dict] = []

        current = infer_gestures(
            person.keypoints,
            person.keypoint_confidences,
            threshold=self.keypoint_confidence,
        )

        # Start debounce.
        for gesture in current:
            person.pending_gesture_end_counts[gesture] = 0
            person.pending_gesture_counts[gesture] = (
                person.pending_gesture_counts.get(gesture, 0) + 1
            )

            if (
                gesture not in person.active_gestures
                and person.pending_gesture_counts[gesture]
                >= self.gesture_confirm_frames
            ):
                person.active_gestures.add(gesture)
                events.append(
                    self._gesture_event(
                        "GESTURE_STARTED",
                        person,
                        gesture,
                        now,
                    )
                )

        # End debounce: one noisy frame no longer ends the gesture immediately.
        known = (
            set(person.pending_gesture_counts)
            | set(person.active_gestures)
            | set(person.pending_gesture_end_counts)
        )

        for gesture in known - current:
            person.pending_gesture_counts[gesture] = 0

            if gesture not in person.active_gestures:
                person.pending_gesture_end_counts[gesture] = 0
                continue

            person.pending_gesture_end_counts[gesture] = (
                person.pending_gesture_end_counts.get(gesture, 0) + 1
            )

            if (
                person.pending_gesture_end_counts[gesture]
                >= self.gesture_end_confirm_frames
            ):
                person.active_gestures.remove(gesture)
                person.pending_gesture_end_counts[gesture] = 0
                events.append(
                    self._gesture_event(
                        "GESTURE_ENDED",
                        person,
                        gesture,
                        now,
                    )
                )

        return events

    def _update_posture(self, person: PersonEntity, now: float) -> list[dict]:
        events: list[dict] = []

        new_posture = infer_posture(
            person.keypoints,
            person.keypoint_confidences,
            threshold=self.keypoint_confidence,
        )

        if new_posture == "unknown":
            return events

        if new_posture == person.posture:
            person.pending_posture = "unknown"
            person.pending_posture_count = 0
            return events

        if new_posture == person.pending_posture:
            person.pending_posture_count += 1
        else:
            person.pending_posture = new_posture
            person.pending_posture_count = 1

        if person.pending_posture_count < self.posture_confirm_frames:
            return events

        old_posture = person.posture
        person.posture = new_posture
        person.pending_posture = "unknown"
        person.pending_posture_count = 0

        if old_posture == "unknown":
            events.append(
                self._posture_event(
                    "POSTURE_INITIALIZED",
                    person,
                    new_posture,
                    now,
                )
            )
            return events

        if old_posture == "standing" and new_posture == "sitting":
            events.append(
                self._posture_event("SAT_DOWN", person, new_posture, now)
            )
        elif old_posture == "sitting" and new_posture == "standing":
            events.append(
                self._posture_event("STOOD_UP", person, new_posture, now)
            )

        return events

    def _update_depth_movement(
        self,
        person: PersonEntity,
        now: float,
    ) -> list[dict]:
        events: list[dict] = []

        points = person.keypoints
        confs = person.keypoint_confidences

        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12

        needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

        if not all(
            idx < len(points)
            and idx < len(confs)
            and confs[idx] >= self.keypoint_confidence
            and points[idx][0] > 0
            and points[idx][1] > 0
            for idx in needed
        ):
            return events

        # Robust multi-signal body scale.
        #
        # We combine several measurements that should all grow when the
        # person approaches the camera:
        #   - shoulder width
        #   - hip width
        #   - left shoulder -> left hip
        #   - right shoulder -> right hip
        #
        # Each measurement is normalized against its own recent baseline
        # before fusion, so values with different physical lengths are
        # comparable. Median fusion prevents one noisy body part from
        # dominating the result.

        def _dist(a_idx, b_idx):
            dx = points[b_idx][0] - points[a_idx][0]
            dy = points[b_idx][1] - points[a_idx][1]
            return max(1.0, (dx * dx + dy * dy) ** 0.5)

        current_measurements = {
            "shoulder_width": _dist(LEFT_SHOULDER, RIGHT_SHOULDER),
            "hip_width": _dist(LEFT_HIP, RIGHT_HIP),
            "left_torso": _dist(LEFT_SHOULDER, LEFT_HIP),
            "right_torso": _dist(RIGHT_SHOULDER, RIGHT_HIP),
        }

        # Store raw measurements in a small per-person runtime cache.
        if not person.depth_measurement_history:
            person.depth_measurement_history = {
                name: []
                for name in current_measurements
            }

        for name, value in current_measurements.items():
            history = person.depth_measurement_history[name]
            history.append((now, value))

            cutoff = now - self.movement_window_s
            person.depth_measurement_history[name] = [
                sample
                for sample in history
                if sample[0] >= cutoff
            ]

        # Give the pose tracker a short moment to settle after ENTERED.
        # This avoids giant scale spikes during initial keypoint reacquisition.
        if now - person.first_seen < 0.6:
            return events

        normalized_scales = []

        for name, history in person.depth_measurement_history.items():
            if len(history) < 6:
                continue

            duration = history[-1][0] - history[0][0]
            if duration < self.movement_window_s * 0.65:
                continue

            n = len(history)
            segment = max(2, n // 3)

            old_values = [v for _, v in history[:segment]]
            new_values = [v for _, v in history[-segment:]]

            old_avg = sum(old_values) / len(old_values)
            new_avg = sum(new_values) / len(new_values)

            if old_avg <= 0:
                continue

            normalized_scales.append(new_avg / old_avg)

        if len(normalized_scales) < 3:
            return events

        # Median is robust against one bad keypoint group.
        scale_ratio = median(normalized_scales)
        change = scale_ratio - 1.0

        # Reject physically implausible one-window jumps.
        # These usually happen when pose keypoints are reacquired badly
        # immediately after a person re-enters the camera.
        if scale_ratio < 0.55 or scale_ratio > 1.80:
            person.pending_movement = "stable"
            person.pending_movement_count = 0
            return events

        # Use the fused change directly below.
        if change >= self.movement_start_threshold:
            candidate = "approaching"
        elif change <= -self.movement_start_threshold:
            candidate = "moving_away"
        elif abs(change) <= self.movement_stop_threshold:
            candidate = "stable"
        else:
            candidate = person.movement_state

        if candidate == person.movement_state:
            person.pending_movement = "stable"
            person.pending_movement_count = 0
            return events

        if candidate == person.pending_movement:
            person.pending_movement_count += 1
        else:
            person.pending_movement = candidate
            person.pending_movement_count = 1

        if person.pending_movement_count < self.movement_confirm_frames:
            return events

        old_state = person.movement_state
        person.movement_state = candidate
        person.pending_movement = "stable"
        person.pending_movement_count = 0

        if candidate == "approaching":
            events.append(
                self._movement_event("APPROACHING", person, change, now)
            )
        elif candidate == "moving_away":
            events.append(
                self._movement_event("MOVING_AWAY", person, change, now)
            )
        elif candidate == "stable" and old_state != "stable":
            events.append(
                self._movement_event("MOVEMENT_STOPPED", person, change, now)
            )

        return events

    def _update_horizontal_movement(
        self,
        person: PersonEntity,
        now: float,
    ) -> list[dict]:
        events: list[dict] = []

        # Use torso center instead of bbox center.
        # Raising an arm changes the bbox shape/center and caused fake LEFT/RIGHT.
        points = person.keypoints
        confs = person.keypoint_confidences

        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12

        needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

        if not all(
            idx < len(points)
            and idx < len(confs)
            and confs[idx] >= self.keypoint_confidence
            and points[idx][0] > 0
            and points[idx][1] > 0
            for idx in needed
        ):
            return events

        shoulder_center_x = (
            points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]
        ) / 2

        hip_center_x = (
            points[LEFT_HIP][0] + points[RIGHT_HIP][0]
        ) / 2

        torso_center_x = (shoulder_center_x + hip_center_x) / 2

        shoulder_width = abs(
            points[RIGHT_SHOULDER][0] - points[LEFT_SHOULDER][0]
        )
        hip_width = abs(
            points[RIGHT_HIP][0] - points[LEFT_HIP][0]
        )

        body_width = max(1.0, (shoulder_width + hip_width) / 2)

        person.x_history.append((now, torso_center_x, body_width))

        cutoff = now - self.horizontal_window_s
        person.x_history = [
            sample
            for sample in person.x_history
            if sample[0] >= cutoff
        ]

        if len(person.x_history) < 6:
            return events

        duration = person.x_history[-1][0] - person.x_history[0][0]
        if duration < self.horizontal_window_s * 0.65:
            return events

        n = len(person.x_history)
        segment = max(2, n // 3)

        old_samples = person.x_history[:segment]
        new_samples = person.x_history[-segment:]

        old_center = sum(x for _, x, _ in old_samples) / len(old_samples)
        new_center = sum(x for _, x, _ in new_samples) / len(new_samples)

        widths = [w for _, _, w in old_samples + new_samples]
        reference_width = max(1.0, sum(widths) / len(widths))

        # Horizontal displacement measured in torso-width units.
        delta = (new_center - old_center) / reference_width

        if delta >= self.horizontal_start_threshold:
            candidate = "moving_right"
        elif delta <= -self.horizontal_start_threshold:
            candidate = "moving_left"
        elif abs(delta) <= self.horizontal_stop_threshold:
            candidate = "stable"
        else:
            candidate = person.horizontal_state

        if candidate == person.horizontal_state:
            person.pending_horizontal = "stable"
            person.pending_horizontal_count = 0
            return events

        if candidate == person.pending_horizontal:
            person.pending_horizontal_count += 1
        else:
            person.pending_horizontal = candidate
            person.pending_horizontal_count = 1

        if person.pending_horizontal_count < self.horizontal_confirm_frames:
            return events

        old_state = person.horizontal_state
        person.horizontal_state = candidate
        person.pending_horizontal = "stable"
        person.pending_horizontal_count = 0

        if candidate == "moving_left":
            events.append(
                self._horizontal_event("MOVING_LEFT", person, delta, now)
            )
        elif candidate == "moving_right":
            events.append(
                self._horizontal_event("MOVING_RIGHT", person, delta, now)
            )
        elif candidate == "stable" and old_state != "stable":
            events.append(
                self._horizontal_event("HORIZONTAL_STOPPED", person, delta, now)
            )

        return events

    def _update_motion_fusion(
        self,
        person: PersonEntity,
        now: float,
    ) -> list[dict]:
        fused = fuse_motion(
            person.horizontal_state,
            person.movement_state,
        )

        if fused == person.fused_motion:
            return []

        old = person.fused_motion
        person.fused_motion = fused

        return [{
            "type": "MOTION_CHANGED",
            "entity_id": person.entity_id,
            "old_motion": old,
            "motion": fused,
            "horizontal_state": person.horizontal_state,
            "depth_state": person.movement_state,
            "timestamp": now,
        }]

    @staticmethod
    def _event(event_type: str, person: PersonEntity, now: float) -> dict:
        return {
            "type": event_type,
            "entity_id": person.entity_id,
            "timestamp": now,
            "confidence": person.confidence,
            "bbox": person.bbox,
            "seen_duration": person.seen_duration,
        }

    @staticmethod
    def _gesture_event(
        event_type: str,
        person: PersonEntity,
        gesture: str,
        now: float,
    ) -> dict:
        return {
            "type": event_type,
            "entity_id": person.entity_id,
            "gesture": gesture,
            "timestamp": now,
        }

    @staticmethod
    def _posture_event(
        event_type: str,
        person: PersonEntity,
        posture: str,
        now: float,
    ) -> dict:
        return {
            "type": event_type,
            "entity_id": person.entity_id,
            "posture": posture,
            "timestamp": now,
        }

    @staticmethod
    def _movement_event(
        event_type: str,
        person: PersonEntity,
        change: float,
        now: float,
    ) -> dict:
        return {
            "type": event_type,
            "entity_id": person.entity_id,
            "scale_change": change,
            "timestamp": now,
        }

    @staticmethod
    def _horizontal_event(
        event_type: str,
        person: PersonEntity,
        delta: float,
        now: float,
    ) -> dict:
        return {
            "type": event_type,
            "entity_id": person.entity_id,
            "horizontal_delta": delta,
            "timestamp": now,
        }

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _resolve_entity_id(self, obs):
        track_id = int(obs.data["track_id"])

        if track_id in self.track_to_entity:
            return self.track_to_entity[track_id]

        bbox = obs.data["bbox"]
        best_entity = None
        best_iou = 0.0

        for entity_id, person in self.people.items():
            if not person.present:
                continue

            iou = self._bbox_iou(bbox, person.bbox)

            if iou > best_iou:
                best_iou = iou
                best_entity = entity_id

        if best_entity is not None and best_iou >= 0.25:
            self.track_to_entity[track_id] = best_entity
            print(
                f"tracker reassigned: track_{track_id} -> "
                f"{best_entity} (IoU={best_iou:.2f})"
            )
            return best_entity

        entity_id = f"person_{self.next_person_id}"
        self.next_person_id += 1
        self.track_to_entity[track_id] = entity_id
        return entity_id
