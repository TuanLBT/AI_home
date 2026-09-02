from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PersonEventMemory:
    last_entered_at: float | None = None
    entered_approach_emitted: bool = False

    sat_at: float | None = None
    sat_and_stayed_emitted: bool = False

    gesture_session_active: dict[str, bool] = field(default_factory=dict)
    gesture_end_candidate_at: dict[str, float] = field(default_factory=dict)
    gesture_started_at: dict[str, float] = field(default_factory=dict)
    both_hands_session_active: bool = False


class EventEngine:
    """
    High-level semantic events built from low-level perception/world-state events.

    Low-level examples:
        ENTERED
        APPROACHING
        SAT_DOWN
        GESTURE_STARTED

    High-level examples:
        ENTERED_AND_APPROACHING
        SAT_AND_STAYED
        GESTURE_WHILE_PRESENT
    """

    def __init__(
        self,
        entered_approach_window_s: float = 3.0,
        sat_stay_s: float = 3.0,
        gesture_rearm_s: float = 1.0,
    ):
        self.entered_approach_window_s = entered_approach_window_s
        self.sat_stay_s = sat_stay_s
        self.gesture_rearm_s = gesture_rearm_s

        self.people: dict[str, PersonEventMemory] = {}

    def process(
        self,
        low_level_events: list[dict],
        world,
        now: float,
    ) -> list[dict]:
        high_level_events: list[dict] = []

        for event in low_level_events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            memory = self.people.setdefault(
                entity_id,
                PersonEventMemory(),
            )

            event_type = event["type"]

            if event_type == "ENTERED":
                memory.last_entered_at = now
                memory.entered_approach_emitted = False
                memory.sat_at = None
                memory.sat_and_stayed_emitted = False

            elif event_type == "APPROACHING":
                if (
                    memory.last_entered_at is not None
                    and not memory.entered_approach_emitted
                    and now - memory.last_entered_at
                    <= self.entered_approach_window_s
                ):
                    memory.entered_approach_emitted = True

                    high_level_events.append({
                        "type": "ENTERED_AND_APPROACHING",
                        "entity_id": entity_id,
                        "timestamp": now,
                    })

            elif event_type == "SAT_DOWN":
                memory.sat_at = now
                memory.sat_and_stayed_emitted = False

            elif event_type == "STOOD_UP":
                memory.sat_at = None
                memory.sat_and_stayed_emitted = False

            elif event_type == "GESTURE_STARTED":
                person = world.people.get(entity_id)
                gesture = event["gesture"]

                if person is not None and person.present:
                    end_candidate_at = memory.gesture_end_candidate_at.get(
                        gesture
                    )

                    if end_candidate_at is not None:
                        if now - end_candidate_at < self.gesture_rearm_s:
                            memory.gesture_end_candidate_at.pop(
                                gesture,
                                None,
                            )
                        else:
                            memory.gesture_session_active[gesture] = False
                            memory.gesture_end_candidate_at.pop(
                                gesture,
                                None,
                            )

                    if not memory.gesture_session_active.get(
                        gesture,
                        False,
                    ):
                        memory.gesture_session_active[gesture] = True
                        memory.gesture_started_at[gesture] = now

                    left_active = memory.gesture_session_active.get(
                        "LEFT_HAND_RAISED",
                        False,
                    )
                    right_active = memory.gesture_session_active.get(
                        "RIGHT_HAND_RAISED",
                        False,
                    )

                    if left_active and right_active:
                        if not memory.both_hands_session_active:
                            memory.both_hands_session_active = True

                            high_level_events.append({
                                "type": "GESTURE_WHILE_PRESENT",
                                "entity_id": entity_id,
                                "gesture": "BOTH_HANDS_RAISED",
                                "timestamp": now,
                            })
                    else:
                        high_level_events.append({
                            "type": "GESTURE_WHILE_PRESENT",
                            "entity_id": entity_id,
                            "gesture": gesture,
                            "timestamp": now,
                        })

            elif event_type == "GESTURE_ENDED":
                gesture = event["gesture"]

                if memory.gesture_session_active.get(
                    gesture,
                    False,
                ):
                    # Do not re-arm immediately. Pose can flicker briefly even
                    # while the user is still holding the same gesture.
                    memory.gesture_end_candidate_at[gesture] = now

            elif event_type == "LEFT":
                memory.sat_at = None
                memory.sat_and_stayed_emitted = False
                memory.gesture_session_active.clear()
                memory.gesture_end_candidate_at.clear()
                memory.gesture_started_at.clear()
                memory.both_hands_session_active = False

        # Timed event: person sat down and actually stayed seated.
        for entity_id, memory in self.people.items():
            if (
                memory.sat_at is None
                or memory.sat_and_stayed_emitted
            ):
                continue

            person = world.people.get(entity_id)

            if (
                person is None
                or not person.present
                or person.posture != "sitting"
            ):
                continue

            seated_for = now - memory.sat_at

            if seated_for >= self.sat_stay_s:
                memory.sat_and_stayed_emitted = True

                high_level_events.append({
                    "type": "SAT_AND_STAYED",
                    "entity_id": entity_id,
                    "duration": seated_for,
                    "timestamp": now,
                })

        # Re-arm gesture sessions only after the gesture has remained
        # ended continuously for long enough.
        for memory in self.people.values():
            for gesture, ended_at in list(
                memory.gesture_end_candidate_at.items()
            ):
                if now - ended_at >= self.gesture_rearm_s:
                    memory.gesture_session_active[gesture] = False
                    memory.gesture_end_candidate_at.pop(
                        gesture,
                        None,
                    )
                    memory.gesture_started_at.pop(
                        gesture,
                        None,
                    )

            left_active = memory.gesture_session_active.get(
                "LEFT_HAND_RAISED",
                False,
            )
            right_active = memory.gesture_session_active.get(
                "RIGHT_HAND_RAISED",
                False,
            )

            if not left_active and not right_active:
                memory.both_hands_session_active = False

        return high_level_events
