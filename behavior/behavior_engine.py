from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BehaviorState:
    state: str = "idle"
    state_since: float = 0.0
    last_action_at: dict[str, float] = None

    def __post_init__(self):
        if self.last_action_at is None:
            self.last_action_at = {}


class BehaviorEngine:
    """
    Persistent behavior / engagement state machine.

    States:
        idle
        observing
        greeting
        attending
        settled
        disengaging

    The engine consumes both low-level world events and high-level semantic
    events. It keeps a persistent state for each person and emits one-shot
    actions only when appropriate.
    """

    def __init__(
        self,
        greeting_duration_s: float = 1.5,
        attention_timeout_s: float = 8.0,
        greet_cooldown_s: float = 20.0,
        gesture_action_cooldown_s: float = 1.0,
        voice_reply_cooldown_s: float = 2.0,
    ):
        self.greeting_duration_s = greeting_duration_s
        self.attention_timeout_s = attention_timeout_s
        self.greet_cooldown_s = greet_cooldown_s
        self.gesture_action_cooldown_s = gesture_action_cooldown_s
        self.voice_reply_cooldown_s = voice_reply_cooldown_s
        self.people: dict[str, BehaviorState] = {}

    def process(
        self,
        low_level_events: list[dict],
        high_level_events: list[dict],
        world,
        now: float,
        voice_events: list[dict] | None = None,
    ) -> list[dict]:
        outputs: list[dict] = []
        voice_events = voice_events or []

        # Ensure every currently visible person has a state.
        for entity_id, person in world.people.items():
            if person.present and entity_id not in self.people:
                self.people[entity_id] = BehaviorState(
                    state="observing",
                    state_since=now,
                )

        # High-level semantic events have priority over lower-level events
        # from the same perception cycle. Example:
        # APPROACHING + ENTERED_AND_APPROACHING should become one direct
        # observing -> greeting transition, not observing -> attending
        # -> greeting.
        high_types_by_entity: dict[str, set[str]] = {}

        for event in high_level_events:
            entity_id = event.get("entity_id")

            if entity_id is not None:
                high_types_by_entity.setdefault(
                    entity_id,
                    set(),
                ).add(event["type"])

        # Low-level world events update engagement continuously.
        for event in low_level_events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            state = self.people.setdefault(
                entity_id,
                BehaviorState(
                    state="idle",
                    state_since=now,
                ),
            )

            event_type = event["type"]

            if event_type == "ENTERED":
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "observing",
                        now,
                        reason="ENTERED",
                    )
                )

            elif event_type == "LEFT":
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "idle",
                        now,
                        reason="LEFT",
                    )
                )

            elif event_type == "APPROACHING":
                # If EventEngine already recognized the stronger semantic
                # event in this same cycle, let that event own the transition.
                if (
                    "ENTERED_AND_APPROACHING"
                    in high_types_by_entity.get(entity_id, set())
                ):
                    continue

                if state.state in (
                    "observing",
                    "disengaging",
                    "settled",
                ):
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "attending",
                            now,
                            reason="APPROACHING",
                        )
                    )

                elif state.state == "attending":
                    outputs.extend(
                        self._refresh_attention(
                            entity_id,
                            state,
                            now,
                            reason="APPROACHING",
                        )
                    )

            elif event_type == "MOVING_AWAY":
                if state.state in (
                    "observing",
                    "attending",
                ):
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "disengaging",
                            now,
                            reason="MOVING_AWAY",
                        )
                    )

            elif event_type == "MOVEMENT_STOPPED":
                if state.state == "disengaging":
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "observing",
                            now,
                            reason="MOVEMENT_STOPPED",
                        )
                    )

            elif event_type == "STOOD_UP":
                if state.state == "settled":
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "attending",
                            now,
                            reason="STOOD_UP",
                        )
                    )

        # High-level semantic events can trigger intentional actions.
        for event in high_level_events:
            entity_id = event["entity_id"]

            state = self.people.setdefault(
                entity_id,
                BehaviorState(
                    state="observing",
                    state_since=now,
                ),
            )

            event_type = event["type"]

            if event_type == "ENTERED_AND_APPROACHING":
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "greeting",
                        now,
                        reason=event_type,
                    )
                )

                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="greet",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.greet_cooldown_s,
                    )
                )

            elif event_type == "GESTURE_WHILE_PRESENT":
                if state.state == "attending":
                    outputs.extend(
                        self._refresh_attention(
                            entity_id,
                            state,
                            now,
                            reason=event_type,
                        )
                    )

                elif state.state != "greeting":
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "attending",
                            now,
                            reason=event_type,
                        )
                    )

                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="acknowledge_gesture",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.gesture_action_cooldown_s,
                        gesture=event["gesture"],
                    )
                )

            elif event_type == "SAT_AND_STAYED":
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "settled",
                        now,
                        reason=event_type,
                    )
                )

                outputs.append({
                    "type": "ACTION",
                    "entity_id": entity_id,
                    "action": "settle_idle",
                    "reason": event_type,
                    "timestamp": now,
                })

        # Voice semantic events.
        for event in voice_events:
            entity_id = event.get("entity_id")

            if entity_id is None:
                continue

            state = self.people.setdefault(
                entity_id,
                BehaviorState(
                    state="observing",
                    state_since=now,
                ),
            )

            event_type = event["type"]

            if event_type == "VOICE_GREETING":
                if state.state == "attending":
                    outputs.extend(
                        self._refresh_attention(
                            entity_id,
                            state,
                            now,
                            reason=event_type,
                        )
                    )
                elif state.state != "greeting":
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "attending",
                            now,
                            reason=event_type,
                        )
                    )

                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="reply_greeting",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.voice_reply_cooldown_s,
                    )
                )

            elif event_type == "VOICE_CALL_ATTENTION":
                if state.state == "attending":
                    outputs.extend(
                        self._refresh_attention(
                            entity_id,
                            state,
                            now,
                            reason=event_type,
                        )
                    )
                elif state.state != "greeting":
                    outputs.extend(
                        self._transition(
                            entity_id,
                            state,
                            "attending",
                            now,
                            reason=event_type,
                        )
                    )

                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="acknowledge_call",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.voice_reply_cooldown_s,
                    )
                )

            elif event_type == "VOICE_SIT_COMMAND":
                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="sit_command",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.voice_reply_cooldown_s,
                    )
                )

            elif event_type == "VOICE_STOP_COMMAND":
                outputs.extend(
                    self._emit_action_if_allowed(
                        entity_id=entity_id,
                        state=state,
                        action="stop_command",
                        reason=event_type,
                        now=now,
                        cooldown_s=self.voice_reply_cooldown_s,
                    )
                )

        # Time-based state transitions.
        for entity_id, state in self.people.items():
            if (
                state.state == "greeting"
                and now - state.state_since >= self.greeting_duration_s
            ):
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "attending",
                        now,
                        reason="GREETING_FINISHED",
                    )
                )

            elif (
                state.state == "attending"
                and now - state.state_since >= self.attention_timeout_s
            ):
                outputs.extend(
                    self._transition(
                        entity_id,
                        state,
                        "observing",
                        now,
                        reason="ATTENTION_TIMEOUT",
                    )
                )

        return outputs

    def _emit_action_if_allowed(
        self,
        entity_id: str,
        state: BehaviorState,
        action: str,
        reason: str,
        now: float,
        cooldown_s: float,
        gesture: str | None = None,
    ) -> list[dict]:
        key = action if gesture is None else f"{action}:{gesture}"
        last_at = state.last_action_at.get(key)

        if last_at is not None and now - last_at < cooldown_s:
            return [{
                "type": "ACTION_SUPPRESSED",
                "entity_id": entity_id,
                "action": action,
                "gesture": gesture,
                "reason": reason,
                "cooldown_remaining_s": cooldown_s - (now - last_at),
                "timestamp": now,
            }]

        state.last_action_at[key] = now

        event = {
            "type": "ACTION",
            "entity_id": entity_id,
            "action": action,
            "reason": reason,
            "timestamp": now,
        }

        if gesture is not None:
            event["gesture"] = gesture

        return [event]

    def _refresh_attention(
        self,
        entity_id: str,
        state: BehaviorState,
        now: float,
        reason: str,
    ) -> list[dict]:
        if state.state != "attending":
            return []

        state.state_since = now

        return [{
            "type": "ATTENTION_REFRESHED",
            "entity_id": entity_id,
            "state": state.state,
            "reason": reason,
            "timestamp": now,
        }]

    def _transition(
        self,
        entity_id: str,
        state: BehaviorState,
        new_state: str,
        now: float,
        reason: str,
    ) -> list[dict]:
        if state.state == new_state:
            return []

        old_state = state.state
        state.state = new_state
        state.state_since = now

        return [{
            "type": "STATE_CHANGED",
            "entity_id": entity_id,
            "old_state": old_state,
            "state": new_state,
            "reason": reason,
            "timestamp": now,
        }]

    def get_state(self, entity_id: str) -> str:
        state = self.people.get(entity_id)

        if state is None:
            return "idle"

        return state.state
