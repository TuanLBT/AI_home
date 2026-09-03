from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DialogueSession:
    active: bool = False
    started_at: float = 0.0
    last_activity_at: float = 0.0
    last_user_text: str | None = None
    last_intent: str | None = None
    turn_count: int = 0


class DialogueStateManager:
    """
    Short-lived dialogue/session state per person.

    This is not an LLM and does not generate replies.
    It only tracks whether a person is currently in an active conversation.

    A dialogue becomes active when an assigned voice event is received.
    It expires after inactivity_timeout_s.

    UNKNOWN ASR text can be kept as VOICE_UTTERANCE only while a dialogue
    is already active. This gives the later LLM layer conversational context
    without treating every random sound as a command.
    """

    def __init__(
        self,
        inactivity_timeout_s: float = 15.0,
    ):
        self.inactivity_timeout_s = inactivity_timeout_s
        self.sessions: dict[str, DialogueSession] = {}

    def observe_voice_event(
        self,
        event: dict,
        now: float,
    ) -> list[dict]:
        entity_id = event.get("entity_id")

        if entity_id is None:
            return []

        session = self.sessions.setdefault(
            entity_id,
            DialogueSession(),
        )

        outputs: list[dict] = []

        if not session.active:
            session.active = True
            session.started_at = now

            outputs.append({
                "type": "DIALOGUE_STARTED",
                "entity_id": entity_id,
                "timestamp": now,
            })

        session.last_activity_at = now
        session.last_user_text = event.get("text")
        session.last_intent = event.get("intent")
        session.turn_count += 1

        return outputs

    def handle_unknown(
        self,
        intent,
        world,
        behavior_engine,
        now: float,
    ) -> list[dict]:
        entity_id = self._resolve_active_speaker(
            world,
            behavior_engine,
        )

        if entity_id is None:
            return []

        session = self.sessions.get(entity_id)

        if session is None or not session.active:
            return []

        session.last_activity_at = now
        session.last_user_text = intent.text
        session.last_intent = "UNKNOWN"
        session.turn_count += 1

        return [{
            "type": "VOICE_UTTERANCE",
            "entity_id": entity_id,
            "intent": "UNKNOWN",
            "text": intent.text,
            "confidence": intent.confidence,
            "timestamp": now,
        }]

    def update(
        self,
        now: float,
    ) -> list[dict]:
        outputs: list[dict] = []

        for entity_id, session in self.sessions.items():
            if not session.active:
                continue

            if now - session.last_activity_at >= self.inactivity_timeout_s:
                session.active = False

                outputs.append({
                    "type": "DIALOGUE_ENDED",
                    "entity_id": entity_id,
                    "timestamp": now,
                    "turn_count": session.turn_count,
                })

        return outputs

    def snapshot(
        self,
        entity_id: str,
        now: float,
    ) -> dict:
        session = self.sessions.get(entity_id)

        if session is None:
            return {
                "active": False,
                "turn_count": 0,
                "last_user_text": None,
                "last_intent": None,
                "age_since_activity_s": None,
            }

        return {
            "active": session.active,
            "turn_count": session.turn_count,
            "last_user_text": session.last_user_text,
            "last_intent": session.last_intent,
            "age_since_activity_s": (
                max(0.0, now - session.last_activity_at)
                if session.last_activity_at
                else None
            ),
        }

    def _resolve_active_speaker(
        self,
        world,
        behavior_engine,
    ) -> str | None:
        present = [
            entity_id
            for entity_id, person in world.people.items()
            if person.present
        ]

        active_present = [
            entity_id
            for entity_id in present
            if self.sessions.get(entity_id)
            and self.sessions[entity_id].active
        ]

        if len(active_present) == 1:
            return active_present[0]

        if len(present) == 1:
            return present[0]

        engaged = [
            entity_id
            for entity_id in present
            if behavior_engine.get_state(entity_id)
            in ("greeting", "attending")
        ]

        if len(engaged) == 1:
            return engaged[0]

        return None
