
from __future__ import annotations


class VoiceEventEngine:
    """
    Converts parsed ASR intents into semantic voice events.

    It also associates the voice with a visible person.

    Association policy for the current single-camera prototype:
      1. If exactly one person is present -> use that person.
      2. If multiple are present -> prefer a person currently in
         greeting/attending state.
      3. Otherwise -> leave the voice event unassigned.

    Later this association can be replaced by speaker localization / diarization.
    """

    SUPPORTED_INTENTS = {
        "GREETING": "VOICE_GREETING",
        "CALL_ATTENTION": "VOICE_CALL_ATTENTION",
        "SIT_COMMAND": "VOICE_SIT_COMMAND",
        "STOP_COMMAND": "VOICE_STOP_COMMAND",
    }

    def process(
        self,
        intent,
        world,
        behavior_engine,
        now: float,
    ) -> list[dict]:
        event_type = self.SUPPORTED_INTENTS.get(intent.type)

        if event_type is None:
            return []

        entity_id = self._resolve_speaker(
            world,
            behavior_engine,
        )

        if entity_id is None:
            return [{
                "type": "VOICE_UNASSIGNED",
                "entity_id": None,
                "intent": intent.type,
                "text": intent.text,
                "confidence": intent.confidence,
                "timestamp": now,
            }]

        return [{
            "type": event_type,
            "entity_id": entity_id,
            "intent": intent.type,
            "text": intent.text,
            "confidence": intent.confidence,
            "timestamp": now,
        }]

    @staticmethod
    def _resolve_speaker(
        world,
        behavior_engine,
    ) -> str | None:
        present = [
            entity_id
            for entity_id, person in world.people.items()
            if person.present
        ]

        if len(present) == 1:
            return present[0]

        if len(present) > 1:
            engaged = [
                entity_id
                for entity_id in present
                if behavior_engine.get_state(entity_id)
                in ("greeting", "attending")
            ]

            if len(engaged) == 1:
                return engaged[0]

        return None
