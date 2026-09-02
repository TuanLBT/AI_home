from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from speech.speech_adapter import SpeechAdapter
from speech.speech_policy import SpeechPolicy


@dataclass(slots=True)
class QueuedAction:
    entity_id: str
    action: str
    reason: str
    timestamp: float
    gesture: str | None = None
    speech_text: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    """
    Executes actions produced by BehaviorEngine.

    BehaviorEngine decides WHAT action is wanted.
    ActionExecutor decides HOW to execute it.
    SpeechPolicy chooses WHAT to say from the action + context snapshot.
    SpeechAdapter handles the actual TTS backend.
    """

    def __init__(
        self,
        speech: SpeechAdapter | None = None,
        speech_policy: SpeechPolicy | None = None,
    ):
        self.queue: deque[QueuedAction] = deque()

        self.speech_policy = speech_policy or SpeechPolicy(
            language="ja",
            randomize=True,
        )

        self.speech = speech or SpeechAdapter(
            language=self.speech_policy.language,
        )

    def submit(
        self,
        event: dict,
        context: dict[str, Any] | None = None,
    ) -> None:
        if event.get("type") != "ACTION":
            return

        self.queue.append(
            QueuedAction(
                entity_id=event["entity_id"],
                action=event["action"],
                reason=event["reason"],
                timestamp=event["timestamp"],
                gesture=event.get("gesture"),
                speech_text=event.get("speech_text"),
                context=dict(context or {}),
            )
        )

    def update(self) -> list[dict]:
        executed: list[dict] = []

        while self.queue:
            queued = self.queue.popleft()

            result = self._execute(queued)

            if result is not None:
                executed.append(result)

        return executed

    def _execute(self, queued: QueuedAction) -> dict | None:
        command = queued.action

        if queued.action == "greet":
            command = "say_greeting"

        elif queued.action == "acknowledge_gesture":
            command = "acknowledge_gesture"

        elif queued.action == "settle_idle":
            command = "settle_idle"

        elif queued.action == "reply_greeting":
            command = "say_reply_greeting"

        elif queued.action == "acknowledge_call":
            command = "say_acknowledge_call"

        elif queued.action == "sit_command":
            command = "sit_command"

        elif queued.action == "stop_command":
            command = "stop_command"

        elif queued.action == "speak_text":
            command = "say_llm_reply"

        speech_text = queued.speech_text

        if speech_text is None:
            speech_text = self.speech_policy.get_text(
                action=queued.action,
                gesture=queued.gesture,
                context=queued.context,
            )

        speech_result = None

        if speech_text is not None:
            speech_result = self.speech.speak(
                speech_text
            )

        event = {
            "type": "ACTION_EXECUTED",
            "entity_id": queued.entity_id,
            "action": queued.action,
            "command": command,
            "gesture": queued.gesture,
            "reason": queued.reason,
            "timestamp": queued.timestamp,
            "context": queued.context,
        }

        if speech_result is not None:
            event["speech_text"] = speech_result.text
            event["speech_ok"] = speech_result.ok
            event["speech_backend"] = speech_result.backend
            event["speech_error"] = speech_result.error

        return event
