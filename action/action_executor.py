from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import uuid

from action.action_protocol import ActionRequest, ActionResult, ActionStatus
from speech.speech_adapter import SpeechAdapter
from speech.speech_policy import SpeechPolicy


@dataclass(slots=True)
class QueuedAction:
    request: ActionRequest
    gesture: str | None = None
    speech_text: str | None = None


class ActionExecutor:
    """
    Executes high-level actions.

    Behavior/Brain decides WHAT action is wanted.
    ActionExecutor decides HOW to execute it.

    update() returns ACTION_EXECUTED-compatible dictionaries so main.py and
    InteractionMemory keep working, but those dictionaries now also contain
    standardized closed-loop result fields such as request_id and status.
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

        request = ActionRequest(
            action=event["action"],
            entity_id=event["entity_id"],
            reason=event["reason"],
            timestamp=event["timestamp"],
            intent=event.get("intent"),
            confidence=event.get("confidence"),
            target_id=event.get("target_id", event.get("entity_id")),
            parameters=dict(event.get("parameters") or {}),
            context=dict(context or {}),
            goal_id=event.get("goal_id"),
            source=event.get("source", "behavior"),
            request_id=event.get("request_id") or str(uuid.uuid4()),
        )

        self.queue.append(
            QueuedAction(
                request=request,
                gesture=event.get("gesture"),
                speech_text=event.get("speech_text"),
            )
        )

    def update(self) -> list[dict]:
        executed: list[dict] = []

        while self.queue:
            queued = self.queue.popleft()
            result_event = self._execute(queued)

            if result_event is not None:
                executed.append(result_event)

        return executed

    def _execute(self, queued: QueuedAction) -> dict | None:
        request = queued.request
        command = request.action

        if request.action == "greet":
            command = "say_greeting"
        elif request.action == "acknowledge_gesture":
            command = "acknowledge_gesture"
        elif request.action == "settle_idle":
            command = "settle_idle"
        elif request.action == "reply_greeting":
            command = "say_reply_greeting"
        elif request.action == "acknowledge_call":
            command = "say_acknowledge_call"
        elif request.action == "sit_command":
            command = "sit_command"
        elif request.action == "stop_command":
            command = "stop_command"
        elif request.action == "speak_text":
            command = "say_llm_reply"

        speech_text = queued.speech_text

        if speech_text is None:
            speech_text = self.speech_policy.get_text(
                action=request.action,
                gesture=queued.gesture,
                context=request.context,
            )

        speech_result = None

        if speech_text is not None:
            speech_result = self.speech.speak(speech_text)

        status = ActionStatus.SUCCESS
        failure_reason = None

        if speech_result is not None and not speech_result.ok:
            status = ActionStatus.FAILED
            failure_reason = speech_result.error or "SPEECH_FAILED"

        result = ActionResult(
            request_id=request.request_id,
            entity_id=request.entity_id,
            action=request.action,
            command=command,
            status=status,
            reason=failure_reason or request.reason,
            timestamp=request.timestamp,
            actual={
                "command": command,
                "speech_attempted": speech_result is not None,
            },
            observations={
                "gesture": queued.gesture,
            },
            goal_id=request.goal_id,
            goal_reached=None,
        )

        event = {
            "type": "ACTION_EXECUTED",
            "entity_id": request.entity_id,
            "action": request.action,
            "command": command,
            "gesture": queued.gesture,
            "reason": request.reason,
            "timestamp": request.timestamp,
            "context": request.context,

            # Closed-loop result fields
            "request_id": result.request_id,
            "status": result.status.value,
            "actual": result.actual,
            "observations": result.observations,
            "goal_id": result.goal_id,
            "goal_reached": result.goal_reached,
            "finished_at": result.finished_at,
        }

        if speech_result is not None:
            event["speech_text"] = speech_result.text
            event["speech_ok"] = speech_result.ok
            event["speech_backend"] = speech_result.backend
            event["speech_error"] = speech_result.error

        return event
