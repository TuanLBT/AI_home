from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class ActionSpec:
    canonical: str
    executor_action: Optional[str]
    description: str


class ActionRegistry:
    """Hard boundary between free-form LLM text and executable actions."""

    def __init__(self):
        self._actions = {
            "WAIT": ActionSpec("WAIT", None, "Do nothing and continue observing."),
            "STOP": ActionSpec("STOP", "stop_command", "Stop the current robot action or movement."),
            "SIT": ActionSpec("SIT", "sit_command", "Execute the currently available sit command."),
            "GREET": ActionSpec("GREET", "greet", "Give the normal greeting."),
            "ACKNOWLEDGE_CALL": ActionSpec("ACKNOWLEDGE_CALL", "acknowledge_call", "Acknowledge that a person called for attention."),
            "ACKNOWLEDGE_GESTURE": ActionSpec("ACKNOWLEDGE_GESTURE", "acknowledge_gesture", "Acknowledge a recognized gesture."),
            "REPLY_GREETING": ActionSpec("REPLY_GREETING", "reply_greeting", "Reply to a greeting."),
            "SETTLE_IDLE": ActionSpec("SETTLE_IDLE", "settle_idle", "Remain settled and idle."),
        }

        self._aliases = {
            "NO_ACTION": "WAIT",
            "DO_NOTHING": "WAIT",
            "CONTINUE_OBSERVING": "WAIT",
            "STAY_IN_PLACE": "WAIT",
            "WAIT_FOR_COMMAND": "WAIT",
            "PAUSE": "WAIT",
            "STOP_MOVING": "STOP",
            "STOP_FOLLOWING": "STOP",
            "HALT": "STOP",
            "FREEZE": "STOP",
            "STOP_COMMAND": "STOP",
            "SIT_DOWN": "SIT",
            "SIT_COMMAND": "SIT",
            "GREET_PERSON": "GREET",
            "ACKNOWLEDGE": "ACKNOWLEDGE_CALL",
        }

    @staticmethod
    def _norm(action: object) -> str:
        return str(action or "").strip().upper()

    def resolve(self, action: object) -> Optional[ActionSpec]:
        name = self._norm(action)
        if not name:
            return None
        canonical = self._aliases.get(name, name)
        return self._actions.get(canonical)

    def is_supported(self, action: object) -> bool:
        return self.resolve(action) is not None

    def canonicalize(self, action: object) -> Optional[str]:
        spec = self.resolve(action)
        return spec.canonical if spec is not None else None

    def executor_action(self, action: object) -> Optional[str]:
        spec = self.resolve(action)
        return spec.executor_action if spec is not None else None

    def describe_for_brain(self) -> list[dict[str, str]]:
        return [
            {"name": spec.canonical, "description": spec.description}
            for spec in self._actions.values()
        ]


if __name__ == "__main__":
    registry = ActionRegistry()
    for action in ["STOP_MOVING", "WAIT", "MOVE_AROUND_OBSTACLE", "sit_command"]:
        spec = registry.resolve(action)
        print(action, "->", None if spec is None else {
            "canonical": spec.canonical,
            "executor_action": spec.executor_action,
        })
