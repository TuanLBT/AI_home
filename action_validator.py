from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    approved: bool
    requested_action: str
    final_action: str
    reason: str
    violations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionValidator:
    """
    Hard safety gate for LLM-proposed robot actions.

    The LLM proposes.
    This validator decides whether that proposal may pass through.

    Expected world_state examples:
    {
        "robot": {
            "state": "attending",
            "battery_percent": 72
        },
        "person": {
            "id": "person_1",
            "distance_m": 1.4
        },
        "audio_context": {
            "speaker_confidence": 0.95
        },
        "speech": "こっち来て"
    }

    Expected proposal:
    {
        "intent": "COME_HERE",
        "action": "MOVE_TOWARD_PERSON",
        "confidence": 0.91
    }
    """

    # Canonical actions that imply physical movement toward / with a person.
    PERSON_APPROACH_ACTIONS = {
        "MOVE_CLOSER",
        "MOVE_TOWARD",
        "MOVE_TOWARD_PERSON",
        "APPROACH_PERSON",
        "APPROACH_USER",
        "MOVE_TO_PERSON",
    }

    FOLLOW_ACTIONS = {
        "FOLLOW_PERSON",
        "FOLLOW_USER",
        "MOVE_WITH_PERSON",
    }

    GENERAL_MOVEMENT_ACTIONS = PERSON_APPROACH_ACTIONS | FOLLOW_ACTIONS | {
        "MOVE_ASIDE",
        "STEP_BACK",
        "MOVE_OUT_OF_WAY",
        "BACK_AWAY",
    }

    STOP_ACTIONS = {
        "STOP",
        "STOP_MOVING",
        "STOP_FOLLOWING",
        "HALT",
    }

    CLARIFY_ACTIONS = {
        "ASK_CLARIFICATION",
        "ASK_FOR_CLARIFICATION",
        "REQUEST_CLARIFICATION",
        "ASK_USER",
        "ASK_WHAT",
        "CLARIFY",
    }

    WAIT_ACTIONS = {
        "WAIT",
        "NO_ACTION",
        "CONTINUE_OBSERVING",
        "STAY_IN_PLACE",
        "DO_NOTHING",
        "IGNORE",
        "WAIT_FOR_COMMAND",
    }

    PRIVILEGED_ACTION_KEYWORDS = {
        "OPEN_DOOR",
        "UNLOCK",
        "LOCK_DOOR",
        "DISABLE_ALARM",
        "ENABLE_ALARM",
    }

    def __init__(
        self,
        min_person_distance_m: float = 0.8,
        critical_battery_percent: float = 5.0,
        min_speaker_confidence_for_movement: float = 0.4,
    ):
        self.min_person_distance_m = min_person_distance_m
        self.critical_battery_percent = critical_battery_percent
        self.min_speaker_confidence_for_movement = min_speaker_confidence_for_movement

    @staticmethod
    def _norm(value: Any) -> str:
        return str(value or "").strip().upper()

    def _is_stop_requested(self, world_state: Dict[str, Any]) -> bool:
        speech = str(world_state.get("speech") or "")
        speech_upper = speech.upper()

        stop_markers = (
            "STOP",
            "HALT",
            "止まって",
            "止まれ",
            "止めて",
            "とまって",
            "とまれ",
        )
        return any(marker in speech_upper or marker in speech for marker in stop_markers)

    def _is_privileged_action(self, action: str) -> bool:
        action = self._norm(action)
        return any(keyword in action for keyword in self.PRIVILEGED_ACTION_KEYWORDS)

    def _is_movement_action(self, action: str) -> bool:
        action = self._norm(action)
        if action in self.GENERAL_MOVEMENT_ACTIONS:
            return True

        # Conservative fallback for model-invented movement labels.
        movement_tokens = (
            "MOVE",
            "APPROACH",
            "FOLLOW",
            "STEP",
            "DRIVE",
            "GO_TO_PERSON",
        )
        return any(token in action for token in movement_tokens)

    def _is_person_approach_action(self, action: str) -> bool:
        action = self._norm(action)
        if action in self.PERSON_APPROACH_ACTIONS:
            return True

        approach_tokens = (
            "MOVE_TOWARD_PERSON",
            "MOVE_TO_PERSON",
            "APPROACH_PERSON",
            "APPROACH_USER",
            "MOVE_CLOSER",
        )
        return any(token in action for token in approach_tokens)

    def validate(
        self,
        world_state: Dict[str, Any],
        proposal: Dict[str, Any],
    ) -> ValidationResult:
        requested_action = self._norm(proposal.get("action"))
        violations: List[str] = []

        # 1) Explicit STOP always overrides the LLM.
        if self._is_stop_requested(world_state):
            if requested_action not in self.STOP_ACTIONS:
                violations.append("EXPLICIT_STOP_OVERRIDE")
            return ValidationResult(
                approved=requested_action in self.STOP_ACTIONS,
                requested_action=requested_action,
                final_action="STOP",
                reason="USER_REQUESTED_STOP",
                violations=violations,
            )

        robot = world_state.get("robot") or {}
        person = world_state.get("person") or {}
        audio_context = world_state.get("audio_context") or {}

        person_id = person.get("id")
        distance_m = person.get("distance_m")
        battery = robot.get("battery_percent")
        speaker_confidence = audio_context.get("speaker_confidence")

        # 2) Unknown people cannot trigger privileged actions.
        if person_id == "unknown_person" and self._is_privileged_action(requested_action):
            violations.append("UNAUTHORIZED_PERSON")
            return ValidationResult(
                approved=False,
                requested_action=requested_action,
                final_action="ASK_CLARIFICATION",
                reason="UNAUTHORIZED_PERSON",
                violations=violations,
            )

        # 3) Too close -> reject movement toward the person.
        if isinstance(distance_m, (int, float)):
            if distance_m < self.min_person_distance_m and self._is_person_approach_action(requested_action):
                violations.append("PERSON_TOO_CLOSE")
                return ValidationResult(
                    approved=False,
                    requested_action=requested_action,
                    final_action="WAIT",
                    reason="PERSON_TOO_CLOSE",
                    violations=violations,
                )

        # 4) Critical battery -> reject nonessential movement.
        if isinstance(battery, (int, float)):
            if battery <= self.critical_battery_percent and self._is_movement_action(requested_action):
                violations.append("CRITICAL_BATTERY")
                return ValidationResult(
                    approved=False,
                    requested_action=requested_action,
                    final_action="WAIT",
                    reason="CRITICAL_BATTERY",
                    violations=violations,
                )

        # 5) Weak speaker attribution -> do not move based on speech.
        if isinstance(speaker_confidence, (int, float)):
            if (
                speaker_confidence < self.min_speaker_confidence_for_movement
                and self._is_movement_action(requested_action)
            ):
                violations.append("UNCERTAIN_SPEAKER")
                return ValidationResult(
                    approved=False,
                    requested_action=requested_action,
                    final_action="ASK_CLARIFICATION",
                    reason="UNCERTAIN_SPEAKER",
                    violations=violations,
                )

        # Proposal passes current hard rules.
        return ValidationResult(
            approved=True,
            requested_action=requested_action,
            final_action=requested_action,
            reason="APPROVED",
            violations=violations,
        )


if __name__ == "__main__":
    # Small smoke test.
    validator = ActionValidator()

    tests = [
        (
            {
                "robot": {"battery_percent": 72},
                "person": {"id": "person_1", "distance_m": 0.45},
                "speech": "もっとこっち",
            },
            {"intent": "COME_HERE", "action": "MOVE_TOWARD_PERSON", "confidence": 0.9},
        ),
        (
            {
                "robot": {"battery_percent": 80},
                "person": {"id": "unknown_person", "distance_m": 1.5},
                "speech": "玄関の鍵を開けて",
            },
            {"intent": "OPEN_DOOR", "action": "OPEN_DOOR", "confidence": 0.95},
        ),
        (
            {
                "robot": {"battery_percent": 60},
                "person": {"id": "person_1", "distance_m": 2.0},
                "speech": "止まって",
            },
            {"intent": "FOLLOW_ME", "action": "FOLLOW_PERSON", "confidence": 0.8},
        ),
    ]

    for i, (state, proposal) in enumerate(tests, 1):
        result = validator.validate(state, proposal)
        print(f"TEST {i}: {result.to_dict()}")
