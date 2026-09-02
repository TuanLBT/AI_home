import json
from typing import Any, Dict

import requests

from action.action_validator import ActionValidator


class BrainClient:
    """
    Thin client for the Indoor AI high-level decision brain.

    Flow:
        structured world state
            -> Ollama/Qwen
            -> semantic proposal {intent, action, confidence}
            -> deterministic ActionValidator
            -> final decision

    Feedback from the previous executed action is included inside world_state.
    The LLM may use that feedback for replanning, but it never bypasses
    ActionValidator.
    """

    SYSTEM_PROMPT = """You are the high-level decision module of an indoor home robot.

Infer what is happening from the structured world state and propose one high-level action.

The world state may include a "feedback" object describing the previously executed action.

Feedback rules:
- Treat feedback as the actual result of the previous control/execution step.
- status=FAILED means the previous action did not execute successfully.
- status=SUCCESS means the action itself executed, but this does not necessarily mean the higher-level goal was achieved.
- goal_reached=true means the active goal is already satisfied. Do not continue acting toward the same goal.
- goal_reached=false means the goal is still not achieved.
- If an action FAILED, do not blindly repeat the exact same action unless the current world state has materially changed or the failure reason is clearly transient.
- If the failure reason indicates an unavailable or unimplemented skill, choose another safe strategy or ask for clarification instead of retrying the unavailable skill.
- If the failure reason indicates a physical obstacle or unsafe condition, prefer a different safe strategy, waiting, or clarification.
- Never assume an action succeeded merely because it was requested.

General principles:
- Prefer safe, reversible actions when uncertain.
- Do not move merely because a person is nearby.
- Do not assume ambiguous words like "that", "there", or "it" refer to a specific object unless context resolves them.
- Explicit stop/cancel language overrides weaker nonverbal cues.
- Respect physical constraints such as close distance, low battery, authorization uncertainty, and uncertain speaker source.
- If the request cannot be inferred safely, ask for clarification.
- Do not directly control motors.
- Propose only one high-level action.
- Output only one JSON object matching the required schema.
- Do not explain reasoning.
"""

    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
            },
            "action": {
                "type": "string",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "intent",
            "action",
            "confidence",
        ],
        "additionalProperties": False,
    }

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        timeout: float = 30.0,
        validator: ActionValidator | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.validator = validator or ActionValidator()

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    @staticmethod
    def _validate_proposal_schema(
        proposal: Dict[str, Any],
    ) -> None:
        intent = proposal.get("intent")
        action = proposal.get("action")
        confidence = proposal.get("confidence")

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError(
                "Invalid/missing intent"
            )

        if not isinstance(action, str) or not action.strip():
            raise ValueError(
                "Invalid/missing action"
            )

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(
                "Invalid/missing confidence"
            )

    @staticmethod
    def _normalize_feedback(
        world_state: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        feedback = world_state.get("feedback")

        if not isinstance(feedback, dict):
            return None

        return {
            "request_id": feedback.get("request_id"),
            "action": feedback.get("action"),
            "command": feedback.get("command"),
            "status": feedback.get("status"),
            "reason": feedback.get("reason"),
            "actual": feedback.get("actual"),
            "observations": feedback.get("observations"),
            "goal_id": feedback.get("goal_id"),
            "goal_reached": feedback.get("goal_reached"),
        }

    def _prepare_world_state(
        self,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepared = dict(world_state)

        feedback = self._normalize_feedback(
            world_state
        )

        if feedback is None:
            prepared.pop(
                "feedback",
                None,
            )
        else:
            prepared["feedback"] = feedback

        return prepared

    @staticmethod
    def _same_failed_action(
        world_state: Dict[str, Any],
        proposal: Dict[str, Any],
    ) -> bool:
        feedback = world_state.get("feedback")

        if not isinstance(feedback, dict):
            return False

        previous_status = str(
            feedback.get("status") or ""
        ).strip().upper()

        previous_action = str(
            feedback.get("action") or ""
        ).strip().upper()

        proposed_action = str(
            proposal.get("action") or ""
        ).strip().upper()

        return (
            previous_status == "FAILED"
            and previous_action
            and previous_action == proposed_action
        )

    @staticmethod
    def _goal_already_reached(
        world_state: Dict[str, Any],
    ) -> bool:
        feedback = world_state.get("feedback")

        if not isinstance(feedback, dict):
            return False

        return feedback.get("goal_reached") is True

    def propose(
        self,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepared_state = self._prepare_world_state(
            world_state
        )

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        "Current world state:\n"
                        + json.dumps(
                            prepared_state,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                },
            ],
            "options": {
                "temperature": 0,
                "num_predict": 80,
            },
            "format": self.RESPONSE_SCHEMA,
        }

        response = self._post_json(
            f"{self.host}/api/chat",
            payload,
        )

        content = (
            response
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            raise ValueError(
                "LLM returned empty content"
            )

        proposal = json.loads(
            content
        )

        if not isinstance(proposal, dict):
            raise ValueError(
                "LLM output is not a JSON object"
            )

        self._validate_proposal_schema(
            proposal
        )

        return {
            "intent": proposal["intent"].strip().upper(),
            "action": proposal["action"].strip().upper(),
            "confidence": float(
                proposal["confidence"]
            ),
        }

    def decide(
        self,
        world_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self._goal_already_reached(
            world_state
        ):
            return {
                "brain_ok": True,
                "proposal": None,
                "approved": False,
                "final_action": "WAIT",
                "reason": "GOAL_ALREADY_REACHED",
                "violations": [],
            }

        try:
            proposal = self.propose(
                world_state
            )

        except requests.Timeout as exc:
            return {
                "brain_ok": False,
                "proposal": None,
                "approved": False,
                "final_action": "WAIT",
                "reason": "BRAIN_TIMEOUT",
                "error": str(exc),
            }

        except requests.RequestException as exc:
            return {
                "brain_ok": False,
                "proposal": None,
                "approved": False,
                "final_action": "WAIT",
                "reason": "BRAIN_UNAVAILABLE",
                "error": str(exc),
            }

        except Exception as exc:
            return {
                "brain_ok": False,
                "proposal": None,
                "approved": False,
                "final_action": "WAIT",
                "reason": "INVALID_BRAIN_OUTPUT",
                "error": str(exc),
            }

        if self._same_failed_action(
            world_state,
            proposal,
        ):
            return {
                "brain_ok": True,
                "proposal": proposal,
                "approved": False,
                "final_action": "WAIT",
                "reason": "REPEATED_FAILED_ACTION",
                "violations": [
                    "REPEATED_FAILED_ACTION",
                ],
            }

        validation = self.validator.validate(
            world_state=world_state,
            proposal=proposal,
        )

        return {
            "brain_ok": True,
            "proposal": proposal,
            "approved": validation.approved,
            "final_action": validation.final_action,
            "reason": validation.reason,
            "violations": validation.violations,
        }


if __name__ == "__main__":
    BRAIN_HOST = "http://192.168.128.120:11434"

    brain = BrainClient(
        host=BRAIN_HOST,
        model="qwen3:8b",
        timeout=30.0,
    )

    test_world_state = {
        "robot": {
            "state": "attending",
            "battery_percent": 75,
        },
        "person": {
            "id": "person_1",
            "distance_m": 1.5,
            "facing_robot": True,
            "motion": "stationary",
        },
        "speech": "こっち来て",
        "feedback": {
            "request_id": "example-request",
            "action": "MOVE_TOWARD_PERSON",
            "command": "move_toward_person",
            "status": "FAILED",
            "reason": "OBSTACLE_DETECTED",
            "actual": {
                "moved_distance_m": 0.0,
            },
            "observations": {
                "obstacle_ahead": True,
            },
            "goal_id": "example-goal",
            "goal_reached": False,
        },
    }

    decision = brain.decide(
        test_world_state
    )

    print(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        )
    )
