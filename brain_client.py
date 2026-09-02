import json
from typing import Any, Dict
from urllib import request, error

from action_validator import ActionValidator


class BrainClient:
    """
    Thin client for the Indoor AI LLM brain.

    Flow:
        world_state
            -> Ollama/Qwen
            -> proposal {intent, action, confidence}
            -> ActionValidator
            -> final decision
    """

    SYSTEM_PROMPT = """You are the high-level decision module of an indoor home robot.

Infer what is happening from the structured world state and propose one high-level action.

Principles:
- Prefer safe, reversible actions when uncertain.
- Do not move merely because a person is nearby.
- Do not assume ambiguous words like "that", "there", or "it" refer to a specific object unless context resolves them.
- Explicit stop/cancel language overrides weaker nonverbal cues.
- Respect physical constraints such as close distance, low battery, authorization uncertainty, and uncertain speaker source.
- If the request cannot be inferred safely, ask for clarification.
- Do not directly control motors.
- Output only one JSON object:
  {"intent":"UPPER_SNAKE_CASE","action":"UPPER_SNAKE_CASE","confidence":0.0}
- Do not explain reasoning.
"""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3:8b",
        timeout: float = 10.0,
        validator: ActionValidator | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.validator = validator or ActionValidator()

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj

        raise ValueError(f"LLM did not return a valid JSON object: {text!r}")

    @staticmethod
    def _validate_proposal_schema(proposal: Dict[str, Any]) -> None:
        intent = proposal.get("intent")
        action = proposal.get("action")
        confidence = proposal.get("confidence")

        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("Invalid/missing intent")

        if not isinstance(action, str) or not action.strip():
            raise ValueError("Invalid/missing action")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("Invalid/missing confidence")

    def propose(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
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
                            world_state,
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
            "format": "json",
        }

        response = self._post_json(
            f"{self.host}/api/chat",
            payload,
        )

        content = (
            response
            .get("message", {})
            .get("content", "")
        )

        proposal = self._extract_json(content)
        self._validate_proposal_schema(proposal)

        return {
            "intent": proposal["intent"].strip().upper(),
            "action": proposal["action"].strip().upper(),
            "confidence": float(proposal["confidence"]),
        }

    def decide(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safe entry point.

        Never returns an LLM action without passing it through ActionValidator.
        """

        try:
            proposal = self.propose(world_state)

        except (error.URLError, TimeoutError) as exc:
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
    # Change this to the Legion IP if running from Dell.
    BRAIN_HOST = "http://192.168.128.120:11434"

    brain = BrainClient(
        host=BRAIN_HOST,
        model="qwen3:8b",
    )

    test_world_state = {
        "robot": {
            "state": "observing",
            "battery_percent": 75,
        },
        "person": {
            "id": "person_1",
            "distance_m": 1.5,
            "facing_robot": True,
            "motion": "raised_hand",
        },
        "events": [
            "PERSON_PRESENT",
            "HAND_RAISED",
        ],
        "speech": "ねえ",
    }

    decision = brain.decide(test_world_state)

    print(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2,
        )
    )
