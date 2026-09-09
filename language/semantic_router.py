from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string"},
        "target": {"type": ["string", "null"]},
        "need": {"type": "string"},
    },
    "required": ["source", "target", "need"],
    "additionalProperties": False,
}

ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "requests": {
            "type": "array",
            "maxItems": 4,
            "items": REQUEST_SCHEMA,
        },
        "correction": {
            "type": "object",
            "properties": {
                "applies_to_previous_turn": {"type": "boolean"},
                "corrected_requests": {
                    "type": "array",
                    "maxItems": 4,
                    "items": REQUEST_SCHEMA,
                },
                "lesson": {"type": "string"},
            },
            "required": [
                "applies_to_previous_turn",
                "corrected_requests",
                "lesson",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["requests", "correction"],
    "additionalProperties": False,
}


class SemanticObservationRouter:
    """Plan evidence requests and interpret human routing corrections.

    The router learns from data supplied at runtime. Human corrections are
    retrieved as examples and remain outside the source code, so new language
    and distinctions do not require adding keyword rules or editing prompts for
    every individual failure.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.128.120:11434",
        model: str | None = None,
        timeout_s: float = 12.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = (
            model
            or os.environ.get("INDOOR_AI_ROUTER_MODEL")
            or os.environ.get("INDOOR_AI_LLM_MODEL")
            or "qwen3:8b"
        )
        self.timeout_s = float(timeout_s)

    def route(
        self,
        text: str,
        *,
        previous_turn: dict[str, Any] | None = None,
        learned_examples: list[dict[str, Any]] | None = None,
    ) -> dict:
        previous_text = "none"
        if previous_turn:
            previous_text = json.dumps(
                {
                    "user_text": previous_turn.get("user_text"),
                    "route": previous_turn.get("route"),
                    "reply": previous_turn.get("reply"),
                },
                ensure_ascii=False,
            )

        examples_text = json.dumps(
            learned_examples or [],
            ensure_ascii=False,
        )

        prompt = (
            "/no_think\n"
            "You are the semantic observation planner for a general AI agent. "
            "Do not answer the user. Decide what fresh evidence, if any, is "
            "needed before another model can answer. Also decide whether the "
            "current utterance is human feedback correcting the previous turn's "
            "observation plan. Return only the schema JSON.\n\n"
            "Available evidence source names today:\n"
            "- screen: current desktop pixels/text/windows/monitors\n"
            "- visual: current camera/person/room scene\n"
            "- memory: stored or recent agent memory/history\n"
            "- audio: current/recent non-transcribed audio evidence\n"
            "- app_state: structured state from desktop applications\n"
            "- files: user files/documents\n"
            "Future source names may also appear. Choose sources by semantic "
            "need, not by literal keywords. If ordinary knowledge or available "
            "conversation is enough, requests must be empty.\n\n"
            "A request has source, semantic target, and need. Need can be any "
            "short semantic description such as current_activity, exact_text, "
            "general_state, error, history, location, identity, or current_value. "
            "Canonicalize a physical screen target to left/right/all when that "
            "is clearly what the user means.\n\n"
            "Correction rules:\n"
            "- Set correction.applies_to_previous_turn=true only when the user "
            "is correcting how the immediately previous request should have "
            "been interpreted or what evidence should have been used.\n"
            "- corrected_requests is the better observation plan for that prior "
            "user request, not for the correction sentence itself.\n"
            "- lesson is a concise semantic principle that can generalize to "
            "future differently-worded requests. Do not copy only the literal "
            "sentence as the lesson.\n"
            "- For a pure correction/teaching sentence, current requests should "
            "normally be empty unless the correction itself asks a new question.\n\n"
            "Human-verified learned corrections from earlier interactions are "
            "provided below. Treat them as examples to generalize from, not as "
            "hardcoded phrase matches. Resolve conflicts using the current "
            "utterance and the more specific semantic distinction.\n"
            f"Learned corrections: {examples_text}\n\n"
            f"Previous turn: {previous_text}\n\n"
            f"Current user utterance: {text}"
        )

        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": ROUTE_SCHEMA,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 220,
            },
            "keep_alive": "10m",
        }

        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"router Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach router Ollama at {self.base_url}"
            ) from exc

        content = str((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise RuntimeError("semantic router returned empty content")

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("semantic router returned malformed JSON") from exc

        if not isinstance(raw, dict):
            raise RuntimeError("semantic router response is not an object")

        correction = raw.get("correction")
        if not isinstance(correction, dict):
            correction = {}

        return {
            "model": self.model,
            "requests": self._clean_requests(raw.get("requests")),
            "correction": {
                "applies_to_previous_turn": bool(
                    correction.get("applies_to_previous_turn")
                ),
                "corrected_requests": self._clean_requests(
                    correction.get("corrected_requests")
                ),
                "lesson": str(correction.get("lesson") or "").strip(),
            },
        }

    @staticmethod
    def _clean_requests(value: Any) -> list[dict]:
        if not isinstance(value, list):
            return []

        cleaned: list[dict] = []
        for item in value[:4]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip().casefold()
            if not source:
                continue
            target_value = item.get("target")
            target = (
                str(target_value).strip().casefold()
                if target_value is not None and str(target_value).strip()
                else None
            )
            need = str(item.get("need") or "general_state").strip().casefold()
            cleaned.append(
                {
                    "source": source,
                    "target": target,
                    "need": need or "general_state",
                }
            )
        return cleaned
