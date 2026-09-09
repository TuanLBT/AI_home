from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "requests": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": ["string", "null"]},
                    "need": {"type": "string"},
                },
                "required": ["source", "target", "need"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requests"],
    "additionalProperties": False,
}


class SemanticObservationRouter:
    """Plan which observations are needed before answering a user turn.

    This router is intentionally source-agnostic. It does not answer the user
    and it does not capture anything itself; it only produces observation
    requests. The executor can support more sources later without changing the
    text source or adding more keyword heuristics.
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

    def route(self, text: str) -> dict:
        prompt = (
            "/no_think\n"
            "You are an observation router for an indoor/desktop AI agent. "
            "Do NOT answer the user. Decide which fresh evidence sources are "
            "needed before another model can answer. Return only the requested "
            "JSON. If the utterance is answerable from ordinary knowledge or "
            "conversation alone, return an empty requests array.\n\n"
            "Known source names and meanings:\n"
            "- screen: current desktop pixels/text/windows/monitors\n"
            "- visual: current camera/person/room scene\n"
            "- memory: stored or recent agent memory/history\n"
            "- audio: current/recent non-transcribed audio evidence\n"
            "- app_state: structured state from desktop applications\n"
            "- files: user files/documents\n\n"
            "For each request, source is one source name; target is a short "
            "semantic target or null; need is a short evidence need. For screen "
            "monitor targets, canonicalize physical monitor selection to left, "
            "right, or all. Examples of need: general_state, exact_text, error, "
            "current_value, history. Request only evidence actually needed.\n\n"
            f"User utterance: {text}"
        )

        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": ROUTE_SCHEMA,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0.0,
                "num_predict": 128,
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

        requests = raw.get("requests") if isinstance(raw, dict) else None
        if not isinstance(requests, list):
            raise RuntimeError("semantic router response has no requests array")

        cleaned: list[dict] = []
        for item in requests[:4]:
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

        return {
            "model": self.model,
            "requests": cleaned,
        }
