from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

from sources.base import SourcePacket
from sources.screen_bus import publish_screen_representation


class ScreenVisionPerception:
    """Turn a raw desktop screenshot into a neutral structured representation.

    The backend is an Ollama VLM, but the rest of the agent only consumes the
    published representation. A future OCR/UI-tree/local model can replace or
    augment this adapter without changing ChatSource or LLMWorker.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.128.120:11434",
        model: str | None = None,
        timeout_s: float = 90.0,
        max_width: int = 2560,
        jpeg_quality: int = 82,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = (
            model
            or os.environ.get("INDOOR_AI_VISION_MODEL")
            or "qwen3-vl:4b"
        )
        self.timeout_s = float(timeout_s)
        self.max_width = max(640, int(max_width))
        self.jpeg_quality = max(40, min(95, int(jpeg_quality)))

    def metadata(self) -> dict:
        return {
            "backend": "ollama_vlm",
            "model": self.model,
            "max_width": self.max_width,
            "representation": "structured_screen_state_v1",
        }

    def describe(self, packet: SourcePacket) -> dict:
        payload = packet.payload if isinstance(packet.payload, dict) else {}
        frame = payload.get("frame")

        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            result = {
                "available": False,
                "timestamp": packet.timestamp,
                "backend": "ollama_vlm",
                "model": self.model,
                "error": "screen packet has no RGB frame",
            }
            publish_screen_representation(result)
            return result

        try:
            encoded, width, height = self._encode_frame(frame)
            state = self._ask_vlm(encoded)
            result = {
                "available": True,
                "timestamp": packet.timestamp,
                "backend": "ollama_vlm",
                "model": self.model,
                "schema": "structured_screen_state_v1",
                "active_app": state.get("active_app"),
                "windows": state.get("windows", []),
                "visible_text": state.get("visible_text", []),
                "errors": state.get("errors", []),
                "dialogs": state.get("dialogs", []),
                "ui_state": state.get("ui_state", []),
                "description": state.get("description", ""),
                "input_width": width,
                "input_height": height,
            }
        except Exception as exc:
            result = {
                "available": False,
                "timestamp": packet.timestamp,
                "backend": "ollama_vlm",
                "model": self.model,
                "error": str(exc),
            }

        publish_screen_representation(result)
        return result

    def _encode_frame(self, frame: np.ndarray) -> tuple[str, int, int]:
        rgb = frame
        height, width = rgb.shape[:2]

        if width > self.max_width:
            scale = self.max_width / float(width)
            target = (
                self.max_width,
                max(1, int(round(height * scale))),
            )
            rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

        # ScreenSource frames are RGB; OpenCV JPEG expects BGR.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode screen frame")

        out_h, out_w = rgb.shape[:2]
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return image_b64, int(out_w), int(out_h)

    def _ask_vlm(self, image_b64: str) -> dict:
        prompt = (
            "Analyze this current desktop screenshot as evidence for another agent. "
            "Return ONLY one JSON object, no markdown and no explanation. "
            "Use exactly these keys:\n"
            "{\n"
            '  "active_app": string or null,\n'
            '  "windows": [string],\n'
            '  "visible_text": [string],\n'
            '  "errors": [string],\n'
            '  "dialogs": [string],\n'
            '  "ui_state": [string],\n'
            '  "description": string\n'
            "}\n"
            "Only include details visibly supported by the screenshot. "
            "Keep each list short and useful. Preserve important readable error text when visible. "
            "Do not infer hidden content or user intent. The description should be a compact summary.\n"
            "/no_think"
        )

        data = self._request_vlm(
            image_b64=image_b64,
            prompt=prompt,
            num_predict=768,
        )

        message = data.get("message") or {}
        text = str(message.get("content") or "").strip()

        if not text:
            thinking = str(message.get("thinking") or "").strip()
            if thinking:
                data = self._request_vlm(
                    image_b64=image_b64,
                    prompt=prompt,
                    num_predict=1536,
                )
                message = data.get("message") or {}
                text = str(message.get("content") or "").strip()

        if not text:
            done_reason = data.get("done_reason")
            thinking_len = len(
                str((data.get("message") or {}).get("thinking") or "")
            )
            raise RuntimeError(
                "vision model returned empty content "
                f"(done_reason={done_reason!r}, thinking_chars={thinking_len})"
            )

        return self._parse_state(text)

    def _parse_state(self, text: str) -> dict:
        candidate = text.strip()

        # Be tolerant if the model ignores the no-markdown request once.
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                raise RuntimeError(
                    "vision model did not return valid structured JSON"
                )
            try:
                raw = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "vision model returned malformed structured JSON"
                ) from exc

        if not isinstance(raw, dict):
            raise RuntimeError("vision structured response is not an object")

        def clean_text(value) -> str | None:
            if value is None:
                return None
            value = str(value).strip()
            return value or None

        def clean_list(value) -> list[str]:
            if not isinstance(value, list):
                return []
            out: list[str] = []
            for item in value:
                cleaned = clean_text(item)
                if cleaned is not None:
                    out.append(cleaned)
            return out[:12]

        description = clean_text(raw.get("description")) or ""
        return {
            "active_app": clean_text(raw.get("active_app")),
            "windows": clean_list(raw.get("windows")),
            "visible_text": clean_list(raw.get("visible_text")),
            "errors": clean_list(raw.get("errors")),
            "dialogs": clean_list(raw.get("dialogs")),
            "ui_state": clean_list(raw.get("ui_state")),
            "description": description,
        }

    def _request_vlm(
        self,
        *,
        image_b64: str,
        prompt: str,
        num_predict: int,
    ) -> dict:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "options": {
                "temperature": 0.1,
                "num_predict": int(num_predict),
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
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vision Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach vision Ollama at {self.base_url}"
            ) from exc
