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
    """Turn a raw desktop screenshot into a neutral text representation.

    The backend is an Ollama VLM, but the rest of the agent only consumes the
    published representation. A future OCR/UI-tree/local model can replace or
    augment this adapter without changing ChatSource or LLMWorker.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.128.120:11434",
        model: str | None = None,
        timeout_s: float = 45.0,
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
            text = self._ask_vlm(encoded)
            result = {
                "available": True,
                "timestamp": packet.timestamp,
                "backend": "ollama_vlm",
                "model": self.model,
                "description": text,
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

    def _ask_vlm(self, image_b64: str) -> str:
        prompt = (
            "Analyze this current desktop screenshot as evidence for another "
            "agent. Describe only what is visibly supported. Focus on active "
            "apps/windows, important readable text, errors, dialogs, terminal "
            "output, and obvious UI state. Be concise but specific. Do not "
            "invent hidden content."
        )

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
                "num_predict": 256,
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
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vision Ollama HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"cannot reach vision Ollama at {self.base_url}"
            ) from exc

        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise RuntimeError("vision model returned an empty description")
        return text
