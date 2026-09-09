from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.request

import cv2
import numpy as np

from sources.base import SourcePacket
from sources.screen_bus import publish_screen_representation


class ScreenVisionPerception:
    """Turn desktop screenshots into grounded per-monitor screen state.

    Very wide frames are treated as an extended desktop and split into monitor
    regions before vision inference. Qwen3-VL 4B is currently more reliable on
    a concise prose task than strict structured generation, so each monitor is
    described independently and kept separate in the published representation.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://192.168.128.120:11434",
        model: str | None = None,
        timeout_s: float = 90.0,
        max_width: int = 2560,
        max_pixels: int = 2_300_000,
        jpeg_quality: int = 82,
        extended_aspect_threshold: float = 2.2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = (
            model
            or os.environ.get("INDOOR_AI_VISION_MODEL")
            or "qwen3-vl:4b"
        )
        self.timeout_s = float(timeout_s)
        self.max_width = max(640, int(max_width))
        self.max_pixels = max(640 * 480, int(max_pixels))
        self.jpeg_quality = max(40, min(95, int(jpeg_quality)))
        self.extended_aspect_threshold = max(
            1.5,
            float(extended_aspect_threshold),
        )

    def metadata(self) -> dict:
        return {
            "backend": "ollama_vlm",
            "model": self.model,
            "max_width": self.max_width,
            "max_pixels": self.max_pixels,
            "representation": "per_monitor_grounded_prose_v1",
            "multi_monitor": True,
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

        regions = self._monitor_regions(frame)
        monitors: list[dict] = []

        for index, region in enumerate(regions):
            position = self._monitor_position(index, len(regions))
            try:
                encoded, width, height = self._encode_frame(region)
                description = self._ask_vlm(encoded)
                monitors.append(
                    {
                        "id": f"monitor_{index}",
                        "position": position,
                        "available": True,
                        "description": description,
                        "input_width": width,
                        "input_height": height,
                    }
                )
            except Exception as exc:
                # One bad monitor must not discard evidence from the other one.
                monitors.append(
                    {
                        "id": f"monitor_{index}",
                        "position": position,
                        "available": False,
                        "description": "",
                        "error": str(exc),
                    }
                )

        available_monitors = [m for m in monitors if m.get("available")]
        description = " | ".join(
            f"{monitor['id']} ({monitor['position']}): "
            f"{monitor.get('description', '')}"
            for monitor in available_monitors
            if monitor.get("description")
        )

        result = {
            "available": bool(available_monitors),
            "timestamp": packet.timestamp,
            "backend": "ollama_vlm",
            "model": self.model,
            "schema": "per_monitor_grounded_prose_v1",
            "monitor_count": len(monitors),
            "monitors": monitors,
            "description": description,
            "source_width": int(frame.shape[1]),
            "source_height": int(frame.shape[0]),
        }

        if not available_monitors:
            errors = [
                str(m.get("error"))
                for m in monitors
                if m.get("error")
            ]
            result["error"] = "; ".join(errors) or "all monitor vision calls failed"

        publish_screen_representation(result)
        return result

    def _monitor_regions(self, frame: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        aspect = width / max(1.0, float(height))

        # Current KDE/Wayland capture returns the whole extended desktop. For
        # the user's side-by-side two-monitor layout, a very wide frame is split
        # into left and right display regions before VLM inference.
        if aspect >= self.extended_aspect_threshold:
            split_x = width // 2
            if split_x >= 640 and (width - split_x) >= 640:
                return [
                    frame[:, :split_x].copy(),
                    frame[:, split_x:].copy(),
                ]

        return [frame]

    @staticmethod
    def _monitor_position(index: int, count: int) -> str:
        if count == 2:
            return "left" if index == 0 else "right"
        if count == 1:
            return "single"
        return f"region_{index}"

    def _encode_frame(self, frame: np.ndarray) -> tuple[str, int, int]:
        rgb = frame
        height, width = rgb.shape[:2]

        # A split 4096x2880 monitor resized only by width becomes 2560x1800,
        # which is roughly twice the pixel load of the previously successful
        # 2560x900 panorama. Cap both width and total pixel count so vision-token
        # usage stays near the known-good range while preserving aspect ratio.
        scale = min(1.0, self.max_width / float(width))
        scaled_pixels = (width * scale) * (height * scale)
        if scaled_pixels > self.max_pixels:
            scale *= math.sqrt(self.max_pixels / float(scaled_pixels))

        if scale < 0.999:
            target = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            rgb = cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)

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
        # This prose task already proved reliable on qwen3-vl:4b. Keep monitor
        # identity outside the model response rather than forcing JSON, which
        # repeatedly caused the model to spend its whole budget in thinking.
        prompt = (
            "/no_think\n"
            "Describe this ONE monitor screenshot concisely using only visible "
            "evidence. Mention apps/windows, important readable text, errors or "
            "dialogs, and obvious UI state. Do not describe anything outside "
            "this screenshot. Do not guess. No reasoning preamble."
        )

        data = self._request_vlm(
            image_b64=image_b64,
            prompt=prompt,
            num_predict=768,
        )
        message = data.get("message") or {}
        text = str(message.get("content") or "").strip()
        if text:
            return text

        done_reason = data.get("done_reason")
        thinking_len = len(str(message.get("thinking") or ""))
        raise RuntimeError(
            "vision model returned empty content "
            f"(done_reason={done_reason!r}, thinking_chars={thinking_len})"
        )

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
                "temperature": 0.0,
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
