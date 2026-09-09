from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np


class ScreenOCR:
    """Optional local OCR adapter for exact small desktop text.

    VLM remains responsible for visual/layout understanding. OCR is separate
    evidence for small text such as terminal output, clocks, dates and status
    labels. The adapter is intentionally best-effort: if Tesseract is missing
    or fails, screen vision still works.
    """

    def __init__(
        self,
        *,
        executable: str | None = None,
        language: str | None = None,
        timeout_s: float = 8.0,
        max_lines: int = 40,
        upscale: float = 1.6,
    ):
        self.executable = (
            executable
            or os.environ.get("INDOOR_AI_TESSERACT")
            or shutil.which("tesseract")
        )
        self.language = (
            language
            or os.environ.get("INDOOR_AI_OCR_LANG")
            or "eng"
        )
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_lines = max(1, int(max_lines))
        self.upscale = max(1.0, float(upscale))

    def metadata(self) -> dict:
        return {
            "backend": "tesseract_cli",
            "available": bool(self.executable),
            "language": self.language,
        }

    def read(self, frame: np.ndarray) -> dict:
        if not self.executable:
            return {
                "available": False,
                "backend": "tesseract_cli",
                "error": "tesseract executable not found",
                "text_lines": [],
            }

        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return {
                "available": False,
                "backend": "tesseract_cli",
                "error": "invalid OCR frame",
                "text_lines": [],
            }

        try:
            image = self._prepare(frame)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise RuntimeError("failed to encode OCR image")

            proc = subprocess.run(
                [
                    self.executable,
                    "stdin",
                    "stdout",
                    "-l",
                    self.language,
                    "--psm",
                    "11",
                ],
                input=encoded.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
            )
            if proc.returncode != 0:
                detail = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(detail or f"tesseract exited {proc.returncode}")

            text = proc.stdout.decode("utf-8", errors="replace")
            lines = self._clean_lines(text)
            return {
                "available": True,
                "backend": "tesseract_cli",
                "language": self.language,
                "text_lines": lines,
            }
        except Exception as exc:
            return {
                "available": False,
                "backend": "tesseract_cli",
                "language": self.language,
                "error": str(exc),
                "text_lines": [],
            }

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        if self.upscale > 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        # Mild contrast normalization helps tiny UI text while keeping both
        # dark and light themes usable. Avoid hard thresholding because a whole
        # desktop contains mixed backgrounds.
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    def _clean_lines(self, text: str) -> list[str]:
        out: list[str] = []
        for raw in text.splitlines():
            line = " ".join(raw.strip().split())
            if len(line) < 2:
                continue
            if len(line) > 180:
                line = line[:180]
            if line not in out:
                out.append(line)
            if len(out) >= self.max_lines:
                break
        return out
