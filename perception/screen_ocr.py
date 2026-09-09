from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


class ScreenOCR:
    """Best-effort local OCR for exact desktop text.

    VLM remains responsible for visual/layout understanding. OCR is kept as
    separate evidence and is run on smaller tiles so tiny UI text can be read
    without sending one huge monitor image through Tesseract. TSV confidence
    filtering removes most of the random wallpaper/icon garbage that plain
    text mode produced.
    """

    def __init__(
        self,
        *,
        executable: str | None = None,
        language: str | None = None,
        timeout_s: float = 7.0,
        max_lines: int = 40,
        upscale: float = 1.35,
        min_confidence: float = 45.0,
        tile_rows: int = 2,
        tile_cols: int = 2,
        max_workers: int = 2,
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
        self.min_confidence = max(0.0, min(100.0, float(min_confidence)))
        self.tile_rows = max(1, int(tile_rows))
        self.tile_cols = max(1, int(tile_cols))
        self.max_workers = max(1, int(max_workers))

    def metadata(self) -> dict:
        return {
            "backend": "tesseract_cli_tiled_tsv",
            "available": bool(self.executable),
            "language": self.language,
            "tiles": f"{self.tile_rows}x{self.tile_cols}",
            "min_confidence": self.min_confidence,
        }

    def read(self, frame: np.ndarray) -> dict:
        if not self.executable:
            return self._error("tesseract executable not found")

        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return self._error("invalid OCR frame")

        try:
            tiles = self._tiles(frame)
            lines: list[str] = []
            errors: list[str] = []

            # Tesseract is CPU-heavy. A small worker pool lets independent
            # screen regions run concurrently without saturating the machine.
            workers = min(self.max_workers, len(tiles))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self._read_tile, tile): index
                    for index, tile in enumerate(tiles)
                }
                ordered: dict[int, list[str]] = {}
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        ordered[index] = future.result()
                    except Exception as exc:
                        errors.append(f"tile_{index}: {exc}")

            for index in range(len(tiles)):
                for line in ordered.get(index, []):
                    if line not in lines:
                        lines.append(line)
                    if len(lines) >= self.max_lines:
                        break
                if len(lines) >= self.max_lines:
                    break

            if lines:
                result = {
                    "available": True,
                    "backend": "tesseract_cli_tiled_tsv",
                    "language": self.language,
                    "text_lines": lines,
                }
                if errors:
                    result["partial_errors"] = errors
                return result

            if errors:
                return self._error("; ".join(errors))

            return {
                "available": True,
                "backend": "tesseract_cli_tiled_tsv",
                "language": self.language,
                "text_lines": [],
            }
        except Exception as exc:
            return self._error(str(exc))

    def _read_tile(self, tile: np.ndarray) -> list[str]:
        image = self._prepare(tile)
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
                "tsv",
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

        tsv = proc.stdout.decode("utf-8", errors="replace")
        return self._lines_from_tsv(tsv)

    def _tiles(self, frame: np.ndarray) -> list[np.ndarray]:
        height, width = frame.shape[:2]
        tiles: list[np.ndarray] = []

        for row in range(self.tile_rows):
            y0 = round(height * row / self.tile_rows)
            y1 = round(height * (row + 1) / self.tile_rows)
            for col in range(self.tile_cols):
                x0 = round(width * col / self.tile_cols)
                x1 = round(width * (col + 1) / self.tile_cols)
                tile = frame[y0:y1, x0:x1]
                if tile.size:
                    tiles.append(tile.copy())

        return tiles or [frame]

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

        # CLAHE improves local contrast on mixed dark/light desktop regions
        # without forcing one global threshold across terminals and wallpaper.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _lines_from_tsv(self, text: str) -> list[str]:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        grouped: dict[tuple[str, str, str, str], list[str]] = {}

        for row in reader:
            word = " ".join(str(row.get("text") or "").strip().split())
            if not word:
                continue

            try:
                confidence = float(row.get("conf") or -1)
            except (TypeError, ValueError):
                confidence = -1.0
            if confidence < self.min_confidence:
                continue

            key = (
                str(row.get("page_num") or ""),
                str(row.get("block_num") or ""),
                str(row.get("par_num") or ""),
                str(row.get("line_num") or ""),
            )
            grouped.setdefault(key, []).append(word)

        out: list[str] = []
        for words in grouped.values():
            line = " ".join(words).strip()
            if len(line) < 2:
                continue
            if len(line) > 180:
                line = line[:180]
            if line not in out:
                out.append(line)
            if len(out) >= self.max_lines:
                break
        return out

    def _error(self, message: str) -> dict:
        return {
            "available": False,
            "backend": "tesseract_cli_tiled_tsv",
            "language": self.language,
            "error": message,
            "text_lines": [],
        }
