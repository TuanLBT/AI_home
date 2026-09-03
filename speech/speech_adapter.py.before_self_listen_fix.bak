from __future__ import annotations

import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SpeechResult:
    ok: bool
    backend: str
    text: str
    error: str | None = None


class SpeechAdapter:
    """
    Non-blocking offline speech adapter.

    Piper runs in a dedicated worker thread so TTS generation/playback
    does not block the camera/perception loop.
    """

    def __init__(
        self,
        language: str = "ja",
        rate: int = 0,
        piper_model: str = "models/piper/tsukuyomi.onnx",
        piper_config: str = "models/piper/config.json",
    ):
        self.language = language
        self.rate = rate

        self.project_root = Path(__file__).resolve().parents[1]
        self.piper_model = self.project_root / piper_model
        self.piper_config = self.project_root / piper_config
        self.piper_bin = shutil.which("piper")

        self.backend = self._find_backend()

        self._speech_queue: queue.Queue[str | None] = queue.Queue(maxsize=8)
        self._worker = threading.Thread(
            target=self._speech_worker,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()

    def _piper_ready(self) -> bool:
        return (
            self.piper_bin is not None
            and self.piper_model.is_file()
            and self.piper_config.is_file()
        )

    def _find_backend(self) -> str | None:
        if self._piper_ready():
            return "piper"

        for name in ("espeak-ng", "spd-say", "espeak"):
            if shutil.which(name):
                return name

        return None

    def speak(self, text: str) -> SpeechResult:
        if not text:
            return SpeechResult(
                ok=False,
                backend=self.backend or "none",
                text=text,
                error="empty_text",
            )

        if self.backend is None:
            return SpeechResult(
                ok=False,
                backend="none",
                text=text,
                error="no_supported_tts_backend",
            )

        try:
            self._speech_queue.put_nowait(text)

            return SpeechResult(
                ok=True,
                backend=self.backend,
                text=text,
            )

        except queue.Full:
            return SpeechResult(
                ok=False,
                backend=self.backend,
                text=text,
                error="speech_queue_full",
            )

    def _speech_worker(self) -> None:
        while True:
            text = self._speech_queue.get()

            if text is None:
                self._speech_queue.task_done()
                return

            try:
                self._speak_blocking(text)
            finally:
                self._speech_queue.task_done()

    def _speak_blocking(self, text: str) -> None:
        if self.backend == "piper":
            cmd = [
                self.piper_bin,
                text,
                "--model",
                str(self.piper_model),
                "--config",
                str(self.piper_config),
                "--auto-play",
            ]

        elif self.backend == "spd-say":
            cmd = [
                "spd-say",
                "-l",
                self.language,
                text,
            ]

        else:
            cmd = [
                self.backend,
                "-v",
                self.language,
            ]

            if self.rate:
                cmd += ["-s", str(self.rate)]

            cmd.append(text)

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
