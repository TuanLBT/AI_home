from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel


@dataclass(slots=True)
class ASRJob:
    audio: np.ndarray
    sample_rate: int
    timestamp: float


class ASRWorker:
    """
    Non-blocking local ASR worker using faster-whisper.

    Default:
        model = base
        device = cpu
        compute_type = int8
        language = ja

    Camera/perception never waits for transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "ja",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.language = language

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self._jobs: queue.Queue[ASRJob | None] = queue.Queue(
            maxsize=4
        )
        self._results: queue.Queue[dict] = queue.Queue(
            maxsize=16
        )

        self._worker = threading.Thread(
            target=self._run,
            name="asr-worker",
            daemon=True,
        )
        self._worker.start()

    def submit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        timestamp: float,
    ) -> bool:
        if sample_rate != 16000:
            raise ValueError(
                "ASRWorker currently expects 16 kHz audio."
            )

        try:
            self._jobs.put_nowait(
                ASRJob(
                    audio=np.asarray(
                        audio,
                        dtype=np.float32,
                    ).copy(),
                    sample_rate=sample_rate,
                    timestamp=timestamp,
                )
            )
            return True
        except queue.Full:
            return False

    def update(self) -> list[dict]:
        results: list[dict] = []

        while True:
            try:
                results.append(
                    self._results.get_nowait()
                )
            except queue.Empty:
                break

        return results

    def _run(self) -> None:
        while True:
            job = self._jobs.get()

            if job is None:
                self._jobs.task_done()
                return

            try:
                segments, info = self.model.transcribe(
                    job.audio,
                    language=self.language,
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )

                text = "".join(
                    segment.text
                    for segment in segments
                ).strip()

                result = {
                    "type": "ASR_TEXT",
                    "timestamp": job.timestamp,
                    "text": text,
                    "language": info.language,
                    "language_probability": (
                        info.language_probability
                    ),
                }

            except Exception as exc:
                result = {
                    "type": "ASR_ERROR",
                    "timestamp": job.timestamp,
                    "error": str(exc),
                }

            try:
                self._results.put_nowait(result)
            except queue.Full:
                pass

            self._jobs.task_done()
