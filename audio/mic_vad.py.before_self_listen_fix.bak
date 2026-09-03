from __future__ import annotations

import queue
import time
from collections import deque

import numpy as np
import torch

try:
    import sounddevice as sd
except ImportError:
    sd = None

from silero_vad import VADIterator, load_silero_vad


class MicVAD:
    """
    Streaming Silero VAD + speech segment capture.

    Events:
        VOICE_ACTIVITY_STARTED
        VOICE_ACTIVITY_ENDED
        SPEECH_SEGMENT_READY

    SPEECH_SEGMENT_READY contains:
        audio: np.ndarray float32 mono @ 16 kHz
        sample_rate: int
        duration_s: float
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 350,
        speech_pad_ms: int = 250,
        pre_roll_ms: int = 250,
        max_segment_s: float = 20.0,
        device=None,
    ):
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. "
                "Install it with: pip install sounddevice"
            )

        if sample_rate != 16000:
            raise ValueError(
                "This MicVAD currently expects sample_rate=16000."
            )

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.max_segment_samples = int(max_segment_s * sample_rate)

        torch.set_num_threads(1)

        self.model = load_silero_vad(onnx=False)

        self.vad = VADIterator(
            self.model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=128
        )

        pre_roll_blocks = max(
            1,
            int(
                round(
                    (pre_roll_ms / 1000.0)
                    * sample_rate
                    / block_size
                )
            ),
        )

        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=pre_roll_blocks
        )

        self._stream = None
        self._voice_active = False
        self._segment_chunks: list[np.ndarray] = []

    @property
    def voice_active(self) -> bool:
        return self._voice_active

    def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self.vad.reset_states()

    def update(self) -> list[dict]:
        events: list[dict] = []

        while True:
            try:
                samples = self._audio_queue.get_nowait()
            except queue.Empty:
                break

            # Always keep a small amount of audio before speech begins.
            if not self._voice_active:
                self._pre_roll.append(samples.copy())

            result = self.vad(
                torch.from_numpy(samples),
                return_seconds=True,
            )

            now = time.monotonic()

            if result and "start" in result and not self._voice_active:
                self._voice_active = True
                self._segment_chunks = list(self._pre_roll)
                self._segment_chunks.append(samples.copy())

                events.append({
                    "type": "VOICE_ACTIVITY_STARTED",
                    "timestamp": now,
                    "level": self._rms(samples),
                })

                continue

            if self._voice_active:
                self._segment_chunks.append(samples.copy())

                total_samples = sum(
                    len(chunk)
                    for chunk in self._segment_chunks
                )

                if total_samples >= self.max_segment_samples:
                    events.extend(
                        self._finish_segment(
                            now=now,
                            samples=samples,
                            forced=True,
                        )
                    )
                    continue

            if result and "end" in result and self._voice_active:
                events.extend(
                    self._finish_segment(
                        now=now,
                        samples=samples,
                        forced=False,
                    )
                )

        return events

    def _finish_segment(
        self,
        now: float,
        samples: np.ndarray,
        forced: bool,
    ) -> list[dict]:
        audio = np.concatenate(
            self._segment_chunks
        ).astype(np.float32, copy=False)

        self._voice_active = False
        self._segment_chunks = []
        self._pre_roll.clear()

        duration_s = len(audio) / self.sample_rate

        return [
            {
                "type": "VOICE_ACTIVITY_ENDED",
                "timestamp": now,
                "level": self._rms(samples),
                "forced": forced,
            },
            {
                "type": "SPEECH_SEGMENT_READY",
                "timestamp": now,
                "audio": audio,
                "sample_rate": self.sample_rate,
                "duration_s": duration_s,
                "forced": forced,
            },
        ]

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(
            np.sqrt(
                np.mean(samples * samples)
            )
        )

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        if frames == 0:
            return

        samples = np.asarray(
            indata[:, 0],
            dtype=np.float32,
        ).copy()

        try:
            self._audio_queue.put_nowait(samples)

        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(samples)
            except queue.Empty:
                pass
