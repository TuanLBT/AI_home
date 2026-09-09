from __future__ import annotations

from abc import ABC, abstractmethod

from audio.mic_vad import MicVAD


class AudioSource(ABC):
    """Hardware-neutral audio event source."""

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self) -> list[dict]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def set_suppressed(self, suppressed: bool) -> None:
        return None

    def metadata(self) -> dict:
        return {"source_type": self.__class__.__name__}


class MicVADAudioSource(AudioSource):
    """Current microphone + Silero VAD adapter.

    ReSpeaker or another array can replace this class later while preserving
    the same audio-event contract for ASR and higher layers.
    """

    def __init__(self, **mic_kwargs):
        self._mic = MicVAD(**mic_kwargs)

    def start(self) -> None:
        self._mic.start()

    def update(self) -> list[dict]:
        return self._mic.update()

    def set_suppressed(self, suppressed: bool) -> None:
        self._mic.set_suppressed(suppressed)

    @property
    def voice_active(self) -> bool:
        return self._mic.voice_active

    def close(self) -> None:
        self._mic.stop()

    def metadata(self) -> dict:
        return {
            "source_type": "microphone_vad",
            "capabilities": ["audio", "vad"],
            "sample_rate": self._mic.sample_rate,
        }
