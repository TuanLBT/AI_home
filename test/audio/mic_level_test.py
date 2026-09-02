from __future__ import annotations

import queue
import time

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
BLOCK_MS = 50
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

levels: queue.Queue[float] = queue.Queue(maxsize=32)


def callback(indata, frames, time_info, status):
    if frames == 0:
        return

    samples = np.asarray(indata[:, 0], dtype=np.float32)
    rms = float(np.sqrt(np.mean(samples * samples)))

    try:
        levels.put_nowait(rms)
    except queue.Full:
        pass


def bar(level: float, width: int = 50) -> str:
    # 0.10 RMS already counts as quite loud for most laptop mics.
    normalized = min(level / 0.10, 1.0)
    n = int(normalized * width)
    return "#" * n + "-" * (width - n)


def main():
    print("Mic level test")
    print("Speak normally, then stay silent.")
    print("Ctrl+C to quit.")
    print()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        try:
            while True:
                try:
                    level = levels.get(timeout=0.2)
                except queue.Empty:
                    continue

                print(
                    f"\rRMS={level:0.4f}  [{bar(level)}]",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
