from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sources.screen import ScreenSource


def main() -> None:
    source = ScreenSource()
    print(f"Screen source: {source.metadata()}")
    print("Press Enter to capture. Ctrl+C to quit.")

    try:
        while True:
            input()
            packet = source.capture_now()
            context = source.latest_context()

            if packet is None:
                print(
                    "SCREEN ERROR: "
                    f"{context.get('capture_error', 'capture failed')}"
                )
                continue

            meta = packet.metadata
            print(
                "SCREEN OK: "
                f"{meta.get('width')}x{meta.get('height')} "
                f"backend={meta.get('capture_backend')} "
                f"changed={meta.get('changed')} "
                f"score={meta.get('change_score', 0.0):.4f}"
            )
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
