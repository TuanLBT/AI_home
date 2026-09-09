from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.screen_vision import ScreenVisionPerception
from sources.screen import ScreenSource


def main() -> None:
    screen = ScreenSource(suppress_backend_stderr=False)
    vision = ScreenVisionPerception()

    print(f"Screen: {screen.metadata()}")
    print(f"Vision: {vision.metadata()}")
    print("Capturing one screenshot...")

    packet = screen.capture_now()
    if packet is None:
        print(f"SCREEN ERROR: {screen.latest_context().get('capture_error')}")
        return

    print(
        "Captured: "
        f"{packet.metadata.get('width')}x{packet.metadata.get('height')}"
    )
    print("Asking vision model...")

    result = vision.describe(packet)
    if not result.get("available"):
        print(f"VISION ERROR: {result.get('error')}")
        return

    print(
        f"VISION OK [{result.get('model')}] "
        f"input={result.get('input_width')}x{result.get('input_height')}"
    )
    print(result.get("description", ""))


if __name__ == "__main__":
    main()
