from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.screen_vision import ScreenVisionPerception
from sources.screen import ScreenSource


def main() -> None:
    screen = ScreenSource()
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
    print("Asking vision model + OCR...")

    result = vision.describe(packet)
    if not result.get("available"):
        print(f"VISION ERROR: {result.get('error')}")
        return

    print(
        f"VISION OK [{result.get('model')}] "
        f"monitors={result.get('monitor_count')}"
    )

    for monitor in result.get("monitors", []):
        print(
            f"\n[{monitor.get('id')} {monitor.get('position')}] "
            f"input={monitor.get('input_width')}x{monitor.get('input_height')}"
        )

        if monitor.get("description"):
            print("VLM:")
            print(monitor["description"])
        elif monitor.get("vision_error"):
            print(f"VLM ERROR: {monitor['vision_error']}")

        ocr = monitor.get("ocr") or {}
        if ocr.get("available"):
            print("OCR:")
            lines = ocr.get("text_lines") or []
            if lines:
                for line in lines:
                    print(f"  {line}")
            else:
                print("  <no text>")
        else:
            print(f"OCR ERROR: {ocr.get('error')}")


if __name__ == "__main__":
    main()
