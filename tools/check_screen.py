from __future__ import annotations

import time

from sources.screen import ScreenSource


def main() -> None:
    source = ScreenSource(capture_interval_s=0.5)
    print(f"Screen source: {source.metadata()}")
    print("Move/change something on screen. Ctrl+C to quit.")

    last_available = False

    try:
        while True:
            packets = source.update()
            context = source.latest_context()

            if context.get("available") and not last_available:
                print(
                    "SCREEN OK: "
                    f"{context.get('width')}x{context.get('height')} "
                    f"backend={context.get('capture_backend')}"
                )
                last_available = True

            for packet in packets:
                meta = packet.metadata
                print(
                    "SCREEN CHANGED: "
                    f"score={meta.get('change_score', 0.0):.4f} "
                    f"size={meta.get('width')}x{meta.get('height')}"
                )

            if not context.get("available") and context.get("capture_error"):
                print(f"SCREEN ERROR: {context['capture_error']}")
                return

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
