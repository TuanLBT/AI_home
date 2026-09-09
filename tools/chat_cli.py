from __future__ import annotations

import argparse
import json
import socket
import threading


def _receiver(sock: socket.socket) -> None:
    try:
        with sock.makefile("r", encoding="utf-8", newline="\n") as reader:
            for line in reader:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = str(event.get("text") or "")
                if event.get("type") == "error":
                    print(f"\nAI ERROR> {text}")
                else:
                    print(f"\nAI> {text}")
                print("you> ", end="", flush=True)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean localhost chat client for Indoor AI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--entity-id", default="desktop_user")
    args = parser.parse_args()

    try:
        sock = socket.create_connection((args.host, args.port), timeout=3.0)
    except OSError as exc:
        raise SystemExit(
            f"Cannot connect to Indoor AI chat server at "
            f"{args.host}:{args.port}: {exc}\n"
            "Start `python main.py` first."
        ) from exc

    sock.settimeout(None)

    threading.Thread(
        target=_receiver,
        args=(sock,),
        name="chat-cli-receiver",
        daemon=True,
    ).start()

    print("Indoor AI chat connected. Ctrl+C to quit.")

    try:
        while True:
            text = input("you> ").strip()
            if not text:
                continue

            payload = {
                "entity_id": args.entity_id,
                "text": text,
            }
            sock.sendall(
                (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
