from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.experience_labels import ExperienceLabelStore


KEY_TO_LABEL = {
    "g": "positive",
    "b": "negative",
    "n": "neutral",
}


def load_experiences(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Experience file not found: {path}. Run main.py first."
        )

    experiences: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if record.get("kind") == "action_result":
                experiences.append(record)

    return experiences


def describe(record: dict[str, Any]) -> str:
    observation = record.get("observation") or {}
    dialogue = observation.get("dialogue") or {}
    decision = record.get("decision") or {}
    result = record.get("result") or {}

    user_text = dialogue.get("last_user_text")
    action = decision.get("requested_action")
    reply = result.get("speech_text")
    status = result.get("status")
    visual = observation.get("visual")

    lines = [
        f"time: {record.get('recorded_at_utc')}",
        f"entity: {record.get('entity_id')}",
        f"user: {user_text!r}",
        f"action: {action} | status: {status}",
        f"reply: {reply!r}",
    ]

    if visual:
        lines.append(
            "visual: "
            + json.dumps(visual, ensure_ascii=False)
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually label unreviewed Indoor AI experiences."
    )
    parser.add_argument(
        "--experiences",
        type=Path,
        default=Path("data/experiences.jsonl"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/experience_labels.jsonl"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of unreviewed actions to show.",
    )
    args = parser.parse_args()

    label_store = ExperienceLabelStore(args.labels)
    reviewed = label_store.latest_by_experience()
    experiences = load_experiences(args.experiences)

    pending = [
        record
        for record in experiences
        if record.get("experience_id") not in reviewed
    ]

    if args.limit > 0:
        pending = pending[-args.limit:]

    if not pending:
        print("No unreviewed action experiences.")
        return

    print(
        "g=good, b=bad, n=neutral, s=skip, q=quit"
    )

    for index, record in enumerate(pending, start=1):
        print(f"\n[{index}/{len(pending)}]")
        print(describe(record))

        while True:
            key = input("label> ").strip().lower()

            if key in KEY_TO_LABEL:
                note = input("note (optional)> ")
                label_store.record(
                    record["experience_id"],
                    KEY_TO_LABEL[key],
                    note=note,
                )
                break

            if key == "s":
                break

            if key == "q":
                return

            print("Use g, b, n, s, or q.")


if __name__ == "__main__":
    main()
