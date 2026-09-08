from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.training_memory import TrainingMemory
from memory.episode_quality import EpisodeQualityStore
from memory.episode_store import EpisodeStore


def find_episode(episode_id: str):
    for episode in EpisodeStore().iter_episodes():
        if episode.episode_id == episode_id:
            return episode
    return None


def print_summary() -> None:
    quality = EpisodeQualityStore()
    excluded = quality.excluded_ids()
    raw = list(EpisodeStore().iter_episodes())
    trainable = list(TrainingMemory().iter_trainable())

    raw_labels: Counter[str] = Counter()
    train_labels: Counter[str] = Counter()
    sources: Counter[str] = Counter()

    for episode in raw:
        label = str(episode.proposed_meaning or "UNLABELED")
        raw_labels[label] += 1
        sources[episode.source] += 1

    for episode in trainable:
        label = str(episode.proposed_meaning or "UNLABELED")
        train_labels[label] += 1

    print(f"raw episodes: {len(raw)}")
    print(f"excluded episodes: {len(excluded)}")
    print(f"trainable episodes: {len(trainable)}")
    print(f"raw labels: {dict(raw_labels)}")
    print(f"trainable labels: {dict(train_labels)}")
    print(f"sources: {dict(sources)}")

    if excluded:
        print("excluded ids:")
        latest = quality.latest_by_episode()
        for episode_id in sorted(excluded):
            record = latest.get(episode_id, {})
            print(
                f"  {episode_id} reason={record.get('reason')} "
                f"note={record.get('note')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and quarantine learning episodes without deleting raw data."
    )
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--exclude", metavar="EPISODE_ID")
    parser.add_argument("--restore", metavar="EPISODE_ID")
    parser.add_argument("--reason", default="low_quality")
    parser.add_argument("--note")
    args = parser.parse_args()

    if args.exclude and args.restore:
        parser.error("Choose only one of --exclude or --restore")

    store = EpisodeQualityStore()

    if args.exclude:
        if find_episode(args.exclude) is None:
            raise SystemExit(f"Episode not found: {args.exclude}")
        store.record(
            episode_id=args.exclude,
            excluded=True,
            reason=args.reason,
            note=args.note,
        )
        print(f"Excluded from training: {args.exclude}")

    if args.restore:
        if find_episode(args.restore) is None:
            raise SystemExit(f"Episode not found: {args.restore}")
        store.record(
            episode_id=args.restore,
            excluded=False,
            reason="restored",
            note=args.note,
        )
        print(f"Restored to training eligibility: {args.restore}")

    if args.summary or (not args.exclude and not args.restore):
        print_summary()


if __name__ == "__main__":
    main()
