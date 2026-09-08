from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.episode_proposals import EpisodeProposalStore
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


def find_episode(episode_id: str):
    for episode in EpisodeStore().iter_episodes():
        if episode.episode_id == episode_id:
            return episode
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List and review Indoor AI learning episodes."
    )
    parser.add_argument("--list", action="store_true", dest="list_pending")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also show policy-trusted auto-labeled episodes.",
    )
    parser.add_argument("--episode-id")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--accept", action="store_true")
    group.add_argument("--correct", metavar="LABEL")
    group.add_argument("--reject", action="store_true")
    parser.add_argument("--note")
    args = parser.parse_args()

    proposal_store = EpisodeProposalStore()
    review_store = EpisodeReviewStore()
    proposals = proposal_store.latest_by_episode()
    reviews = review_store.latest_by_episode()

    if args.list_pending or not args.episode_id:
        shown = 0
        auto_hidden = 0

        for episode in EpisodeStore().iter_episodes():
            if episode.episode_id in reviews:
                continue

            proposal_record = proposals.get(episode.episode_id, {})
            proposal = proposal_record.get("proposal", {})
            trust = proposal.get("auto_trust") or {}

            if trust.get("trusted") and not args.all:
                auto_hidden += 1
                continue

            candidate = proposal.get("candidate_label") or proposal.get("label")
            confidence = float(proposal.get("confidence", 0.0) or 0.0)
            print(
                f"{episode.episode_id} source={episode.source} "
                f"current={episode.proposed_meaning} "
                f"candidate={candidate} confidence={confidence:.2f} "
                f"auto_trusted={bool(trust.get('trusted'))}"
            )
            shown += 1

        print(f"Needs review: {shown}")
        if auto_hidden:
            print(f"Auto-trusted hidden: {auto_hidden} (use --all to inspect)")
        if not args.episode_id:
            return

    episode = find_episode(args.episode_id)
    if episode is None:
        raise SystemExit(f"Episode not found: {args.episode_id}")

    if args.accept:
        proposal = proposals.get(args.episode_id, {}).get("proposal", {})
        label = proposal.get("candidate_label") or proposal.get("label")
        if not label or label == "UNKNOWN":
            raise SystemExit("No usable proposal to accept; use --correct LABEL")
        record = review_store.record(
            episode_id=args.episode_id,
            decision="accept",
            label=str(label),
            note=args.note,
        )
    elif args.correct:
        record = review_store.record(
            episode_id=args.episode_id,
            decision="correct",
            label=args.correct,
            note=args.note,
        )
    elif args.reject:
        record = review_store.record(
            episode_id=args.episode_id,
            decision="reject",
            note=args.note,
        )
    else:
        raise SystemExit("Choose --accept, --correct LABEL, or --reject")

    print(
        f"Reviewed {record['episode_id']}: "
        f"decision={record['decision']} label={record.get('label')}"
    )


if __name__ == "__main__":
    main()
