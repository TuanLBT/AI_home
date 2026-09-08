from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.auto_trust import AutoTrustPolicy
from learning.meaning_proposer import MeaningProposer
from learning.pose_concept_learner import PoseConceptLearner
from learning.training_memory import TrainingMemory
from memory.episode_proposals import EpisodeProposalStore
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


def _source_name(episode) -> str | None:
    metadata = episode.metadata or {}
    for key in ("source_path", "path", "filename", "source_sample"):
        value = metadata.get(key)
        if value:
            return Path(str(value)).name
    return None


def _print_timeline(proposal: dict) -> None:
    windows = proposal.get("windows") or []
    if not windows:
        return
    print("  timeline:")
    for item in windows:
        print(
            f"    {float(item.get('start_s', 0.0)):5.2f}-{float(item.get('end_s', 0.0)):5.2f}s "
            f"{item.get('label')} {float(item.get('confidence', 0.0)):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose meanings for unlabeled/unreviewed learning episodes."
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--auto-confidence", type=float, default=0.55)
    parser.add_argument("--min-human", type=int, default=5)
    parser.add_argument("--max-auto-ratio", type=float, default=0.25)
    parser.add_argument("--all", action="store_true", help="Re-propose even if a proposal already exists.")
    parser.add_argument("--timeline", action="store_true", help="Print per-window predictions for temporal episodes.")
    args = parser.parse_args()

    training_memory = TrainingMemory()
    trainable = list(training_memory.iter_trainable(include_auto_trusted=False))
    learner = PoseConceptLearner.from_training_memory()
    proposer = MeaningProposer(learner, accept_threshold=args.threshold)
    policy = AutoTrustPolicy(
        min_human_examples=args.min_human,
        min_confidence=args.auto_confidence,
        max_auto_per_human_ratio=args.max_auto_ratio,
    )

    human_counts = policy.human_counts(trainable)
    auto_counts: Counter[str] = Counter()

    episodes = EpisodeStore()
    proposals = EpisodeProposalStore()
    reviews = EpisodeReviewStore()

    latest_proposals = proposals.latest_by_episode()
    latest_reviews = reviews.latest_by_episode()

    for record in latest_proposals.values():
        proposal = record.get("proposal") or {}
        trust = proposal.get("auto_trust") or {}
        if not trust.get("trusted"):
            continue
        label = str(trust.get("label") or "").strip()
        if label:
            auto_counts[label] += 1

    count = 0

    for episode in episodes.iter_episodes():
        if episode.episode_id in latest_reviews:
            continue
        if not args.all and episode.episode_id in latest_proposals:
            continue
        if episode.verified and episode.proposed_meaning:
            continue

        proposal = proposer.propose(episode)
        label = str(
            proposal.get("candidate_label")
            or proposal.get("label")
            or ""
        ).strip()

        trust = policy.evaluate(
            proposal,
            human_examples=human_counts.get(label, 0),
            existing_auto_examples=auto_counts.get(label, 0),
            source=episode.source,
        )
        proposal["auto_trust"] = trust.to_dict()

        if trust.trusted and trust.label:
            auto_counts[trust.label] += 1

        proposals.record(episode_id=episode.episode_id, proposal=proposal)
        count += 1

        candidate = proposal.get("candidate_label") or proposal.get("label")
        margin = trust.margin_ratio
        margin_text = "n/a" if margin is None else f"{margin:.2f}"
        source_name = _source_name(episode)
        name_text = f" file={source_name}" if source_name else ""
        print(
            f"{episode.episode_id} source={episode.source}{name_text} "
            f"candidate={candidate} confidence={proposal.get('confidence', 0.0):.2f} "
            f"margin={margin_text} accepted={proposal.get('accepted', False)} "
            f"auto_trusted={trust.trusted} reason={trust.reason}"
        )
        if args.timeline and episode.source != "image":
            _print_timeline(proposal)

    print(f"Done. Wrote {count} proposal(s).")


if __name__ == "__main__":
    main()
