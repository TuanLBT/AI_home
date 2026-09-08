from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.meaning_proposer import MeaningProposer
from learning.pose_concept_learner import PoseConceptLearner
from memory.episode_proposals import EpisodeProposalStore
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose meanings for unlabeled/unreviewed learning episodes."
    )
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--all", action="store_true", help="Re-propose even if a proposal already exists.")
    args = parser.parse_args()

    learner = PoseConceptLearner.from_training_memory()
    proposer = MeaningProposer(learner, accept_threshold=args.threshold)
    episodes = EpisodeStore()
    proposals = EpisodeProposalStore()
    reviews = EpisodeReviewStore()

    latest_proposals = proposals.latest_by_episode()
    latest_reviews = reviews.latest_by_episode()
    count = 0

    for episode in episodes.iter_episodes():
        if episode.episode_id in latest_reviews:
            continue
        if not args.all and episode.episode_id in latest_proposals:
            continue
        if episode.verified and episode.proposed_meaning:
            continue

        proposal = proposer.propose(episode)
        proposals.record(episode_id=episode.episode_id, proposal=proposal)
        count += 1

        candidate = proposal.get("candidate_label") or proposal.get("label")
        print(
            f"{episode.episode_id} source={episode.source} "
            f"candidate={candidate} confidence={proposal.get('confidence', 0.0):.2f} "
            f"accepted={proposal.get('accepted', False)}"
        )

    print(f"Done. Wrote {count} proposal(s).")


if __name__ == "__main__":
    main()
