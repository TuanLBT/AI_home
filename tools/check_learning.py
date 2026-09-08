from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning.pose_concept_learner import PoseConceptLearner
from learning.training_memory import TrainingMemory
from memory.episode_proposals import EpisodeProposalStore
from memory.episode_reviews import EpisodeReviewStore
from memory.episode_store import EpisodeStore


def main() -> None:
    episodes = list(EpisodeStore().iter_episodes() or [])
    proposals = EpisodeProposalStore().latest_by_episode()
    reviews = EpisodeReviewStore().latest_by_episode()
    trainable = list(TrainingMemory().iter_trainable())

    raw_labels = Counter(
        episode.proposed_meaning
        for episode in episodes
        if episode.proposed_meaning
    )
    train_labels = Counter(
        episode.proposed_meaning
        for episode in trainable
        if episode.proposed_meaning
    )

    print("Indoor AI learning status")
    print(f"raw episodes: {len(episodes)}")
    print(f"automatic proposals: {len(proposals)}")
    print(f"human-reviewed episodes: {len(reviews)}")
    print(f"trainable episodes: {len(trainable)}")
    print(f"raw labels: {dict(raw_labels)}")
    print(f"training labels: {dict(train_labels)}")

    try:
        learner = PoseConceptLearner.from_training_memory()
    except (FileNotFoundError, ValueError) as exc:
        print(f"pose learner: NOT READY ({exc})")
        return

    print(f"pose learner: READY {learner.summary()}")


if __name__ == "__main__":
    main()
