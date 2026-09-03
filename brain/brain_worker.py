from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Any

from brain.brain_client import BrainClient


@dataclass(slots=True)
class BrainJob:
    entity_id: str
    world_state: dict[str, Any]
    timestamp: float


class BrainWorker:
    """
    Non-blocking worker for high-level action reasoning.

    main.py owns WorldState and ActionExecutor.

    BrainWorker only:
        receives an immutable-ish snapshot
        -> calls BrainClient in a background thread
        -> returns a decision event

    It does NOT execute actions and does NOT mutate WorldState.
    This keeps the camera/perception loop non-blocking and avoids
    cross-thread ownership of robot/control state.
    """

    def __init__(
        self,
        brain: BrainClient | None = None,
        *,
        host: str = "http://192.168.128.120:11434",
        model: str = "qwen3:8b",
        timeout: float = 30.0,
        max_pending: int = 2,
    ):
        self.brain = brain or BrainClient(
            host=host,
            model=model,
            timeout=timeout,
        )

        self._jobs: queue.Queue[BrainJob | None] = queue.Queue(
            maxsize=max_pending
        )

        self._results: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=16
        )

        self._busy = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            name="brain-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(
        self,
        *,
        entity_id: str,
        world_state: dict[str, Any],
        timestamp: float,
    ) -> bool:
        """
        Queue one brain reasoning job.

        Returns False instead of blocking when the queue is full.
        """
        job = BrainJob(
            entity_id=entity_id,
            world_state=dict(world_state),
            timestamp=timestamp,
        )

        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            return False

    def update(self) -> list[dict[str, Any]]:
        """
        Drain completed brain decisions.

        Safe to call every frame from main.py.
        """
        results: list[dict[str, Any]] = []

        while True:
            try:
                results.append(
                    self._results.get_nowait()
                )
            except queue.Empty:
                break

        return results

    def close(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            job = self._jobs.get()

            if job is None:
                self._jobs.task_done()
                return

            self._busy.set()

            try:
                decision = self.brain.decide(
                    job.world_state
                )

                result = {
                    "type": "BRAIN_DECISION",
                    "entity_id": job.entity_id,
                    "timestamp": job.timestamp,
                    **decision,
                }

            except Exception as exc:
                # BrainClient normally converts its own errors into decision
                # dictionaries, but this is the final worker-level guard.
                result = {
                    "type": "BRAIN_ERROR",
                    "entity_id": job.entity_id,
                    "timestamp": job.timestamp,
                    "brain_ok": False,
                    "approved": False,
                    "final_action": "WAIT",
                    "executor_action": None,
                    "reason": "BRAIN_WORKER_ERROR",
                    "error": str(exc),
                }

            try:
                self._results.put_nowait(
                    result
                )
            except queue.Full:
                pass
            finally:
                self._busy.clear()
                self._jobs.task_done()


if __name__ == "__main__":
    import json
    import time

    worker = BrainWorker()

    test_state = {
        "robot": {
            "state": "attending",
            "battery_percent": 75,
        },
        "person": {
            "id": "person_1",
            "distance_m": 1.5,
            "facing_robot": True,
            "motion": "stationary",
        },
        "speech": "こっち来て",
        "feedback": {
            "request_id": "example-request",
            "action": "MOVE_TOWARD_PERSON",
            "command": "move_toward_person",
            "status": "FAILED",
            "reason": "OBSTACLE_DETECTED",
            "actual": {
                "moved_distance_m": 0.0,
            },
            "observations": {
                "obstacle_ahead": True,
            },
            "goal_id": "example-goal",
            "goal_reached": False,
        },
    }

    accepted = worker.submit(
        entity_id="person_1",
        world_state=test_state,
        timestamp=time.monotonic(),
    )

    print(
        "submitted:",
        accepted,
    )

    # Demonstrates that submit() itself does not wait for Qwen.
    print(
        "main thread is free; waiting only in this smoke test..."
    )

    deadline = time.monotonic() + 35.0

    while time.monotonic() < deadline:
        results = worker.update()

        if results:
            print(
                json.dumps(
                    results[0],
                    ensure_ascii=False,
                    indent=2,
                )
            )
            break

        time.sleep(0.05)
    else:
        print("No result before smoke-test deadline.")
