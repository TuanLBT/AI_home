from __future__ import annotations

import time
from typing import Any, Dict, Optional

from action.action_executor import ActionExecutor
from action.action_protocol import GoalState, GoalStatus
from brain.brain_client import BrainClient
from world.world_state import WorldState


class AgentLoop:
    """
    High-level closed loop using the REAL runtime interfaces:

        brain state
            -> BrainClient
            -> validated action
            -> ActionExecutor
            -> ACTION_EXECUTED
            -> WorldState.record_action_result()
            -> next brain state

    This class does not own a second WorldState.
    The real WorldState instance remains the source of truth.
    """

    def __init__(
        self,
        brain: BrainClient,
        action_executor: ActionExecutor,
        world: WorldState,
        max_steps: int = 10,
    ):
        self.brain = brain
        self.action_executor = action_executor
        self.world = world
        self.max_steps = max_steps

    def build_brain_state(
        self,
        entity_id: str,
        *,
        behavior_state: Optional[str] = None,
        speech: Optional[str] = None,
        battery_percent: Optional[float] = None,
        speaker_confidence: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert runtime WorldState into the structured state consumed by
        BrainClient. Executor feedback is included automatically.
        """
        person = self.world.people.get(entity_id)

        person_state: Dict[str, Any] = {
            "id": entity_id,
        }

        if person is not None:
            person_state.update({
                "present": person.present,
                "posture": getattr(person, "posture", None),
                "motion": getattr(person, "fused_motion", None),
                "horizontal_state": getattr(person, "horizontal_state", None),
                "depth_state": getattr(person, "movement_state", None),
                "gestures": sorted(
                    getattr(person, "active_gestures", set())
                ),
            })

        state: Dict[str, Any] = {
            "robot": {
                "state": behavior_state,
                "battery_percent": battery_percent,
            },
            "person": person_state,
            "speech": speech,
            "feedback": self.world.get_action_feedback(entity_id),
        }

        if speaker_confidence is not None:
            state["audio_context"] = {
                "speaker_confidence": speaker_confidence,
            }

        if extra:
            state.update(extra)

        return state

    def step(
        self,
        entity_id: str,
        *,
        behavior_state: Optional[str] = None,
        speech: Optional[str] = None,
        battery_percent: Optional[float] = None,
        speaker_confidence: Optional[float] = None,
        goal: Optional[GoalState] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run one real high-level decision cycle.

        This is intentionally ONE step. The camera/perception loop should
        update WorldState before another step is attempted.
        """
        now = time.monotonic()

        brain_state = self.build_brain_state(
            entity_id,
            behavior_state=behavior_state,
            speech=speech,
            battery_percent=battery_percent,
            speaker_confidence=speaker_confidence,
            extra=extra_state,
        )

        if goal is not None:
            brain_state["goal"] = goal.to_dict()

        decision = self.brain.decide(brain_state)

        if not decision.get("brain_ok"):
            return {
                "type": "BRAIN_DECISION_FAILED",
                "entity_id": entity_id,
                "decision": decision,
                "timestamp": now,
            }

        proposal = decision.get("proposal") or {}

        if not decision.get("approved", False):
            return {
                "type": "ACTION_REJECTED",
                "entity_id": entity_id,
                "proposed_action": proposal.get("action"),
                "final_action": decision.get("final_action"),
                "reason": decision.get("reason"),
                "violations": decision.get("violations", []),
                "timestamp": now,
            }

        action_event = {
            "type": "ACTION",
            "entity_id": entity_id,
            "action": decision["final_action"],
            "reason": "BRAIN_DECISION",
            "timestamp": now,
            "intent": proposal.get("intent"),
            "confidence": proposal.get("confidence"),
            "goal_id": goal.goal_id if goal is not None else None,
            "source": "brain",
        }

        self.action_executor.submit(
            action_event,
            context={
                "brain_state": brain_state,
                "brain_proposal": proposal,
            },
        )

        executed_actions = self.action_executor.update()

        for executed in executed_actions:
            self.world.record_action_result(executed)

        latest_result = executed_actions[-1] if executed_actions else None

        if goal is not None and latest_result is not None:
            goal.attempts += 1
            goal.last_action_request_id = latest_result.get("request_id")
            goal.last_action_result = dict(latest_result)
            goal.updated_at = time.time()

            if latest_result.get("goal_reached") is True:
                goal.status = GoalStatus.ACHIEVED
            elif goal.attempts >= goal.max_attempts:
                goal.status = GoalStatus.FAILED

        return {
            "type": "AGENT_STEP_COMPLETED",
            "entity_id": entity_id,
            "decision": decision,
            "executed_actions": executed_actions,
            "feedback": self.world.get_action_feedback(entity_id),
            "goal": goal.to_dict() if goal is not None else None,
            "timestamp": now,
        }
