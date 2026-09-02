from __future__ import annotations
import copy
from typing import Any, Callable, Dict, Optional
from action.action_protocol import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    GoalState,
    GoalStatus,
    result_to_world_feedback,
)

from brain.brain_client import BrainClient

ExecutorFn = Callable[[ActionRequest, Dict[str, Any]], ActionResult]

class AgentLoop:
    '''Closed loop: observe -> reason -> validate -> execute -> feedback -> observe again.'''
    def __init__(self, brain: BrainClient, executor: ExecutorFn, max_steps: int = 10):
        self.brain = brain
        self.executor = executor
        self.max_steps = max_steps

    @staticmethod
    def _target_id(world_state: Dict[str, Any]) -> Optional[str]:
        p = world_state.get('person')
        return p.get('id') if isinstance(p, dict) else None

    @staticmethod
    def _inject_feedback(world_state: Dict[str, Any], result: ActionResult, goal: Optional[GoalState]) -> Dict[str, Any]:
        state = copy.deepcopy(world_state)
        state.setdefault('feedback', {}).update(result_to_world_feedback(result))
        if goal is not None:
            state['goal'] = goal.to_dict()
        return state

    def step(self, world_state: Dict[str, Any], goal: Optional[GoalState] = None):
        decision = self.brain.decide(world_state)

        if not decision.get('brain_ok'):
            result = ActionResult(
                request_id='brain-unavailable', action='WAIT', status=ActionStatus.REJECTED,
                reason=decision.get('reason', 'BRAIN_UNAVAILABLE'), goal_reached=False)
            return self._inject_feedback(world_state, result, goal), result

        proposal = decision['proposal']
        req = ActionRequest(
            action=decision['final_action'], intent=proposal.get('intent'),
            confidence=proposal.get('confidence'), target_id=self._target_id(world_state),
            goal_id=goal.goal_id if goal else None)

        if not decision.get('approved', False):
            result = ActionResult(
                request_id=req.request_id, action=req.action, status=ActionStatus.REJECTED,
                reason=decision.get('reason', 'VALIDATOR_REJECTED'),
                observations={'llm_proposal': proposal, 'validator_violations': decision.get('violations', [])},
                goal_reached=False)
        else:
            result = self.executor(req, world_state)

        if goal is not None:
            goal.record_result(result)
        return self._inject_feedback(world_state, result, goal), result

    def run_goal(self, world_state: Dict[str, Any], goal: GoalState):
        state = copy.deepcopy(world_state)
        for _ in range(self.max_steps):
            if goal.status != GoalStatus.ACTIVE:
                break
            state, _ = self.step(state, goal)
        if goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.FAILED
        return state, goal


def fake_executor(req: ActionRequest, world_state: Dict[str, Any]) -> ActionResult:
    '''Temporary control block to prove feedback flow before real motors are connected.'''
    person = world_state.get('person') or {}
    distance = person.get('distance_m')

    if req.action in {'WAIT', 'SPEAK', 'ATTEND_PERSON', 'STOP'}:
        return ActionResult(req.request_id, req.action, ActionStatus.SUCCESS, reason='FAKE_EXECUTOR_OK')

    if req.action in {'MOVE_CLOSER', 'MOVE_TOWARD_PERSON', 'APPROACH_PERSON', 'APPROACH_USER'} and isinstance(distance, (int, float)):
        new_distance = max(0.8, distance - 0.5)
        return ActionResult(
            req.request_id, req.action, ActionStatus.SUCCESS,
            reason='FAKE_MOVE_COMPLETED',
            actual={'previous_distance_m': distance, 'distance_to_person_m': new_distance, 'moved_distance_m': distance-new_distance},
            goal_reached=(new_distance <= 1.2))

    return ActionResult(req.request_id, req.action, ActionStatus.FAILED, reason='SKILL_NOT_IMPLEMENTED', goal_reached=False)

if __name__ == '__main__':
    brain = BrainClient(host='http://192.168.128.120:11434', model='qwen3:8b')
    agent = AgentLoop(brain, fake_executor, max_steps=3)
    world = {
        'robot': {'state': 'observing', 'battery_percent': 75},
        'person': {'id': 'person_1', 'distance_m': 2.0, 'facing_robot': True, 'motion': 'stationary'},
        'events': ['USER_SPEAKING'],
        'speech': '少しこちらに来てもらえる？',
    }
    goal = GoalState(description='be near person_1', target_id='person_1', success_conditions={'distance_to_person_m_lte': 1.2}, max_attempts=3)
    new_state, result = agent.step(world, goal)
    print('RESULT:', result.to_dict())
    print('FEEDBACK:', new_state.get('feedback'))
    print('GOAL:', goal.to_dict())
