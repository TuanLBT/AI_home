from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional
import time, uuid

class ActionStatus(str, Enum):
    PENDING='PENDING'; STARTED='STARTED'; RUNNING='RUNNING'; SUCCESS='SUCCESS'; FAILED='FAILED'; CANCELLED='CANCELLED'; REJECTED='REJECTED'

class GoalStatus(str, Enum):
    ACTIVE='ACTIVE'; ACHIEVED='ACHIEVED'; FAILED='FAILED'; CANCELLED='CANCELLED'

@dataclass
class ActionRequest:
    action: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    goal_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    source: str = 'llm'
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

@dataclass
class ActionResult:
    request_id: str
    action: str
    status: ActionStatus
    reason: Optional[str] = None
    actual: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)
    goal_reached: Optional[bool] = None
    started_at: Optional[float] = None
    finished_at: float = field(default_factory=time.time)
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self); d['status'] = self.status.value; return d

@dataclass
class GoalState:
    description: str
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.ACTIVE
    target_id: Optional[str] = None
    success_conditions: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 10
    last_action_request_id: Optional[str] = None
    last_action_result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    def record_result(self, result: ActionResult) -> None:
        self.attempts += 1
        self.last_action_request_id = result.request_id
        self.last_action_result = result.to_dict()
        self.updated_at = time.time()
        if result.goal_reached is True:
            self.status = GoalStatus.ACHIEVED
        elif self.attempts >= self.max_attempts:
            self.status = GoalStatus.FAILED
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self); d['status'] = self.status.value; return d

def result_to_world_feedback(result: ActionResult) -> Dict[str, Any]:
    return {
        'last_action': result.action,
        'last_action_status': result.status.value,
        'last_action_reason': result.reason,
        'last_action_actual': result.actual,
        'last_action_observations': result.observations,
        'last_action_goal_reached': result.goal_reached,
        'last_action_request_id': result.request_id,
        'last_action_finished_at': result.finished_at,
    }
