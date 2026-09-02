from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
import time
import uuid


class ActionStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class GoalStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ACHIEVED = "ACHIEVED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class ActionRequest:
    action: str
    entity_id: str
    reason: str
    timestamp: float

    intent: Optional[str] = None
    confidence: Optional[float] = None
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    goal_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "behavior"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ActionResult:
    request_id: str
    entity_id: str
    action: str
    status: ActionStatus
    timestamp: float

    command: Optional[str] = None
    reason: Optional[str] = None

    actual: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, Any] = field(default_factory=dict)

    goal_id: Optional[str] = None
    goal_reached: Optional[bool] = None

    started_at: Optional[float] = None
    finished_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(slots=True)
class GoalState:
    description: str
    target_id: Optional[str] = None
    success_conditions: Dict[str, Any] = field(default_factory=dict)

    goal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.ACTIVE

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
        data = asdict(self)
        data["status"] = self.status.value
        return data


def result_to_world_feedback(result: ActionResult) -> Dict[str, Any]:
    return {
        "request_id": result.request_id,
        "entity_id": result.entity_id,
        "action": result.action,
        "command": result.command,
        "status": result.status.value,
        "reason": result.reason,
        "actual": dict(result.actual),
        "observations": dict(result.observations),
        "goal_id": result.goal_id,
        "goal_reached": result.goal_reached,
        "timestamp": result.timestamp,
        "finished_at": result.finished_at,
    }
