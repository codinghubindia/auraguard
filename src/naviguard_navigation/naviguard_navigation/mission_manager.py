"""Mission State Machine for NAVIGUARD autonomous navigation."""

from enum import IntEnum
from typing import Optional, Dict, Any


class MissionState(IntEnum):
    """Mission execution lifecycle states."""
    IDLE = 0
    GOAL_SET = 1
    PLANNING = 2
    NAVIGATING = 3
    REPLANNING = 4
    RECOVERY_WAIT = 5
    GOAL_REACHED = 6
    MISSION_FAILED = 7

    def to_string(self) -> str:
        return self.name


class MissionManager:
    """Manages mission state transitions and planning retry limits."""

    VALID_TRANSITIONS = {
        MissionState.IDLE: {MissionState.GOAL_SET, MissionState.MISSION_FAILED},
        MissionState.GOAL_SET: {MissionState.PLANNING, MissionState.IDLE},
        MissionState.PLANNING: {MissionState.NAVIGATING, MissionState.MISSION_FAILED, MissionState.IDLE},
        MissionState.NAVIGATING: {
            MissionState.GOAL_REACHED,
            MissionState.REPLANNING,
            MissionState.RECOVERY_WAIT,
            MissionState.MISSION_FAILED,
            MissionState.IDLE,
        },
        MissionState.REPLANNING: {
            MissionState.NAVIGATING,
            MissionState.RECOVERY_WAIT,
            MissionState.MISSION_FAILED,
            MissionState.IDLE,
        },
        MissionState.RECOVERY_WAIT: {
            MissionState.REPLANNING,
            MissionState.MISSION_FAILED,
            MissionState.IDLE,
        },
        MissionState.GOAL_REACHED: {MissionState.GOAL_SET, MissionState.IDLE},
        MissionState.MISSION_FAILED: {MissionState.GOAL_SET, MissionState.IDLE},
    }

    def __init__(self, max_replan_retries: int = 5) -> None:
        self.state: MissionState = MissionState.IDLE
        self.state_entry_time: float = 0.0
        self.replan_count: int = 0
        self.max_replan_retries: int = max_replan_retries
        self.failure_reason: str = "NONE"

    def transition_to(self, new_state: MissionState, now_sec: float, reason: str = "NONE") -> bool:
        """Attempt to transition to a new mission state."""
        allowed = self.VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            return False

        self.state = new_state
        self.state_entry_time = now_sec
        self.failure_reason = reason

        if new_state == MissionState.REPLANNING:
            self.replan_count += 1
        elif new_state in (MissionState.IDLE, MissionState.GOAL_SET, MissionState.GOAL_REACHED):
            self.replan_count = 0

        return True

    def record_planning_failure(self, now_sec: float, reason: str) -> bool:
        """Handle planning failure and check retry budget."""
        self.replan_count += 1
        if self.replan_count >= self.max_replan_retries:
            self.transition_to(MissionState.MISSION_FAILED, now_sec, f"MAX_RETRIES_EXCEEDED: {reason}")
            return False
        return True

    def reset(self, now_sec: float) -> None:
        """Reset mission state to IDLE."""
        self.state = MissionState.IDLE
        self.state_entry_time = now_sec
        self.replan_count = 0
        self.failure_reason = "NONE"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state summary to dictionary."""
        return {
            "mission_state": self.state.to_string(),
            "mission_state_code": int(self.state),
            "replan_count": self.replan_count,
            "failure_reason": self.failure_reason,
        }
