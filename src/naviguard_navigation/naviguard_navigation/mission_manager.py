"""Mission State Machine and Structured Failure/Success Explanation for NAVIGUARD."""

import time
from enum import IntEnum
from typing import Optional, Dict, Any, List


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


class FailureCode:
    """Controlled Failure Taxonomy for NAVIGUARD Autonomy Stack."""
    LOCALIZATION_LOST = "LOCALIZATION_LOST"
    VISUAL_FEATURES_INSUFFICIENT = "VISUAL_FEATURES_INSUFFICIENT"
    RELOCALIZATION_FAILED = "RELOCALIZATION_FAILED"
    RECOVERY_BUDGET_EXHAUSTED = "RECOVERY_BUDGET_EXHAUSTED"
    NO_SAFE_PATH = "NO_SAFE_PATH"
    PATH_BLOCKED = "PATH_BLOCKED"
    GOAL_IN_COLLISION_ZONE = "GOAL_IN_COLLISION_ZONE"
    GOAL_OUTSIDE_MAP = "GOAL_OUTSIDE_MAP"
    SENSOR_TIMEOUT = "SENSOR_TIMEOUT"
    CAMERA_DEGRADED = "CAMERA_DEGRADED"
    IMU_DEGRADED = "IMU_DEGRADED"
    ODOMETRY_INVALID = "ODOMETRY_INVALID"
    MAP_INVALID = "MAP_INVALID"
    MAP_TOO_UNKNOWN = "MAP_TOO_UNKNOWN"
    NAVIGATION_TIMEOUT = "NAVIGATION_TIMEOUT"
    GOAL_TIMEOUT = "GOAL_TIMEOUT"
    EXCESSIVE_SLIP = "EXCESSIVE_SLIP"
    OBSTACLE_UNRESOLVED = "OBSTACLE_UNRESOLVED"
    ENVIRONMENT_UNTRAVERSABLE = "ENVIRONMENT_UNTRAVERSABLE"
    SYSTEM_FAULT = "SYSTEM_FAULT"


FAILURE_TAXONOMY: Dict[str, Dict[str, str]] = {
    FailureCode.NO_SAFE_PATH: {
        "category": "NAVIGATION",
        "human_reason": "No safe collision-free path exists to the goal.",
        "detail": "The requested goal is reachable geometrically but all candidate routes violate safety, clearance, or traversability constraints.",
    },
    FailureCode.PATH_BLOCKED: {
        "category": "NAVIGATION",
        "human_reason": "Planned trajectory is blocked by obstacles.",
        "detail": "An obstacle appeared in the active path corridor and dynamic replanning found no collision-free bypass.",
    },
    FailureCode.GOAL_IN_COLLISION_ZONE: {
        "category": "NAVIGATION",
        "human_reason": "Target goal is inside an obstacle or safety buffer zone.",
        "detail": "The dispatched goal coordinate falls inside a lethal obstacle cell or violates minimum vehicle footprint clearance.",
    },
    FailureCode.GOAL_OUTSIDE_MAP: {
        "category": "NAVIGATION",
        "human_reason": "Target goal coordinates are outside explored map bounds.",
        "detail": "The requested goal position lies outside the operational boundary of the navigation occupancy grid.",
    },
    FailureCode.RECOVERY_BUDGET_EXHAUSTED: {
        "category": "RECOVERY",
        "human_reason": "Maximum recovery attempts exhausted without regaining clear path.",
        "detail": "The robot executed the maximum allowed recovery maneuvers (safe stop, backtrack, visual reacquisition) but conditions remained non-navigable.",
    },
    FailureCode.LOCALIZATION_LOST: {
        "category": "LOCALIZATION",
        "human_reason": "Localization confidence degraded below safe threshold.",
        "detail": "SLAM and visual odometry tracking quality dropped below operational limits, preventing safe pose estimation.",
    },
    FailureCode.VISUAL_FEATURES_INSUFFICIENT: {
        "category": "PERCEPTION",
        "human_reason": "Insufficient visual features in environment for odometry.",
        "detail": "Camera scene lacks sufficient distinctive visual keypoints or suffers from extreme optical degradation.",
    },
    FailureCode.RELOCALIZATION_FAILED: {
        "category": "LOCALIZATION",
        "human_reason": "Unable to re-establish pose against known landmarks.",
        "detail": "Closed-loop relocalization attempts timed out without confirming robot position with sufficient confidence.",
    },
    FailureCode.ENVIRONMENT_UNTRAVERSABLE: {
        "category": "ENVIRONMENT",
        "human_reason": "Terrain difficulty or slope exceeds vehicle safety limits.",
        "detail": "Slope estimation or mud/traction analysis indicates traversing this terrain would result in vehicle rollover or entrapment.",
    },
    FailureCode.NAVIGATION_TIMEOUT: {
        "category": "NAVIGATION",
        "human_reason": "Mission execution duration exceeded allotted time budget.",
        "detail": "The vehicle was unable to reach the target within the expected mission duration.",
    },
    FailureCode.SYSTEM_FAULT: {
        "category": "SYSTEM",
        "human_reason": "Critical autonomy subsystem fault detected.",
        "detail": "One or more core autonomy subsystems (IMU, Camera, VO, SLAM) experienced an operational fault.",
    },
    FailureCode.SENSOR_TIMEOUT: {
        "category": "SENSOR",
        "human_reason": "Sensor stream timed out or halted.",
        "detail": "Required perception sensors stopped publishing telemetry within the required latency threshold.",
    },
    FailureCode.CAMERA_DEGRADED: {
        "category": "PERCEPTION",
        "human_reason": "Camera feed degraded or obscured.",
        "detail": "Front optical sensor frame age exceeded threshold or frame quality dropped below minimum visibility.",
    },
    FailureCode.IMU_DEGRADED: {
        "category": "SENSOR",
        "human_reason": "IMU attitude tracking degraded.",
        "detail": "Inertial measurement unit angular rate or acceleration measurements experienced drift or divergence.",
    },
    FailureCode.ODOMETRY_INVALID: {
        "category": "LOCALIZATION",
        "human_reason": "Odometry calculation produced non-finite or divergent pose.",
        "detail": "Visual odometry transform contained NaN, infinite values, or unphysical velocity leaps.",
    },
    FailureCode.MAP_INVALID: {
        "category": "SLAM",
        "human_reason": "Navigation occupancy grid is uninitialized or corrupted.",
        "detail": "Occupancy grid dimensions, resolution, or origin are invalid for global planning.",
    },
    FailureCode.MAP_TOO_UNKNOWN: {
        "category": "SLAM",
        "human_reason": "Target path traverses too much unexplored space.",
        "detail": "Candidate trajectory exceeds maximum allowable unknown cells and exploration is forbidden.",
    },
    FailureCode.GOAL_TIMEOUT: {
        "category": "NAVIGATION",
        "human_reason": "Target goal timed out before arrival.",
        "detail": "Mission did not achieve goal convergence within the allotted time window.",
    },
    FailureCode.EXCESSIVE_SLIP: {
        "category": "CONTROL",
        "human_reason": "Severe wheel slippage or sinkage detected on difficult terrain.",
        "detail": "Visual odometry progress failed to correlate with wheel velocity commands in mud or sand.",
    },
    FailureCode.OBSTACLE_UNRESOLVED: {
        "category": "RECOVERY",
        "human_reason": "Obstacle could not be bypassed after recovery.",
        "detail": "Dynamic obstacle persists across replanning attempts and vehicle is constrained.",
    },
}


class MissionManager:
    """Manages mission state transitions, performance metrics, and structured failure explanations."""

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
        self.failure_record: Optional[Dict[str, Any]] = None
        self.success_record: Optional[Dict[str, Any]] = None

        # Mission performance metrics
        self.mission_start_time: float = 0.0
        self.travel_distance_m: float = 0.0
        self.last_pose_pos: Optional[Tuple[float, float]] = None
        self.recovery_attempts: int = 0
        self.initial_path_length_m: float = 0.0

    def start_mission(self, now_sec: float, initial_path_length: float = 0.0) -> None:
        """Record mission start timestamp and reset path counters."""
        self.mission_start_time = now_sec
        self.travel_distance_m = 0.0
        self.last_pose_pos = None
        self.recovery_attempts = 0
        self.initial_path_length_m = initial_path_length
        self.failure_record = None
        self.success_record = None

    def update_odometry_distance(self, x: float, y: float) -> None:
        """Accumulate actual physical distance traversed by the robot."""
        if self.state in (MissionState.NAVIGATING, MissionState.REPLANNING, MissionState.RECOVERY_WAIT):
            if self.last_pose_pos is not None:
                dx = x - self.last_pose_pos[0]
                dy = y - self.last_pose_pos[1]
                step = (dx * dx + dy * dy) ** 0.5
                if 0.001 < step < 1.0:  # Ignore stationary jitter or teleport
                    self.travel_distance_m += step
            self.last_pose_pos = (x, y)

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
        elif new_state in (MissionState.IDLE, MissionState.GOAL_SET):
            self.replan_count = 0

        return True

    def record_planning_failure(
        self,
        now_sec: float,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Handle planning failure and check retry budget."""
        self.replan_count += 1
        if self.replan_count >= self.max_replan_retries:
            code = FailureCode.NO_SAFE_PATH
            if "COLLISION" in reason:
                code = FailureCode.GOAL_IN_COLLISION_ZONE
            elif "BLOCKED" in reason:
                code = FailureCode.PATH_BLOCKED
            self.trigger_failure(code, now_sec, context=context)
            return False
        return True

    def trigger_failure(
        self,
        failure_code: str,
        now_sec: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate structured mission failure record and transition to MISSION_FAILED."""
        meta = FAILURE_TAXONOMY.get(failure_code, {
            "category": "NAVIGATION",
            "human_reason": f"Mission failed: {failure_code}",
            "detail": "An unresolvable navigational or system constraint prevented mission completion.",
        })

        ctx = context or {}
        record = {
            "failure_code": failure_code,
            "failure_category": meta["category"],
            "human_reason": meta["human_reason"],
            "detail": meta["detail"],
            "timestamp": round(now_sec, 3),
            "mission_state": "MISSION_FAILED",
            "navigation_state": "FAILED",
            "recovery_state": ctx.get("recovery_state", "NORMAL"),
            "confidence": round(ctx.get("confidence", 1.0), 3),
            "visual_confidence": round(ctx.get("visual_confidence", 1.0), 3),
            "localization_confidence": round(ctx.get("localization_confidence", 1.0), 3),
            "goal": ctx.get("goal"),
            "robot_pose": ctx.get("robot_pose"),
            "distance_to_goal": round(ctx.get("distance_to_goal", 0.0), 3),
            "last_valid_pose": ctx.get("last_valid_pose", ctx.get("robot_pose")),
            "recovery_attempts": ctx.get("recovery_attempts", self.recovery_attempts),
            "recovery_max_attempts": ctx.get("recovery_max_attempts", 3),
            "replan_count": self.replan_count,
            "blocked_regions": ctx.get("blocked_regions", 0),
            "path_status": ctx.get("path_status", "BLOCKED"),
            "sensor_status": ctx.get("sensor_status", {}),
        }

        self.failure_record = record
        self.transition_to(MissionState.MISSION_FAILED, now_sec, f"{failure_code}: {meta['human_reason']}")
        return record

    def trigger_success(
        self,
        now_sec: float,
        final_error_m: float,
        path_length_m: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate structured success report upon reaching goal."""
        duration = max(0.1, now_sec - self.mission_start_time) if self.mission_start_time > 0 else 0.0
        record = {
            "mission_state": "GOAL_REACHED",
            "travel_distance_m": round(self.travel_distance_m, 2),
            "path_length_m": round(path_length_m or self.initial_path_length_m, 2),
            "duration_sec": round(duration, 1),
            "replan_count": self.replan_count,
            "recovery_attempts": self.recovery_attempts,
            "final_error_m": round(final_error_m, 3),
            "average_speed_mps": round(self.travel_distance_m / duration, 2) if duration > 0 else 0.0,
        }
        self.success_record = record
        self.transition_to(MissionState.GOAL_REACHED, now_sec, "GOAL_REACHED_NOMINAL")
        return record

    def reset(self, now_sec: float) -> None:
        """Reset mission state to IDLE."""
        self.state = MissionState.IDLE
        self.state_entry_time = now_sec
        self.replan_count = 0
        self.failure_reason = "NONE"
        self.failure_record = None
        self.success_record = None
        self.travel_distance_m = 0.0
        self.last_pose_pos = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state summary to dictionary."""
        return {
            "mission_state": self.state.to_string(),
            "mission_state_code": int(self.state),
            "replan_count": self.replan_count,
            "failure_reason": self.failure_reason,
            "failure_record": self.failure_record,
            "success_record": self.success_record,
        }
