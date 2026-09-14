"""Recovery Strategy Policy and Bounded Motion Trajectory Planner.

Selects bounded recovery maneuvers according to root failure diagnostics and
computes obstacle-cleared backtrack paths and rotation sweeps.
"""

from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from naviguard_recovery.trusted_state_manager import TrustedCheckpoint


class RecoveryStrategy(IntEnum):
    """Supported closed-loop recovery action types."""
    STOP_AND_RELOCALIZE = 1
    ROTATE_FOR_VISUAL_REACQUISITION = 2
    SHORT_BACKTRACK = 3

    def to_string(self) -> str:
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "RecoveryStrategy":
        clean_name = name.strip().upper()
        if clean_name in cls.__members__:
            return cls.__members__[clean_name]
        raise ValueError(f"Unknown RecoveryStrategy: {name}")


class RecoveryPlanner:
    """Selects recovery strategy and computes collision-free trajectories."""

    def __init__(self) -> None:
        self.max_rotation_angle_rad = 0.65  # ~37 deg
        self.max_backtrack_dist_m = 2.50
        self.obstacle_clearance_thresh = 50

    def select_strategy(
        self,
        failure_reason: str,
        attempt_number: int,
        has_checkpoint: bool,
    ) -> RecoveryStrategy:
        """Explicit policy mapping root failure reason to bounded recovery strategy."""
        reason = failure_reason.upper()

        # Rule 1: Hardware sensor dropouts or numerical faults -> NEVER move physically
        if "TIMEOUT" in reason and ("IMU" in reason or "WHEEL" in reason):
            return RecoveryStrategy.STOP_AND_RELOCALIZE
        if "NAN_OR_INF" in reason or "SHOCK" in reason:
            return RecoveryStrategy.STOP_AND_RELOCALIZE

        # Rule 2: Obstacle proximity -> backtrack away if checkpoint exists
        if "OBSTACLE" in reason:
            if has_checkpoint:
                return RecoveryStrategy.SHORT_BACKTRACK
            return RecoveryStrategy.STOP_AND_RELOCALIZE

        # Rule 3: Visual feature degradation / low inliers
        if "VO_DEGRADED" in reason or "VO_LOW" in reason or "INLIER" in reason:
            if attempt_number == 1:
                return RecoveryStrategy.STOP_AND_RELOCALIZE
            else:
                return RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

        # Rule 4: SLAM localization tracking loss
        if "SLAM_TRACKING_LOST" in reason or "INSUFFICIENT_FEATURES" in reason:
            if has_checkpoint:
                return RecoveryStrategy.SHORT_BACKTRACK
            return RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

        # Rule 5: Dynamic slip / wheel-IMU disagreement
        if "SLIP" in reason or "DISAGREEMENT" in reason or "RESIDUAL" in reason:
            return RecoveryStrategy.STOP_AND_RELOCALIZE

        # Default fallback
        if has_checkpoint and attempt_number <= 2:
            return RecoveryStrategy.SHORT_BACKTRACK
        elif attempt_number == 1:
            return RecoveryStrategy.STOP_AND_RELOCALIZE
        else:
            return RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

    def plan_backtrack(
        self,
        current_pose: Tuple[float, float, float],
        target_checkpoint: TrustedCheckpoint,
        grid_data: Optional[List[int]] = None,
        grid_res: float = 0.05,
        grid_w: int = 0,
        grid_h: int = 0,
        grid_ox: float = 0.0,
        grid_oy: float = 0.0,
    ) -> Tuple[bool, List[Tuple[float, float, float]], str]:
        """Compute discrete collision-cleared backtrack waypoints."""
        dx = target_checkpoint.map_pose[0] - current_pose[0]
        dy = target_checkpoint.map_pose[1] - current_pose[1]
        dist = float(np.hypot(dx, dy))

        if dist > self.max_backtrack_dist_m:
            return False, [], f"BACKTRACK_DIST_EXCEEDS_LIMIT_{dist:.2f}M"

        # Interpolate waypoints every 0.1m
        num_steps = max(2, int(np.ceil(dist / 0.10)))
        waypoints: List[Tuple[float, float, float]] = []

        for i in range(num_steps + 1):
            alpha = float(i) / float(num_steps)
            wx = current_pose[0] + alpha * dx
            wy = current_pose[1] + alpha * dy
            wyaw = current_pose[2]

            # Obstacle check along path
            if grid_data and grid_w > 0 and grid_h > 0:
                cx = int((wx - grid_ox) / max(0.01, grid_res))
                cy = int((wy - grid_oy) / max(0.01, grid_res))
                if 0 <= cx < grid_w and 0 <= cy < grid_h:
                    occ = grid_data[cy * grid_w + cx]
                    if occ >= self.obstacle_clearance_thresh:
                        return False, [], f"OBSTACLE_INTERSECTS_BACKTRACK_PATH_OCC_{occ}"

            waypoints.append((wx, wy, wyaw))

        return True, waypoints, "BACKTRACK_PATH_CLEARED"

    def plan_rotation_sweep(
        self,
        current_yaw: float,
        direction: int = 1,  # +1 for CCW, -1 for CW
    ) -> Tuple[float, float]:
        """Compute relative and target yaw for visual reacquisition."""
        delta_yaw = direction * self.max_rotation_angle_rad
        target_yaw = current_yaw + delta_yaw
        return delta_yaw, target_yaw

    def plan_rotation_scan(
        self,
        current_yaw: float,
        attempt_number: int = 1,
    ) -> Tuple[float, float, str]:
        """Compute structured scan angle and direction depending on attempt number.

        Attempt 1: Controlled CCW visual scan (+35 deg).
        Attempt 2: Controlled CW visual scan with wider swing (-70 deg).
        Attempt 3+: Broad visual scan (+90 deg).
        """
        if attempt_number == 1:
            dyaw = 0.60   # ~+34.4 deg
            direction_name = "SCAN_CCW_35DEG"
        elif attempt_number == 2:
            dyaw = -1.20  # ~-68.8 deg
            direction_name = "SCAN_CW_70DEG"
        else:
            dyaw = 1.57   # ~+90.0 deg
            direction_name = "SCAN_WIDE_90DEG"

        target_yaw = current_yaw + dyaw
        return dyaw, target_yaw, direction_name

