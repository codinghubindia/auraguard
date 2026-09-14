"""Closed-Loop Velocity Controller for Recovery Maneuvers.

Generates safe, bounded Twist commands for active stopping, slow reverse backtracking,
and controlled rotation sweeps.
"""

from typing import Optional, Tuple
import numpy as np


class RecoveryController:
    """Computes bounded Twist commands for recovery execution."""

    def __init__(self) -> None:
        self.max_reverse_speed_mps = 0.15
        self.max_rotation_speed_radps = 0.25
        self.goal_tolerance_dist_m = 0.12
        self.goal_tolerance_yaw_rad = 0.06

    def compute_stop(self) -> Tuple[float, float]:
        """Produce safe zero velocity."""
        return 0.0, 0.0

    def compute_backtrack_step(
        self,
        current_pose: Tuple[float, float, float],
        target_waypoint: Tuple[float, float, float],
    ) -> Tuple[float, float, bool]:
        """Compute reverse linear velocity and proportional steering toward target."""
        dx = target_waypoint[0] - current_pose[0]
        dy = target_waypoint[1] - current_pose[1]
        dist = float(np.hypot(dx, dy))

        if dist <= self.goal_tolerance_dist_m:
            return 0.0, 0.0, True  # Reached

        # Heading toward waypoint
        target_heading = float(np.arctan2(dy, dx))
        # When reversing, the rear of the robot (-x axis) points toward the target
        reverse_heading = float(np.arctan2(-dy, -dx))

        heading_err = float(np.arctan2(
            np.sin(reverse_heading - current_pose[2]),
            np.cos(reverse_heading - current_pose[2]),
        ))

        # Proportional velocity commands
        vx = -float(np.clip(dist * 0.4, 0.06, self.max_reverse_speed_mps))
        wz = float(np.clip(heading_err * 0.8, -self.max_rotation_speed_radps, self.max_rotation_speed_radps))

        return vx, wz, False

    def compute_rotation_step(
        self,
        current_yaw: float,
        target_yaw: float,
    ) -> Tuple[float, float, bool]:
        """Compute bounded in-place rotation."""
        yaw_err = float(np.arctan2(
            np.sin(target_yaw - current_yaw),
            np.cos(target_yaw - current_yaw),
        ))

        if abs(yaw_err) <= self.goal_tolerance_yaw_rad:
            return 0.0, 0.0, True  # Reached

        wz = float(np.sign(yaw_err) * np.clip(abs(yaw_err) * 0.6, 0.08, self.max_rotation_speed_radps))
        return 0.0, wz, False
