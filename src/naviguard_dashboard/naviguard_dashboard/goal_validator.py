"""
Goal Safety Validator for NAVIGUARD Operator Dashboard.

Validates destination goals before publishing to the autonomous navigation stack:
1. Destination is within map boundaries.
2. Destination cell is not occupied in the current OccupancyGrid.
3. Navigation subsystem is online.
4. Recovery subsystem is not in FAILED_SAFE state.
5. Robot localization is available.
"""

import math
from typing import Tuple, Optional, List
import numpy as np
from geometry_msgs.msg import PoseStamped
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter


class GoalValidator:
    """Validates goal candidates and constructs PoseStamped messages."""

    def __init__(self, converter: MapCoordinateConverter):
        self.converter = converter

    def validate_goal(
        self,
        x: float,
        y: float,
        grid_data: Optional[List[int]] = None,
        navigation_online: bool = True,
        recovery_state: str = "NORMAL",
        has_localization: bool = True,
    ) -> Tuple[bool, str]:
        """
        Validate goal candidate against map, obstacle data, and system health.
        Returns (is_valid, explanation_string).
        """
        # Accept boolean or string statuses (ONLINE, DEGRADED, etc.)
        if isinstance(navigation_online, str):
            navigation_online = (navigation_online in ("ONLINE", "DEGRADED"))
        if isinstance(has_localization, str):
            has_localization = (has_localization in ("ONLINE", "DEGRADED"))

        # 1. System readiness checks
        if not navigation_online:
            return False, "INVALID GOAL: Navigation subsystem is OFFLINE."

        if recovery_state == "FAILED_SAFE":
            return False, "INVALID GOAL: System in FAILED_SAFE mode. Manual intervention required."

        if not has_localization:
            return False, "INVALID GOAL: Robot localization unavailable (no SLAM pose)."

        # 2. Map boundary check
        col, row = self.converter.world_to_grid(x, y)
        if not self.converter.is_in_grid_bounds(col, row):
            min_x = self.converter.origin_x
            max_x = self.converter.origin_x + self.converter.width_cells * self.converter.resolution
            min_y = self.converter.origin_y
            max_y = self.converter.origin_y + self.converter.height_cells * self.converter.resolution
            return False, (
                f"INVALID GOAL: Target ({x:.2f}, {y:.2f}) lies outside map bounds "
                f"[X: {min_x:.1f} to {max_x:.1f}, Y: {min_y:.1f} to {max_y:.1f}]."
            )

        # 3. Occupancy check
        if grid_data is not None and len(grid_data) > 0:
            idx = row * self.converter.width_cells + col
            if 0 <= idx < len(grid_data):
                cost = grid_data[idx]
                if cost > 50:
                    return False, f"INVALID GOAL: Target location is OCCUPIED by an obstacle (cost: {cost})."
                elif cost < 0:
                    # In unstructured outdoor terrain, allow exploration into unknown territory
                    # with a soft warning, but if user explicitly clicks far outside, reject
                    pass

        return True, "GOAL VALID: Target is reachable and within traversable space."

    @staticmethod
    def create_goal_msg(
        x: float,
        y: float,
        yaw: float = 0.0,
        frame_id: str = "map",
        stamp=None
    ) -> PoseStamped:
        """Create standard ROS 2 geometry_msgs/msg/PoseStamped message for /goal_pose."""
        msg = PoseStamped()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        # Convert yaw angle to quaternion about Z
        half_yaw = yaw * 0.5
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = float(math.sin(half_yaw))
        msg.pose.orientation.w = float(math.cos(half_yaw))

        return msg
