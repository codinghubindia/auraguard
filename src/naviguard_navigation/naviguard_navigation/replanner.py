"""Replanning monitor and obstacle change detection for NAVIGUARD."""

import math
from typing import List, Tuple, Optional
from naviguard_navigation.waypoint_generator import Waypoint
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid


class Replanner:
    """Monitors active path collision validity and tracking deviations to trigger replans."""

    def __init__(
        self,
        min_replan_interval_sec: float = 1.0,
        max_path_deviation_m: float = 0.60,
    ) -> None:
        self.min_replan_interval_sec = min_replan_interval_sec
        self.max_path_deviation_m = max_path_deviation_m
        self.last_replan_time: float = 0.0

    def find_blocked_segment(
        self,
        waypoints: List[Waypoint],
        current_idx: int,
        grid: NavigationOccupancyGrid,
    ) -> Optional[Tuple[float, float, int]]:
        """Return (mid_x, mid_y, waypoint_index) of the first blocked segment if any."""
        if not waypoints or not grid.is_initialized:
            return None
        for i in range(current_idx, len(waypoints) - 1):
            p1 = (waypoints[i].x, waypoints[i].y)
            p2 = (waypoints[i + 1].x, waypoints[i + 1].y)
            if not grid.is_segment_collision_free(p1[0], p1[1], p2[0], p2[1]):
                mx = 0.5 * (p1[0] + p2[0])
                my = 0.5 * (p1[1] + p2[1])
                return mx, my, i
        return None

    def should_replan_due_to_obstacle(
        self,
        waypoints: List[Waypoint],
        current_idx: int,
        grid: NavigationOccupancyGrid,
        now_sec: float,
    ) -> Tuple[bool, str]:
        """Check if any upcoming waypoint or segment on the path intersects an inflated obstacle."""
        if not waypoints or not grid.is_initialized:
            return False, "NO_PATH_OR_GRID"

        if (now_sec - self.last_replan_time) < self.min_replan_interval_sec:
            return False, "COOLDOWN"

        blocked = self.find_blocked_segment(waypoints, current_idx, grid)
        if blocked is not None:
            self.last_replan_time = now_sec
            return True, f"PATH_BLOCKED_AT_WAYPOINT_{blocked[2]}_TO_{blocked[2]+1}"

        return False, "PATH_CLEAR"

    def should_replan_due_to_deviation(
        self,
        cross_track_error: float,
        now_sec: float,
    ) -> Tuple[bool, str]:
        """Check if robot has deviated excessively from the planned path."""
        if (now_sec - self.last_replan_time) < self.min_replan_interval_sec:
            return False, "COOLDOWN"

        if cross_track_error > self.max_path_deviation_m:
            self.last_replan_time = now_sec
            return True, f"EXCESSIVE_CROSS_TRACK_DEVIATION_{cross_track_error:.2f}M"

        return False, "TRACKING_NOMINAL"

    def record_replan(self, now_sec: float) -> None:
        """Manually record a replanning event timestamp."""
        self.last_replan_time = now_sec
