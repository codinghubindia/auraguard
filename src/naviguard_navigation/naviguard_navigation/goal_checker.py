"""Goal reached verification with spatial and continuous dwell validation."""

import math
from typing import Optional
from naviguard_navigation.goal_manager import NavigationGoal
from naviguard_navigation.path_follower import normalize_angle


class GoalChecker:
    """Verifies that the UGV has arrived safely and settled at the target destination."""

    def __init__(
        self,
        xy_tolerance_m: float = 0.20,
        yaw_tolerance_rad: float = 0.25,
        required_dwell_sec: float = 1.0,
        standoff_tolerance_m: float = 0.85,
    ) -> None:
        self.xy_tolerance_m = xy_tolerance_m
        self.yaw_tolerance_rad = yaw_tolerance_rad
        self.required_dwell_sec = required_dwell_sec
        self.standoff_tolerance_m = standoff_tolerance_m

        self.dwell_start_time: Optional[float] = None

    def reset(self) -> None:
        """Reset dwell timer."""
        self.dwell_start_time = None

    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        goal: Optional[NavigationGoal],
        now_sec: float,
        is_obstructed: bool = False,
    ) -> bool:
        """Check if robot is within goal tolerances for required continuous dwell duration.
        
        If is_obstructed=True (an obstacle is placed directly at or in front of the goal),
        permits safe arrival at standoff distance (up to standoff_tolerance_m) without collision.
        """
        if goal is None:
            self.reset()
            return False

        dist = math.hypot(goal.x - robot_x, goal.y - robot_y)
        allowed_dist = self.standoff_tolerance_m if is_obstructed else self.xy_tolerance_m
        if dist > allowed_dist:
            self.reset()
            return False

        if goal.yaw is not None and not is_obstructed:
            yaw_err = abs(normalize_angle(goal.yaw - robot_yaw))
            if yaw_err > self.yaw_tolerance_rad:
                self.reset()
                return False

        # Tolerances met; check dwell (if obstructed, dwell is reduced to 0.4s for prompt arrival)
        dwell_target = min(0.4, self.required_dwell_sec) if is_obstructed else self.required_dwell_sec
        if self.dwell_start_time is None:
            self.dwell_start_time = now_sec
            return False

        elapsed_dwell = now_sec - self.dwell_start_time
        return elapsed_dwell >= dwell_target
