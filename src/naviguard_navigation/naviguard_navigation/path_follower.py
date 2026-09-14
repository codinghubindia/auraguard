"""Pure pursuit path following controller for differential-drive UGV."""

import math
from typing import List, Tuple, Optional
from naviguard_navigation.waypoint_generator import Waypoint


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PathFollower:
    """Computes bounded linear and angular velocities to track waypoint path."""

    def __init__(
        self,
        lookahead_distance_m: float = 0.40,
        max_linear_velocity: float = 0.25,
        min_linear_velocity: float = 0.05,
        max_angular_velocity: float = 0.35,
        kp_heading: float = 1.2,
        kp_cross_track: float = 0.5,
        turn_in_place_angle_threshold: float = 0.75,  # ~43 deg
    ) -> None:
        self.lookahead_distance_m = lookahead_distance_m
        self.max_linear_velocity = max_linear_velocity
        self.min_linear_velocity = min_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.kp_heading = kp_heading
        self.kp_cross_track = kp_cross_track
        self.turn_in_place_angle_threshold = turn_in_place_angle_threshold

        self.current_waypoint_idx: int = 0

    def reset(self) -> None:
        """Reset path follower state."""
        self.current_waypoint_idx = 0

    def compute_commands(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        waypoints: List[Waypoint],
        speed_scale: float = 1.0,
    ) -> Tuple[float, float, Optional[Waypoint], float]:
        """Compute (vx, wz, lookahead_waypoint, cross_track_error)."""
        if not waypoints:
            return 0.0, 0.0, None, 0.0

        # 1. Update current closest waypoint (search forward)
        best_idx = self.current_waypoint_idx
        min_dist_sq = float('inf')
        search_end = min(len(waypoints), self.current_waypoint_idx + 8)

        for i in range(self.current_waypoint_idx, search_end):
            wp = waypoints[i]
            d2 = (wp.x - robot_x) ** 2 + (wp.y - robot_y) ** 2
            if d2 < min_dist_sq:
                min_dist_sq = d2
                best_idx = i

        self.current_waypoint_idx = best_idx

        # 2. Find lookahead point
        lookahead_wp = waypoints[-1]
        for i in range(self.current_waypoint_idx, len(waypoints)):
            wp = waypoints[i]
            d = math.hypot(wp.x - robot_x, wp.y - robot_y)
            if d >= self.lookahead_distance_m:
                lookahead_wp = wp
                break

        # 3. Calculate heading error to lookahead point
        target_heading = math.atan2(lookahead_wp.y - robot_y, lookahead_wp.x - robot_x)
        heading_error = normalize_angle(target_heading - robot_yaw)

        # 4. Calculate signed cross-track error relative to current path segment
        curr_wp = waypoints[self.current_waypoint_idx]
        dx = robot_x - curr_wp.x
        dy = robot_y - curr_wp.y
        # Vector along path
        path_yaw = curr_wp.yaw
        cross_track_error = -math.sin(path_yaw) * dx + math.cos(path_yaw) * dy

        # 5. Controller calculations
        # Angular control
        wz = self.kp_heading * heading_error - self.kp_cross_track * cross_track_error
        wz = max(-self.max_angular_velocity, min(self.max_angular_velocity, wz))

        # Linear control
        dist_to_goal = math.hypot(waypoints[-1].x - robot_x, waypoints[-1].y - robot_y)
        v_limit = self.max_linear_velocity * max(0.2, min(1.0, speed_scale))

        if abs(heading_error) > self.turn_in_place_angle_threshold:
            # Turn in place to align with path
            vx = 0.0
        else:
            # Scale linear speed with heading alignment and goal proximity
            cos_factor = max(0.0, math.cos(heading_error))
            vx = v_limit * cos_factor

            # Slow down on final approach (within 1.0 m)
            if dist_to_goal < 1.0:
                slowdown = max(0.3, dist_to_goal / 1.0)
                vx *= slowdown

            vx = max(self.min_linear_velocity, min(v_limit, vx))

        return float(vx), float(wz), lookahead_wp, float(abs(cross_track_error))
