"""Waypoint generation and orientation assignment for NAVIGUARD."""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Waypoint:
    x: float
    y: float
    yaw: float
    index: int


class WaypointGenerator:
    """Generates evenly spaced waypoints with smooth headings along a planned path."""

    def __init__(
        self,
        target_spacing_m: float = 0.35,
        min_spacing_m: float = 0.15,
        max_spacing_m: float = 0.50,
    ) -> None:
        self.target_spacing_m = target_spacing_m
        self.min_spacing_m = min_spacing_m
        self.max_spacing_m = max_spacing_m

    def generate(
        self,
        path: List[Tuple[float, float]],
        final_yaw: Optional[float] = None,
    ) -> List[Waypoint]:
        """Discretize continuous path into uniformly spaced waypoints with assigned headings."""
        if not path:
            return []

        if len(path) == 1:
            yaw = final_yaw if final_yaw is not None else 0.0
            return [Waypoint(x=path[0][0], y=path[0][1], yaw=yaw, index=0)]

        sampled_points: List[Tuple[float, float]] = [path[0]]

        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            seg_dx = p2[0] - p1[0]
            seg_dy = p2[1] - p1[1]
            seg_len = math.hypot(seg_dx, seg_dy)

            if seg_len <= 1e-4:
                continue

            num_steps = max(1, int(round(seg_len / self.target_spacing_m)))
            step_len = seg_len / num_steps

            for s in range(1, num_steps + 1):
                t = (s * step_len) / seg_len
                px = p1[0] + t * seg_dx
                py = p1[1] + t * seg_dy

                # Check spacing against last added point
                last = sampled_points[-1]
                if math.hypot(px - last[0], py - last[1]) >= self.min_spacing_m:
                    sampled_points.append((px, py))

        # Ensure goal point is included
        goal_pt = path[-1]
        if math.hypot(goal_pt[0] - sampled_points[-1][0], goal_pt[1] - sampled_points[-1][1]) > 0.05:
            sampled_points.append(goal_pt)
        else:
            sampled_points[-1] = goal_pt

        # Assign orientations
        waypoints: List[Waypoint] = []
        for i in range(len(sampled_points)):
            pt = sampled_points[i]
            if i < len(sampled_points) - 1:
                next_pt = sampled_points[i + 1]
                yaw = math.atan2(next_pt[1] - pt[1], next_pt[0] - pt[0])
            else:
                if final_yaw is not None:
                    yaw = final_yaw
                elif len(waypoints) > 0:
                    yaw = waypoints[-1].yaw
                else:
                    yaw = 0.0

            waypoints.append(Waypoint(x=pt[0], y=pt[1], yaw=yaw, index=i))

        return waypoints
