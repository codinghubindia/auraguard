"""Collision-safe path smoothing using line-of-sight shortcutting for NAVIGUARD."""

from typing import List, Tuple
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid


class PathSmoother:
    """Smoothes raw grid paths while rigorously guaranteeing collision-freedom."""

    def __init__(self, max_shortcut_distance_m: float = 3.0) -> None:
        self.max_shortcut_distance_m = max_shortcut_distance_m

    def smooth(
        self,
        raw_path: List[Tuple[float, float]],
        grid: NavigationOccupancyGrid,
    ) -> List[Tuple[float, float]]:
        """Perform collision-safe line-of-sight shortcutting on raw waypoint path."""
        if len(raw_path) <= 2:
            return list(raw_path)

        # 1. Greedy string-pulling shortcutting
        smoothed: List[Tuple[float, float]] = [raw_path[0]]
        current_idx = 0

        while current_idx < len(raw_path) - 1:
            furthest_idx = current_idx + 1
            # Look ahead for furthest visible collision-free point
            for target_idx in range(len(raw_path) - 1, current_idx, -1):
                p1 = raw_path[current_idx]
                p2 = raw_path[target_idx]

                # Check maximum shortcut length
                import math
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > self.max_shortcut_distance_m:
                    continue

                if grid.is_segment_collision_free(p1[0], p1[1], p2[0], p2[1]) and grid.is_swept_footprint_collision_free(p1, p2):
                    furthest_idx = target_idx
                    break

            smoothed.append(raw_path[furthest_idx])
            current_idx = furthest_idx

        # 2. Strict collision verification pass: ensure all points in smoothed path remain safe
        verified: List[Tuple[float, float]] = [smoothed[0]]
        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i + 1]
            if grid.is_segment_collision_free(p1[0], p1[1], p2[0], p2[1]) and grid.is_swept_footprint_collision_free(p1, p2):
                verified.append(p2)
            else:
                # If smoothing failed safety test, reject and retain raw path
                return list(raw_path)

        return verified
