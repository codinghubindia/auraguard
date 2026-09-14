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
    LOOKAROUND_360_SCAN = 4

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
        reason = failure_reason.upper()
        # Explicit request for 360 lookaround scan
        if "360" in reason or "LOOKAROUND" in reason or "PANORAMA" in reason:
            return RecoveryStrategy.LOOKAROUND_360_SCAN

        # Rule 1: Hardware sensor dropouts or numerical faults -> NEVER move physically
        if "TIMEOUT" in reason and ("IMU" in reason or "WHEEL" in reason):
            return RecoveryStrategy.STOP_AND_RELOCALIZE
        if "NAN_OR_INF" in reason or "SHOCK" in reason:
            return RecoveryStrategy.STOP_AND_RELOCALIZE

        # Rule 2: Path blocked / no safe path -> 360 Lookaround to find open passages
        if "PATH_BLOCKED" in reason or "NO_SAFE_PATH" in reason or "CLEARANCE" in reason:
            return RecoveryStrategy.LOOKAROUND_360_SCAN

        # Rule 3: Obstacle proximity -> backtrack away if checkpoint exists on attempt 1,
        # but on subsequent retry attempts perform 360 lookaround scan for open corridors.
        if "OBSTACLE" in reason:
            if has_checkpoint and attempt_number == 1:
                return RecoveryStrategy.SHORT_BACKTRACK
            return RecoveryStrategy.LOOKAROUND_360_SCAN

        # Rule 4: Visual feature degradation / low inliers
        if "VO_DEGRADED" in reason or "VO_LOW" in reason or "INLIER" in reason:
            if attempt_number == 1:
                return RecoveryStrategy.STOP_AND_RELOCALIZE
            else:
                return RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

        # Rule 5: SLAM localization tracking loss
        if "SLAM_TRACKING_LOST" in reason or "INSUFFICIENT_FEATURES" in reason:
            if has_checkpoint:
                return RecoveryStrategy.SHORT_BACKTRACK
            return RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

        # Rule 6: Dynamic slip / wheel-IMU disagreement
        if "SLIP" in reason or "DISAGREEMENT" in reason or "RESIDUAL" in reason:
            return RecoveryStrategy.STOP_AND_RELOCALIZE

        # Default fallback
        if has_checkpoint and attempt_number <= 2:
            return RecoveryStrategy.SHORT_BACKTRACK
        elif attempt_number == 1:
            return RecoveryStrategy.STOP_AND_RELOCALIZE
        else:
            return RecoveryStrategy.LOOKAROUND_360_SCAN

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

    def plan_360_lookaround(
        self,
        current_yaw: float,
        direction: int = 1,
    ) -> Tuple[float, float, str]:
        """Compute relative rotation and target yaw for a full 360-degree lookaround scan."""
        dyaw = float(direction * 2.0 * np.pi)
        target_yaw = current_yaw + dyaw
        return dyaw, target_yaw, "LOOKAROUND_360_SCAN"

    def evaluate_360_passages(
        self,
        current_pose: Tuple[float, float, float],
        grid_data: Optional[List[int]] = None,
        grid_res: float = 0.05,
        grid_w: int = 0,
        grid_h: int = 0,
        grid_ox: float = 0.0,
        grid_oy: float = 0.0,
        goal_pose: Optional[Tuple[float, float]] = None,
        num_sectors: int = 36,
        max_range_m: float = 3.5,
        vehicle_width_m: float = 0.48,
    ) -> Dict[str, Any]:
        """Evaluate clearance and traversability in 360 degrees around vehicle.

        Scans num_sectors azimuths to identify unobstructed passages, narrow gateways
        (>= vehicle_width_m), and selects the highest-scoring retry exit heading.
        """
        rx, ry, ryaw = current_pose
        res = max(0.01, grid_res)

        passages: List[Dict[str, Any]] = []
        best_heading_rad = ryaw
        best_score = -1e9
        widest_corridor_m = 0.0

        if not grid_data or grid_w <= 0 or grid_h <= 0:
            return {
                "scan_complete": True,
                "sectors_evaluated": num_sectors,
                "best_heading_rad": ryaw,
                "best_heading_deg": round(float(np.degrees(ryaw)), 1),
                "passages": [],
                "widest_corridor_m": 0.0,
                "selected_passage": None,
            }

        goal_angle = None
        if goal_pose is not None:
            gx, gy = goal_pose
            goal_angle = float(np.arctan2(gy - ry, gx - rx))

        # Sector sweep 0..360 deg
        for s in range(num_sectors):
            angle = float(s * (2.0 * np.pi / num_sectors))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Step along ray
            free_dist = 0.0
            step_m = max(0.05, res)
            steps = int(max_range_m / step_m)
            min_side_clearance = max_range_m

            for st in range(1, steps + 1):
                cur_dist = st * step_m
                px = rx + cur_dist * cos_a
                py = ry + cur_dist * sin_a

                cx = int((px - grid_ox) / res)
                cy = int((py - grid_oy) / res)

                if 0 <= cx < grid_w and 0 <= cy < grid_h:
                    occ = grid_data[cy * grid_w + cx]
                    if occ >= self.obstacle_clearance_thresh:
                        break
                else:
                    break

                free_dist = cur_dist

                # Check lateral clearance perpendicular to ray at cur_dist
                perp_cos = -sin_a
                perp_sin = cos_a
                half_w = vehicle_width_m * 0.5
                for side in (-1.0, 1.0):
                    lx = px + side * half_w * perp_cos
                    ly = py + side * half_w * perp_sin
                    lcx = int((lx - grid_ox) / res)
                    lcy = int((ly - grid_oy) / res)
                    if 0 <= lcx < grid_w and 0 <= lcy < grid_h:
                        locc = grid_data[lcy * grid_w + lcx]
                        if locc >= self.obstacle_clearance_thresh:
                            min_side_clearance = min(min_side_clearance, half_w)

            corridor_width = 2.0 * min(free_dist * 0.5, min_side_clearance)
            widest_corridor_m = max(widest_corridor_m, corridor_width)

            # Is passable for vehicle?
            is_passable = (free_dist >= 0.60) and (corridor_width >= vehicle_width_m)
            is_narrow = is_passable and (corridor_width < 0.85)

            # Score this sector
            score = free_dist * 2.0 + corridor_width * 3.0
            if goal_angle is not None:
                # Goal alignment bonus [-1.0 .. +1.0]
                alignment = float(np.cos(angle - goal_angle))
                score += alignment * 4.0

            if is_passable:
                passages.append({
                    "sector_idx": s,
                    "heading_rad": angle,
                    "heading_deg": round(float(np.degrees(angle)), 1),
                    "free_dist_m": round(free_dist, 2),
                    "corridor_width_m": round(corridor_width, 2),
                    "is_narrow_passage": is_narrow,
                    "score": round(score, 2),
                })

                if score > best_score:
                    best_score = score
                    best_heading_rad = angle

        selected = None
        for p in passages:
            if abs(p["heading_rad"] - best_heading_rad) < 1e-3:
                selected = p
                break

        return {
            "scan_complete": True,
            "sectors_evaluated": num_sectors,
            "best_heading_rad": best_heading_rad,
            "best_heading_deg": round(float(np.degrees(best_heading_rad)), 1),
            "passages": passages,
            "widest_corridor_m": round(widest_corridor_m, 2),
            "selected_passage": selected,
        }


