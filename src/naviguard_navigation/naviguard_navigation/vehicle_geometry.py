"""NAVIGUARD Centralized Vehicle Geometry Module.

Source of Truth:
Derived strictly from URDF and XACRO specifications in naviguard_description:
- Chassis: 0.50m (L) x 0.32m (W) x 0.16m (H)
- Wheelbase: 0.30m, Track width: 0.42m
- Wheel radius: 0.10m, Wheel width: 0.06m
- Wheel outer bounds: y = +/- 0.24m (Total width = 0.48m)
- Bumpers: Front x = +0.28m, Rear x = -0.28m (Total length = 0.56m)
"""

import math
from typing import List, Tuple, Dict, Any, Optional


class VehicleGeometry:
    """Centralized vehicle geometry specification and footprint collision analyzer."""

    # 1. Exact Physical Dimensions (meters)
    LENGTH_M: float = 0.56
    WIDTH_M: float = 0.48
    HEIGHT_M: float = 0.26
    WHEELBASE_M: float = 0.30
    TRACK_WIDTH_M: float = 0.42
    WHEEL_RADIUS_M: float = 0.10
    WHEEL_WIDTH_M: float = 0.06
    GROUND_CLEARANCE_M: float = 0.04

    FRONT_OVERHANG_M: float = 0.13
    REAR_OVERHANG_M: float = 0.13

    # Inscribed and Circumscribed Radii
    INSCRIBED_RADIUS_M: float = 0.2400     # Half of width
    CIRCUMSCRIBED_RADIUS_M: float = 0.3688 # sqrt(0.28^2 + 0.24^2)

    # Operational & Safety Margins
    SAFETY_MARGIN_M: float = 0.10
    MINIMUM_CLEARANCE_M: float = 0.05
    INFLATION_MARGIN_M: float = 0.10

    # Required Passages
    REQUIRED_NOMINAL_PASSAGE_M: float = 0.68   # 0.48 + 2 * 0.10
    REQUIRED_TIGHT_PASSAGE_M: float = 0.58     # 0.48 + 2 * 0.05
    REQUIRED_TURN_IN_PLACE_DIAM_M: float = 0.94 # 2 * (0.3688 + 0.10)

    # Convenience Aliases
    TOTAL_LENGTH_M: float = LENGTH_M
    TOTAL_WIDTH_M: float = WIDTH_M
    CHASSIS_HEIGHT_M: float = HEIGHT_M
    NOMINAL_PASSAGE_WIDTH_M: float = REQUIRED_NOMINAL_PASSAGE_M
    TIGHT_PASSAGE_LIMIT_M: float = REQUIRED_TIGHT_PASSAGE_M
    MIN_TURN_DIAMETER_M: float = REQUIRED_TURN_IN_PLACE_DIAM_M

    # Base Footprint Polygon (Front-Left, Rear-Left, Rear-Right, Front-Right)
    BASE_FOOTPRINT: List[Tuple[float, float]] = [
        (0.28, 0.24),
        (-0.28, 0.24),
        (-0.28, -0.24),
        (0.28, -0.24),
    ]

    def __init__(
        self,
        safety_margin_m: float = 0.10,
        minimum_clearance_m: float = 0.05,
    ) -> None:
        self.safety_margin_m = safety_margin_m
        self.minimum_clearance_m = minimum_clearance_m

    @classmethod
    def get_base_footprint(cls) -> List[Tuple[float, float]]:
        """Return 4-corner footprint polygon relative to base_link / base_footprint."""
        return list(cls.BASE_FOOTPRINT)

    @classmethod
    def get_oriented_footprint(cls, x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
        """Transform base footprint polygon to global frame given (x, y, yaw)."""
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        oriented: List[Tuple[float, float]] = []
        for bx, by in cls.BASE_FOOTPRINT:
            gx = x + bx * cos_yaw - by * sin_yaw
            gy = y + bx * sin_yaw + by * cos_yaw
            oriented.append((gx, gy))
        return oriented

    @classmethod
    def get_footprint_sample_points(cls, x: float, y: float, yaw: float) -> List[Tuple[float, float]]:
        """Return boundary and interior sample points representing the complete footprint.
        
        Samples:
        - 4 outer corners
        - 4 midpoints of edges
        - 4 wheel center contacts
        - 1 center point
        """
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        # Local sample points (corners, edge midpoints, wheel hubs, center)
        local_samples = [
            (0.0, 0.0),            # Center
            (0.28, 0.24),          # Front-Left corner
            (0.28, -0.24),         # Front-Right corner
            (-0.28, 0.24),         # Rear-Left corner
            (-0.28, -0.24),        # Rear-Right corner
            (0.28, 0.0),           # Front bumper center
            (-0.28, 0.0),          # Rear bumper center
            (0.0, 0.24),           # Left side center
            (0.0, -0.24),          # Right side center
            (0.15, 0.21),          # Front-Left wheel center
            (0.15, -0.21),         # Front-Right wheel center
            (-0.15, 0.21),         # Rear-Left wheel center
            (-0.15, -0.21),        # Rear-Right wheel center
        ]

        global_pts = []
        for lx, ly in local_samples:
            gx = x + lx * cos_yaw - ly * sin_yaw
            gy = y + lx * sin_yaw + ly * cos_yaw
            global_pts.append((gx, gy))
        return global_pts

    @classmethod
    def is_footprint_collision_free(cls, x: float, y: float, yaw: float, grid: Any) -> bool:
        """Verify whether the complete physical UGV footprint at (x, y, yaw) is collision-free."""
        if not grid.is_initialized:
            return False

        samples = cls.get_footprint_sample_points(x, y, yaw)
        for gx, gy in samples:
            pt = grid.world_to_map(gx, gy)
            if pt is None:
                return False  # Footprint extends outside map
            if grid.is_lethal(pt[0], pt[1]):
                return False  # Collision detected with footprint

        return True

    @classmethod
    def is_swept_footprint_collision_free(
        cls,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        grid: Any,
        step_m: float = 0.08,
    ) -> bool:
        """Verify that the swept vehicle footprint along the motion segment p1 -> p2 remains safe."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 0.01:
            yaw = 0.0
            return cls.is_footprint_collision_free(p1[0], p1[1], yaw, grid)

        yaw = math.atan2(dy, dx)
        num_steps = max(2, int(math.ceil(dist / step_m)))

        for i in range(num_steps + 1):
            alpha = float(i) / float(num_steps)
            sx = p1[0] + alpha * dx
            sy = p1[1] + alpha * dy
            if not cls.is_footprint_collision_free(sx, sy, yaw, grid):
                return False

        return True

    @classmethod
    def evaluate_passage(
        cls,
        available_clear_width_m: float,
        heading_change_rad: float = 0.0,
        safety_margin_m: Optional[float] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate passage width against vehicle dimensions and turning space.
        
        Returns:
            can_fit (bool): Whether vehicle can physically navigate this passage.
            status (str): 'SAFE', 'TIGHT', or 'BLOCKED'.
            diagnostics (dict): Structured parameters for UI display and failure logs.
        """
        margin = safety_margin_m if safety_margin_m is not None else cls.SAFETY_MARGIN_M
        required_nominal = cls.WIDTH_M + 2.0 * margin
        required_tight = cls.WIDTH_M + 2.0 * cls.MINIMUM_CLEARANCE_M

        diag = {
            "vehicle_width": cls.WIDTH_M,
            "vehicle_length": cls.LENGTH_M,
            "safety_margin": margin,
            "required_clear_width": round(required_nominal, 3),
            "available_clear_width": round(available_clear_width_m, 3),
            "clearance_margin": round(available_clear_width_m - cls.WIDTH_M, 3),
            "heading_change_deg": round(math.degrees(abs(heading_change_rad)), 1),
            "turning_space_required_m": round(cls.REQUIRED_TURN_IN_PLACE_DIAM_M, 3) if abs(heading_change_rad) > math.radians(35.0) else round(required_nominal, 3),
        }

        # Check turn space feasibility if high-angle turn
        is_sharp_turn = abs(heading_change_rad) > math.radians(35.0)
        can_turn = not (is_sharp_turn and available_clear_width_m < (2.0 * cls.CIRCUMSCRIBED_RADIUS_M + 2.0 * cls.MINIMUM_CLEARANCE_M))
        diag["can_turn"] = can_turn
        diag["turning_diameter_available_m"] = round(available_clear_width_m, 3)
        diag["available_width_m"] = diag["available_clear_width"]
        diag["required_width_m"] = diag["required_clear_width"]
        diag["clearance_margin_m"] = diag["clearance_margin"]

        if not can_turn:
            diag["passage_status"] = "BLOCKED"
            diag["reason"] = "INSUFFICIENT_TURNING_SPACE"
            diag["can_fit"] = False
            return False, "BLOCKED", diag

        if available_clear_width_m >= required_nominal:
            diag["passage_status"] = "SAFE"
            diag["reason"] = "ADEQUATE_CORRIDOR_CLEARANCE"
            diag["can_fit"] = True
            return True, "SAFE", diag
        elif available_clear_width_m >= required_tight:
            diag["passage_status"] = "TIGHT"
            diag["reason"] = "TIGHT_CORRIDOR_PROCEED_CAUTIOUSLY"
            diag["can_fit"] = True
            return True, "TIGHT", diag
        else:
            diag["passage_status"] = "BLOCKED"
            diag["reason"] = "PASSAGE_BELOW_VEHICLE_WIDTH_AND_SAFETY_MARGIN"
            diag["can_fit"] = False
            return False, "BLOCKED", diag
