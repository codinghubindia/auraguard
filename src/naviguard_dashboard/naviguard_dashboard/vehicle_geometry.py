"""NAVIGUARD Dashboard Vehicle Geometry Interface.

Shares identical parameters with naviguard_navigation.vehicle_geometry derived from URDF.
"""

try:
    from naviguard_navigation.vehicle_geometry import VehicleGeometry
except ImportError:
    import math
    from typing import List, Tuple, Dict, Any, Optional

    class VehicleGeometry:
        LENGTH_M: float = 0.56
        WIDTH_M: float = 0.48
        HEIGHT_M: float = 0.26
        WHEELBASE_M: float = 0.30
        TRACK_WIDTH_M: float = 0.42
        WHEEL_RADIUS_M: float = 0.10
        WHEEL_WIDTH_M: float = 0.06
        GROUND_CLEARANCE_M: float = 0.04

        INSCRIBED_RADIUS_M: float = 0.2400
        CIRCUMSCRIBED_RADIUS_M: float = 0.3688
        SAFETY_MARGIN_M: float = 0.10
        MINIMUM_CLEARANCE_M: float = 0.05
        INFLATION_MARGIN_M: float = 0.10

        REQUIRED_NOMINAL_PASSAGE_M: float = 0.68
        REQUIRED_TIGHT_PASSAGE_M: float = 0.58
        REQUIRED_TURN_IN_PLACE_DIAM_M: float = 0.94

        TOTAL_LENGTH_M: float = LENGTH_M
        TOTAL_WIDTH_M: float = WIDTH_M
        CHASSIS_HEIGHT_M: float = HEIGHT_M
        NOMINAL_PASSAGE_WIDTH_M: float = REQUIRED_NOMINAL_PASSAGE_M
        TIGHT_PASSAGE_LIMIT_M: float = REQUIRED_TIGHT_PASSAGE_M
        MIN_TURN_DIAMETER_M: float = REQUIRED_TURN_IN_PLACE_DIAM_M

        BASE_FOOTPRINT: List[Tuple[float, float]] = [
            (0.28, 0.24),
            (-0.28, 0.24),
            (-0.28, -0.24),
            (0.28, -0.24),
        ]

        @classmethod
        def get_base_footprint(cls) -> List[Tuple[float, float]]:
            return list(cls.BASE_FOOTPRINT)

        @classmethod
        def evaluate_passage(
            cls,
            available_clear_width_m: float,
            heading_change_rad: float = 0.0,
            safety_margin_m: Optional[float] = None,
        ) -> Tuple[bool, str, Dict[str, Any]]:
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

            if abs(heading_change_rad) > math.radians(35.0) and available_clear_width_m < (2.0 * cls.CIRCUMSCRIBED_RADIUS_M + 2.0 * cls.MINIMUM_CLEARANCE_M):
                diag["passage_status"] = "BLOCKED"
                diag["reason"] = "INSUFFICIENT_TURNING_SPACE"
                return False, "BLOCKED", diag

            if available_clear_width_m >= required_nominal:
                diag["passage_status"] = "SAFE"
                diag["reason"] = "ADEQUATE_CORRIDOR_CLEARANCE"
                return True, "SAFE", diag
            elif available_clear_width_m >= required_tight:
                diag["passage_status"] = "TIGHT"
                diag["reason"] = "TIGHT_CORRIDOR_PROCEED_CAUTIOUSLY"
                return True, "TIGHT", diag
            else:
                diag["passage_status"] = "BLOCKED"
                diag["reason"] = "PASSAGE_BELOW_VEHICLE_WIDTH_AND_SAFETY_MARGIN"
                return False, "BLOCKED", diag
