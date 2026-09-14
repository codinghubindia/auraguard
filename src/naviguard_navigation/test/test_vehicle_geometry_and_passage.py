"""Unit and regression tests for centralized UGV vehicle geometry and corridor passage evaluation."""

import math
import pytest
from naviguard_navigation.vehicle_geometry import VehicleGeometry


def test_urdf_geometry_constants():
    """Verify that centralized geometry constants strictly match URDF and XACRO specifications."""
    # Chassis: 0.50 x 0.32 x 0.16; Wheels at y=+/-0.21, width 0.06 -> outer y=+/-0.24 -> width=0.48m
    assert VehicleGeometry.WIDTH_M == 0.48
    assert VehicleGeometry.TOTAL_WIDTH_M == 0.48
    # Bumpers at x=+/-0.28 -> total length = 0.56m
    assert VehicleGeometry.LENGTH_M == 0.56
    assert VehicleGeometry.TOTAL_LENGTH_M == 0.56

    assert VehicleGeometry.WHEELBASE_M == 0.30
    assert VehicleGeometry.TRACK_WIDTH_M == 0.42
    assert VehicleGeometry.WHEEL_RADIUS_M == 0.10
    assert VehicleGeometry.WHEEL_WIDTH_M == 0.06

    # Inscribed and circumscribed radii
    assert VehicleGeometry.INSCRIBED_RADIUS_M == pytest.approx(0.24, abs=1e-4)
    expected_circumscribed = math.hypot(0.28, 0.24)
    assert VehicleGeometry.CIRCUMSCRIBED_RADIUS_M == pytest.approx(expected_circumscribed, abs=1e-4)

    # Operational margins and passage limits
    assert VehicleGeometry.SAFETY_MARGIN_M == 0.10
    assert VehicleGeometry.MINIMUM_CLEARANCE_M == 0.05
    assert VehicleGeometry.REQUIRED_NOMINAL_PASSAGE_M == pytest.approx(0.68, abs=1e-4)
    assert VehicleGeometry.REQUIRED_TIGHT_PASSAGE_M == pytest.approx(0.58, abs=1e-4)
    assert VehicleGeometry.REQUIRED_TURN_IN_PLACE_DIAM_M == pytest.approx(0.94, abs=0.01)


def test_base_footprint_polygon():
    """Verify base footprint 4 corners are oriented correctly."""
    pts = VehicleGeometry.get_base_footprint()
    assert len(pts) == 4
    # Front-left, rear-left, rear-right, front-right
    fl, rl, rr, fr = pts
    assert fl == (0.28, 0.24)
    assert rl == (-0.28, 0.24)
    assert rr == (-0.28, -0.24)
    assert fr == (0.28, -0.24)


def test_oriented_footprint_transformation():
    """Verify coordinate transformation of vehicle polygon for arbitrary (x, y, yaw)."""
    # Robot at (10.0, 5.0) with yaw = 90 deg (pi/2)
    pts = VehicleGeometry.get_oriented_footprint(10.0, 5.0, math.pi / 2.0)
    assert len(pts) == 4
    # With 90 deg rotation: x_global = x - y_local, y_global = y + x_local
    # Front-left (0.28, 0.24) -> (10.0 - 0.24, 5.0 + 0.28) = (9.76, 5.28)
    fl = pts[0]
    assert fl[0] == pytest.approx(9.76, abs=1e-3)
    assert fl[1] == pytest.approx(5.28, abs=1e-3)


def test_passage_evaluation_safe_corridor():
    """Verify wide passage (W >= 0.68m) reports SAFE and can_fit=True."""
    can_fit, status, diag = VehicleGeometry.evaluate_passage(available_clear_width_m=1.20)
    assert can_fit is True
    assert status == "SAFE"
    assert diag["passage_status"] == "SAFE"
    assert diag["can_fit"] is True
    assert diag["can_turn"] is True
    assert diag["available_clear_width"] == 1.20
    assert diag["required_clear_width"] == 0.68
    assert diag["clearance_margin"] == pytest.approx(0.72, abs=1e-3)


def test_passage_evaluation_tight_corridor():
    """Verify narrow passage between 0.58m and 0.68m reports TIGHT and can_fit=True."""
    can_fit, status, diag = VehicleGeometry.evaluate_passage(available_clear_width_m=0.62)
    assert can_fit is True
    assert status == "TIGHT"
    assert diag["passage_status"] == "TIGHT"
    assert diag["can_fit"] is True
    assert diag["clearance_margin"] == pytest.approx(0.14, abs=1e-3)


def test_passage_evaluation_blocked_corridor():
    """Verify passage narrower than 0.58m (< vehicle width 0.48m + 2*0.05m) reports BLOCKED."""
    can_fit, status, diag = VehicleGeometry.evaluate_passage(available_clear_width_m=0.52)
    assert can_fit is False
    assert status == "BLOCKED"
    assert diag["passage_status"] == "BLOCKED"
    assert diag["can_fit"] is False
    assert "PASSAGE_BELOW_VEHICLE_WIDTH" in diag["reason"]


def test_passage_evaluation_sharp_turn_constraint():
    """Verify sharp turn (>35 deg) requires turning diameter >= 0.94m."""
    # Corridor is 0.75m wide: straight passage would be SAFE/TIGHT, but a 60 deg turn requires 0.94m
    can_fit, status, diag = VehicleGeometry.evaluate_passage(
        available_clear_width_m=0.75, heading_change_rad=math.radians(60.0)
    )
    assert can_fit is False
    assert status == "BLOCKED"
    assert diag["can_turn"] is False
    assert diag["reason"] == "INSUFFICIENT_TURNING_SPACE"

    # With 1.10m clearance, turning space is sufficient
    can_fit_wide, status_wide, diag_wide = VehicleGeometry.evaluate_passage(
        available_clear_width_m=1.10, heading_change_rad=math.radians(60.0)
    )
    assert can_fit_wide is True
    assert status_wide == "SAFE"
    assert diag_wide["can_turn"] is True
