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


def test_trajectory_adjustment_and_small_gap_centering():
    """Verify PathFollower dynamically adjusts direction based on trajectory and centers in small gaps."""
    from naviguard_navigation.path_follower import PathFollower
    from naviguard_navigation.waypoint_generator import Waypoint
    from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from geometry_msgs.msg import Pose

    follower = PathFollower(max_linear_velocity=0.25, min_linear_velocity=0.05, max_angular_velocity=0.35)

    # 1. Straight path along x-axis from (0, 0) to (3.0, 0.0)
    wps = [
        Waypoint(x=0.0, y=0.0, yaw=0.0, index=0),
        Waypoint(x=1.0, y=0.0, yaw=0.0, index=1),
        Waypoint(x=2.0, y=0.0, yaw=0.0, index=2),
        Waypoint(x=3.0, y=0.0, yaw=0.0, index=3),
    ]

    # Setup 100x100 grid (5m x 5m, resolution 0.05m, origin -2.5, -2.5)
    grid_msg = OccupancyGrid()
    grid_msg.info = MapMetaData(
        resolution=0.05,
        width=100,
        height=100,
        origin=Pose(),
    )
    grid_msg.info.origin.position.x = -2.5
    grid_msg.info.origin.position.y = -2.5
    grid_msg.data = [0] * (100 * 100)

    nav_grid = NavigationOccupancyGrid(inflation_radius_m=0.34)
    nav_grid.update_from_msg(grid_msg)

    # Test 1: Open clear path -> SAFE, can_pass=True, nominal speed
    vx, wz, look_wp, cross_err, traj_info = follower.compute_commands_with_trajectory_adjustment(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, waypoints=wps, occ_grid=nav_grid
    )
    assert traj_info["can_pass"] is True
    assert traj_info["status"] == "SAFE"
    assert vx > 0.15

    # Test 2: Obstacle encroaching on left side at x=1.0, y=+0.35 -> should adjust steering to right
    # Mark obstacle on left
    nav_grid.mark_blocked_region(1.0, 0.35, radius_m=0.15)
    vx_adj, wz_adj, _, _, traj_adj = follower.compute_commands_with_trajectory_adjustment(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, waypoints=wps, occ_grid=nav_grid
    )
    assert traj_adj["can_pass"] is True
    # wz should be negative (turning right away from left obstacle)
    assert wz_adj <= 0.0 or traj_adj["steering_adjustment_rad"] <= 0.0

    # Test 3: Completely blocked forward corridor at x=0.6, y=0.0 -> BLOCKED, vx=0
    nav_grid.mark_blocked_region(0.6, 0.0, radius_m=0.35)
    vx_blk, wz_blk, _, _, traj_blk = follower.compute_commands_with_trajectory_adjustment(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, waypoints=wps, occ_grid=nav_grid
    )
    assert traj_blk["can_pass"] is False
    assert traj_blk["status"] == "BLOCKED"
    assert vx_blk == 0.0

    # Test 4: Distant obstacle at x=1.5m does NOT freeze vehicle at start (can_pass=True, vx > 0.10)
    nav_grid.clear_blocked_regions()
    nav_grid.mark_blocked_region(1.5, 0.0, radius_m=0.30)
    vx_start, wz_start, _, _, traj_start = follower.compute_commands_with_trajectory_adjustment(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0, waypoints=wps, occ_grid=nav_grid
    )
    assert traj_start["can_pass"] is True
    assert vx_start > 0.10


def test_path_follower_heading_hysteresis_and_arc_turning():
    """Verify human-like continuous arc turning and in-place turn hysteresis."""
    from naviguard_navigation.path_follower import PathFollower
    from naviguard_navigation.waypoint_generator import Waypoint

    follower = PathFollower(
        max_linear_velocity=0.25,
        min_linear_velocity=0.05,
        turn_in_place_angle_threshold=1.30,  # ~75 deg
        turn_in_place_exit_threshold=0.65,   # ~37 deg
    )

    wps = [
        Waypoint(x=0.0, y=0.0, yaw=0.0, index=0),
        Waypoint(x=2.0, y=0.0, yaw=0.0, index=1),
    ]

    # 1. Straight heading (yaw=0.0) -> full speed forward
    vx, wz, _, _ = follower.compute_commands(0.0, 0.0, 0.0, wps)
    assert vx > 0.20
    assert not follower.is_turning_in_place

    # 2. Moderate heading error (yaw = -0.80 rad, ~46 deg):
    # Continuous rolling crawl along arc (vx ~ 0.05 - 0.08 m/s)
    vx_arc, wz_arc, _, _ = follower.compute_commands(0.0, 0.0, -0.80, wps)
    assert not follower.is_turning_in_place
    assert 0.05 <= vx_arc <= 0.10
    assert wz_arc > 0.0  # Steer back to 0.0

    # 3. Large heading error (yaw = -1.40 rad, ~80 deg > 1.30 rad):
    # Enters in-place turn (vx = 0.0, is_turning_in_place = True)
    vx_turn, wz_turn, _, _ = follower.compute_commands(0.0, 0.0, -1.40, wps)
    assert follower.is_turning_in_place
    assert vx_turn == 0.0
    assert wz_turn > 0.0

    # 4. Hysteresis: As robot rotates, heading error reduces to 0.80 rad (~46 deg).
    # Since 0.80 > 0.65 (exit threshold), it remains in in-place turn without chattering
    vx_hys, wz_hys, _, _ = follower.compute_commands(0.0, 0.0, -0.80, wps)
    assert follower.is_turning_in_place
    assert vx_hys == 0.0

    # 5. Heading error drops below 0.65 rad (yaw = -0.50 rad, ~28 deg):
    # Exits in-place turn and resumes continuous rolling forward
    vx_exit, wz_exit, _, _ = follower.compute_commands(0.0, 0.0, -0.50, wps)
    assert not follower.is_turning_in_place
    assert vx_exit >= follower.min_linear_velocity


def test_path_follower_can_pass_during_in_place_turn():
    """Verify that during in-place turning, traj_info reports can_pass=True so replanning is not falsely triggered."""
    from naviguard_navigation.path_follower import PathFollower
    from naviguard_navigation.waypoint_generator import Waypoint

    follower = PathFollower(
        turn_in_place_angle_threshold=1.30,
        turn_in_place_exit_threshold=0.65,
    )

    wps = [
        Waypoint(x=0.0, y=0.0, yaw=0.0, index=0),
        Waypoint(x=2.0, y=0.0, yaw=0.0, index=1),
    ]

    # Robot facing opposite direction (-pi rad heading error)
    vx, wz, _, _, traj_info = follower.compute_commands_with_trajectory_adjustment(
        0.0, 0.0, 3.14, wps
    )
    assert vx == 0.0
    assert traj_info["can_pass"] is True
    assert traj_info["status"] == "TURNING_IN_PLACE"
    assert traj_info["reason"] == "ALIGNING_HEADING"


def test_preview_horizon_smooth_deceleration_and_anticipation():
    """Verify anticipatory deceleration and steering without slamming emergency stops."""
    from naviguard_navigation.path_follower import PathFollower
    from naviguard_navigation.waypoint_generator import Waypoint
    from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
    from nav_msgs.msg import OccupancyGrid, MapMetaData
    from geometry_msgs.msg import Pose

    follower = PathFollower(max_linear_velocity=0.25, min_linear_velocity=0.05)

    wps = [
        Waypoint(x=0.0, y=0.0, yaw=0.0, index=0),
        Waypoint(x=1.0, y=0.0, yaw=0.0, index=1),
        Waypoint(x=2.0, y=0.0, yaw=0.0, index=2),
    ]

    grid_msg = OccupancyGrid()
    grid_msg.info = MapMetaData(
        resolution=0.05,
        width=100,
        height=100,
        origin=Pose(),
    )
    grid_msg.info.origin.position.x = -2.5
    grid_msg.info.origin.position.y = -2.5
    grid_msg.data = [0] * (100 * 100)

    nav_grid = NavigationOccupancyGrid(inflation_radius_m=0.34)
    nav_grid.update_from_msg(grid_msg)

    # Obstacle placed at x=1.10m, y=0.0m:
    # Vehicle detects obstacle in preview horizon and adjusts steering around it smoothly
    nav_grid.mark_blocked_region(1.10, 0.0, radius_m=0.20)
    vx, wz, _, _, traj = follower.compute_commands_with_trajectory_adjustment(
        0.0, 0.0, 0.0, wps, occ_grid=nav_grid
    )
    assert traj["can_pass"] is True
    assert abs(traj["steering_adjustment_rad"]) > 0.05  # Actively steered around obstacle

    # Now test progressive deceleration: obstacle placed directly ahead at x=1.0m, y=0.0m with radius 0.25m
    nav_grid.clear_blocked_regions()
    nav_grid.mark_blocked_region(1.0, 0.0, radius_m=0.25)
    vx_slow, wz_slow, _, _, traj_slow = follower.compute_commands_with_trajectory_adjustment(
        0.0, 0.0, 0.0, wps, occ_grid=nav_grid
    )
    # Forward progress constrained: vehicle slows down progressively (0.05 <= vx < 0.25) rather than slamming to 0 or charging full speed
    assert traj_slow["can_pass"] is True
    assert 0.05 <= vx_slow < 0.25
    assert traj_slow["status"] == "TIGHT"




