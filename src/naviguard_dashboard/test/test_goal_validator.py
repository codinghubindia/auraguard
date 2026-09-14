import math
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
from naviguard_dashboard.goal_validator import GoalValidator


def test_valid_goal_acceptance():
    conv = MapCoordinateConverter(
        resolution=0.1, width_cells=200, height_cells=200,
        origin_x=-10.0, origin_y=-10.0
    )
    validator = GoalValidator(conv)

    dummy_grid = [0] * (200 * 200)  # Free space
    is_valid, reason = validator.validate_goal(
        x=2.0, y=1.0, grid_data=dummy_grid,
        navigation_online=True, recovery_state="NORMAL", has_localization=True
    )
    assert is_valid is True
    assert "GOAL VALID" in reason


def test_goal_rejection_out_of_bounds():
    conv = MapCoordinateConverter(
        resolution=0.1, width_cells=100, height_cells=100,
        origin_x=0.0, origin_y=0.0
    )
    validator = GoalValidator(conv)

    # (15.0, 5.0) is beyond 10.0m map
    is_valid, reason = validator.validate_goal(x=15.0, y=5.0)
    assert is_valid is False
    assert "outside map bounds" in reason


def test_goal_rejection_occupied():
    conv = MapCoordinateConverter(
        resolution=0.1, width_cells=100, height_cells=100,
        origin_x=0.0, origin_y=0.0
    )
    validator = GoalValidator(conv)

    grid = [0] * (100 * 100)
    col, row = conv.world_to_grid(3.0, 3.0)
    grid[row * 100 + col] = 100  # Mark obstacle

    is_valid, reason = validator.validate_goal(x=3.0, y=3.0, grid_data=grid)
    assert is_valid is False
    assert "OCCUPIED" in reason


def test_goal_rejection_subsystem_offline():
    conv = MapCoordinateConverter()
    validator = GoalValidator(conv)

    # Navigation offline
    ok, r = validator.validate_goal(0.0, 0.0, navigation_online=False)
    assert ok is False
    assert "Navigation subsystem is OFFLINE" in r

    # Recovery in FAILED_SAFE
    ok, r = validator.validate_goal(0.0, 0.0, recovery_state="FAILED_SAFE")
    assert ok is False
    assert "FAILED_SAFE" in r

    # No localization
    ok, r = validator.validate_goal(0.0, 0.0, has_localization=False)
    assert ok is False
    assert "localization unavailable" in r


def test_create_goal_msg():
    msg = GoalValidator.create_goal_msg(x=3.5, y=-2.1, yaw=math.pi / 2, frame_id="map")
    assert msg.header.frame_id == "map"
    assert math.isclose(msg.pose.position.x, 3.5)
    assert math.isclose(msg.pose.position.y, -2.1)
    # Yaw 90 deg -> qz = sin(45 deg) = 0.7071, qw = cos(45 deg) = 0.7071
    assert math.isclose(msg.pose.orientation.z, math.sin(math.pi / 4), abs_tol=1e-4)
    assert math.isclose(msg.pose.orientation.w, math.cos(math.pi / 4), abs_tol=1e-4)


def test_goal_acceptance_degraded_and_string_statuses():
    conv = MapCoordinateConverter(
        resolution=0.1, width_cells=200, height_cells=200,
        origin_x=-10.0, origin_y=-10.0
    )
    validator = GoalValidator(conv)
    dummy_grid = [0] * (200 * 200)

    # String status "ONLINE"
    ok, r = validator.validate_goal(
        x=2.0, y=1.0, grid_data=dummy_grid,
        navigation_online="ONLINE", recovery_state="NORMAL", has_localization="ONLINE"
    )
    assert ok is True

    # Degraded status should still allow goal dispatch
    ok, r = validator.validate_goal(
        x=2.0, y=1.0, grid_data=dummy_grid,
        navigation_online="DEGRADED", recovery_state="NORMAL", has_localization="DEGRADED"
    )
    assert ok is True

    # OFFLINE string status rejected
    ok, r = validator.validate_goal(
        x=2.0, y=1.0, grid_data=dummy_grid,
        navigation_online="OFFLINE", recovery_state="NORMAL", has_localization="ONLINE"
    )
    assert ok is False
    assert "Navigation subsystem is OFFLINE" in r

