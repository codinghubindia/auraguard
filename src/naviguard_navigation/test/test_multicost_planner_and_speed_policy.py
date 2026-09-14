import math
import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.global_planner import GlobalPlannerAStar
from naviguard_navigation.path_follower import PathFollower
from naviguard_navigation.waypoint_generator import Waypoint


def make_grid(width=60, height=60, res=0.1, origin_x=0.0, origin_y=0.0):
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = res
    meta.width = width
    meta.height = height
    origin = Pose()
    origin.position.x = origin_x
    origin.position.y = origin_y
    meta.origin = origin
    msg.info = meta
    data = np.zeros((height, width), dtype=np.int8)
    msg.data = data.flatten().tolist()
    grid = NavigationOccupancyGrid(
        inflation_radius_m=0.15,
        proximity_radius_m=0.40,
        safe_clearance_m=1.20,
    )
    grid.update_from_msg(msg)
    return grid


def test_multicost_planner_avoids_high_terrain_cost_patch():
    """Verify planner detours around difficult terrain (mud/rough patch)."""
    grid = make_grid(width=80, height=80, res=0.1)
    planner = GlobalPlannerAStar(clearance_weight=3.0, terrain_weight=5.0)

    # Start at (1.5, 4.0), Goal at (6.5, 4.0) - straight line along y=4.0
    # Place a patch of severe terrain (e.g. cost 80) directly in center at (4.0, 4.0)
    grid.mark_terrain_patch(cx=4.0, cy=4.0, radius_m=1.0, terrain_cost=80.0)

    start = (1.5, 4.0)
    goal = (6.5, 4.0)
    path = planner.plan(grid, start, goal)

    assert path is not None
    assert len(path) > 2

    # Verify that the path routed away from center (4.0, 4.0)
    # The middle points of the path should deviate significantly from y=4.0
    mid_points = [p for p in path if 3.0 <= p[0] <= 5.0]
    assert len(mid_points) > 0
    # Max deviation from y=4.0 should be at least 0.8m to clear the 1.0m patch
    max_dev = max(abs(p[1] - 4.0) for p in mid_points)
    assert max_dev >= 0.7, f"Expected path to deviate from terrain patch, but max deviation was {max_dev:.2f}m"


def test_multicost_planner_avoids_steep_slope_patch():
    """Verify planner routes around an unsafe/high slope zone."""
    grid = make_grid(width=80, height=80, res=0.1)
    planner = GlobalPlannerAStar()

    # Place steep slope (>22 deg -> lethal cost 100) at (4.0, 4.0)
    grid.mark_slope_patch(cx=4.0, cy=4.0, radius_m=0.8, slope_rad=math.radians(25.0))

    start = (1.5, 4.0)
    goal = (6.5, 4.0)
    path = planner.plan(grid, start, goal)

    assert path is not None
    # No point in the planned path should be within the lethal slope patch
    for p in path:
        dist_to_slope_center = math.hypot(p[0] - 4.0, p[1] - 4.0)
        assert dist_to_slope_center >= 0.8, f"Path passed through steep slope at {p}"


def test_adaptive_speed_scaling_terrain_and_clearance():
    """Verify PathFollower adapts linear speed based on terrain, clearance, and confidence."""
    follower = PathFollower(
        max_linear_velocity=0.25,
        min_linear_velocity=0.05,
        lookahead_distance_m=0.40,
    )

    waypoints = [
        Waypoint(x=0.0, y=0.0, yaw=0.0, index=0),
        Waypoint(x=1.0, y=0.0, yaw=0.0, index=1),
        Waypoint(x=2.0, y=0.0, yaw=0.0, index=2),
    ]

    # Baseline: open trail, full clearance, full confidence
    v_base, _, _, _ = follower.compute_commands(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        waypoints=waypoints,
        speed_scale=1.0, terrain_factor=1.0, clearance_factor=1.0,
    )
    assert pytest.approx(v_base, abs=0.02) == 0.25

    # Degraded terrain (mud): terrain_factor = 0.50
    v_mud, _, _, _ = follower.compute_commands(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        waypoints=waypoints,
        speed_scale=1.0, terrain_factor=0.50, clearance_factor=1.0,
    )
    assert v_mud < v_base
    assert v_mud >= follower.min_linear_velocity

    # Tight clearance corridor: clearance_factor = 0.40
    v_tight, _, _, _ = follower.compute_commands(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        waypoints=waypoints,
        speed_scale=1.0, terrain_factor=1.0, clearance_factor=0.40,
    )
    assert v_tight < v_base
    assert v_tight >= follower.min_linear_velocity

    # Low visual odometry confidence: speed_scale = 0.30
    v_low_conf, _, _, _ = follower.compute_commands(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        waypoints=waypoints,
        speed_scale=0.30, terrain_factor=1.0, clearance_factor=1.0,
    )
    assert v_low_conf < v_base
    assert v_low_conf >= follower.min_linear_velocity

    # Compound worst-case: all factors low -> must clamp to min_linear_velocity
    v_worst, _, _, _ = follower.compute_commands(
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        waypoints=waypoints,
        speed_scale=0.2, terrain_factor=0.2, clearance_factor=0.2,
    )
    assert pytest.approx(v_worst, abs=0.001) == follower.min_linear_velocity
