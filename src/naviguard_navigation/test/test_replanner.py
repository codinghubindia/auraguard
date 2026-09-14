import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.waypoint_generator import Waypoint
from naviguard_navigation.replanner import Replanner


def test_replanner_detects_blocked_path():
    replanner = Replanner(min_replan_interval_sec=1.0, max_path_deviation_m=0.50)

    # Free grid initially
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 40
    meta.height = 40
    meta.origin = Pose()
    msg.info = meta
    msg.data = np.zeros((40, 40), dtype=np.int8).flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.15)
    grid.update_from_msg(msg)

    wps = [
        Waypoint(x=1.0, y=1.0, yaw=0.0, index=0),
        Waypoint(x=2.0, y=1.0, yaw=0.0, index=1),
        Waypoint(x=3.0, y=1.0, yaw=0.0, index=2),
    ]

    # Path is clear
    should_replan, reason = replanner.should_replan_due_to_obstacle(wps, 0, grid, now_sec=10.0)
    assert should_replan is False

    # Now a new obstacle appears between waypoint 1 and 2 at (2.5, 1.0)
    data = np.zeros((40, 40), dtype=np.int8)
    data[10, 25] = 100
    msg.data = data.flatten().tolist()
    grid.update_from_msg(msg)

    # Must detect path blockage and trigger replan
    should_replan, reason = replanner.should_replan_due_to_obstacle(wps, 0, grid, now_sec=12.0)
    assert should_replan is True
    assert "PATH_BLOCKED" in reason


def test_replanner_detects_excessive_cross_track_deviation():
    replanner = Replanner(min_replan_interval_sec=1.0, max_path_deviation_m=0.50)

    # Nominal tracking: error 0.2m
    should, _ = replanner.should_replan_due_to_deviation(0.20, now_sec=5.0)
    assert should is False

    # Excessive deviation: error 0.65m
    should, reason = replanner.should_replan_due_to_deviation(0.65, now_sec=7.0)
    assert should is True
    assert "EXCESSIVE_CROSS_TRACK_DEVIATION" in reason


def test_replanner_suppresses_deviation_during_active_maneuver():
    """Verify replanner does NOT trigger deviation replan while actively maneuvering or bypassing obstacles."""
    replanner = Replanner(min_replan_interval_sec=1.0, max_path_deviation_m=0.50)

    # Large cross-track error (0.75m > 0.50m) but robot is actively maneuvering
    should, reason = replanner.should_replan_due_to_deviation(0.75, now_sec=10.0, is_maneuvering=True)
    assert should is False
    assert reason == "MANEUVER_ACTIVE"

    # Once maneuver completes, excessive deviation triggers replan
    should, reason = replanner.should_replan_due_to_deviation(0.75, now_sec=12.0, is_maneuvering=False)
    assert should is True
    assert "EXCESSIVE_CROSS_TRACK_DEVIATION" in reason
