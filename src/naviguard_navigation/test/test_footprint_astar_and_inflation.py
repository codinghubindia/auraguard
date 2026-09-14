"""Integration tests for footprint-based costmap inflation, narrow passage clearance, and A* planning."""

import math
import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData

from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.global_planner import GlobalPlannerAStar
from naviguard_navigation.vehicle_geometry import VehicleGeometry


def create_test_grid(width=60, height=60, resolution=0.05, origin_x=0.0, origin_y=0.0):
    """Utility to create an initialized 3.0m x 3.0m test grid."""
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = resolution
    meta.width = width
    meta.height = height
    meta.origin.position.x = origin_x
    meta.origin.position.y = origin_y
    msg.info = meta
    msg.data = np.zeros((height, width), dtype=np.int8).flatten().tolist()
    grid = NavigationOccupancyGrid()
    grid.update_from_msg(msg)
    return grid


def test_inflation_radius_derived_from_vehicle_geometry():
    """Verify default inflation radius is geometrically derived from inscribed radius + margin."""
    grid = NavigationOccupancyGrid()
    expected_inflation = VehicleGeometry.INSCRIBED_RADIUS_M + VehicleGeometry.INFLATION_MARGIN_M
    assert grid.inflation_radius_m == pytest.approx(expected_inflation, abs=1e-4)
    assert grid.inflation_radius_m == pytest.approx(0.34, abs=1e-2)


def test_footprint_collision_detection():
    """Verify oriented footprint collision checks respect the 0.56m x 0.48m envelope."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    # Robot at center (2.5m, 2.5m) with yaw = 0
    rx, ry, ryaw = 2.5, 2.5, 0.0
    assert grid.is_footprint_collision_free(rx, ry, ryaw) is True

    # Place an obstacle comfortably outside the footprint and its inflation zone (e.g. at x = 2.5 + 0.75m)
    # Footprint should still be free
    grid.mark_blocked_region(2.5 + 0.75, 2.5, radius_m=0.05)
    assert grid.is_footprint_collision_free(rx, ry, ryaw) is True

    # Place an obstacle directly on the vehicle corner (front-left: 2.5 + 0.25, 2.5 + 0.20)
    grid.mark_blocked_region(2.5 + 0.25, 2.5 + 0.20, radius_m=0.04)
    assert grid.is_footprint_collision_free(rx, ry, ryaw) is False


def test_swept_footprint_collision_checking():
    """Verify swept footprint detects collisions along the line segment."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    p1 = (1.0, 2.5)
    p2 = (4.0, 2.5)

    # Empty grid: swept path should be free
    assert grid.is_swept_footprint_collision_free(p1, p2) is True

    # Place an obstacle directly on the trajectory path
    grid.mark_blocked_region(2.5, 2.5, radius_m=0.10)
    assert grid.is_swept_footprint_collision_free(p1, p2) is False


def test_wide_corridor_planning_succeeds():
    """Verify A* planner finds a valid path through a wide corridor (width >= 1.20m > 0.68m)."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    # Create a corridor from x=1.0 to x=4.0 with walls at y=1.5 and y=3.5 (corridor width = 2.0m)
    # With inflation (0.34m each side), free passage is 2.0 - 0.68 = 1.32m > 0.68m nominal
    for x_idx in range(20, 80):
        for y_idx in range(0, 30):   # y <= 1.50
            grid.base_raw_grid[y_idx, x_idx] = 100
        for y_idx in range(70, 100): # y >= 3.50
            grid.base_raw_grid[y_idx, x_idx] = 100
    grid._compute_cost_grid()

    planner = GlobalPlannerAStar()
    path = planner.plan(grid, (1.2, 2.5), (3.8, 2.5))
    assert path is not None
    assert len(path) > 0
    # Final waypoint should be near goal
    assert math.hypot(path[-1][0] - 3.8, path[-1][1] - 2.5) < 0.25


def test_narrow_corridor_below_tight_limit_rejected():
    """Verify A* planner rejects or avoids corridor narrower than tight passage limit (0.58m)."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    # Create an impassable dividing wall across the entire map at x=50
    for y_idx in range(0, 100):
        grid.base_raw_grid[y_idx, 50] = 100
    # Open only a 0.30m narrow slit in the center: y from 47 to 53 (6 cells * 0.05m = 0.30m < 0.58m tight limit)
    for y_idx in range(47, 54):
        grid.base_raw_grid[y_idx, 50] = 0
    grid._compute_cost_grid()

    planner = GlobalPlannerAStar()
    # Try to plan directly through the 0.30m bottleneck with no detour possible
    path = planner.plan(grid, (1.5, 2.5), (3.5, 2.5))
    # Path must be None because corridor clearance fails the tight passage constraint
    assert path is None


def test_small_passage_wide_enough_succeeds():
    """Verify A* planner successfully finds path through a small passage wide enough for the vehicle (0.65m > 0.48m)."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    # Create dividing wall at x=50 (x=2.5m)
    for y_idx in range(0, 100):
        grid.base_raw_grid[y_idx, 50] = 100
    # Open a 0.65m passage: 13 cells * 0.05m = 0.65m centered at y=50 (y_idx 44 to 56 inclusive is 13 cells)
    for y_idx in range(44, 57):
        grid.base_raw_grid[y_idx, 50] = 0
    grid._compute_cost_grid()

    planner = GlobalPlannerAStar()
    # Plan from left side to right side through the 0.65m small passage
    path = planner.plan(grid, (1.5, 2.5), (3.5, 2.5))
    assert path is not None
    assert len(path) > 0
    # Destination reached near (3.5, 2.5)
    assert math.hypot(path[-1][0] - 3.5, path[-1][1] - 2.5) < 0.25
    # Verify the vehicle footprint remains collision free along the path through the narrow passage
    mid_point = None
    for pt in path:
        if abs(pt[0] - 2.5) < 0.1:
            mid_point = pt
            break
    assert mid_point is not None
    assert grid.is_footprint_collision_free(mid_point[0], mid_point[1], yaw=0.0) is True


def test_tight_passage_above_inscribed_width_succeeds():
    """Verify A* planner successfully traverses a 0.52m passage (>0.48m vehicle width, <0.58m nominal tight limit)."""
    grid = create_test_grid(width=100, height=100, resolution=0.05)
    # Dividing wall at x=50 (x=2.5m)
    for y_idx in range(0, 100):
        grid.base_raw_grid[y_idx, 50] = 100
    # Open 0.52m passage: ~10 cells (0.50m) to 11 cells (0.55m) centered at y=50
    for y_idx in range(45, 56):
        grid.base_raw_grid[y_idx, 50] = 0
    grid._compute_cost_grid()

    planner = GlobalPlannerAStar()
    path = planner.plan(grid, (1.5, 2.5), (3.5, 2.5))
    assert path is not None
    assert len(path) > 0
    assert math.hypot(path[-1][0] - 3.5, path[-1][1] - 2.5) < 0.25

