import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid


def make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5):
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = resolution
    meta.width = width
    meta.height = height
    origin = Pose()
    origin.position.x = origin_x
    origin.position.y = origin_y
    meta.origin = origin
    msg.info = meta
    # Default to free space
    data = np.zeros((height, width), dtype=np.int8)
    msg.data = data.flatten().tolist()
    return msg


def test_occupancy_grid_conversion_and_coordinates():
    grid = NavigationOccupancyGrid()
    msg = make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5)
    grid.update_from_msg(msg)

    assert grid.is_initialized is True
    assert grid.width_cells == 100
    assert grid.height_cells == 100
    assert grid.resolution == 0.05

    # Center is (0.0, 0.0) -> cell (50, 50)
    cell = grid.world_to_map(0.0, 0.0)
    assert cell == (50, 50)

    # Convert back
    wx, wy = grid.map_to_world(50, 50)
    assert pytest.approx(wx, abs=0.03) == 0.0
    assert pytest.approx(wy, abs=0.03) == 0.0


def test_obstacle_inflation():
    grid = NavigationOccupancyGrid(inflation_radius_m=0.20, proximity_radius_m=0.50)
    msg = make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5)

    # Place a single obstacle at center cell (50, 50)
    data = np.zeros((100, 100), dtype=np.int8)
    data[50, 50] = 100
    msg.data = data.flatten().tolist()
    grid.update_from_msg(msg)

    # Center is lethal
    assert grid.is_lethal(50, 50) == True

    # Cell 2 cells away (0.10m <= 0.20m inflation radius) must be lethal
    assert grid.is_lethal(52, 50) == True
    assert grid.is_lethal(50, 52) == True

    # Cell 6 cells away (0.30m > 0.20m inflation radius) is NOT lethal, but has proximity cost
    assert grid.is_lethal(56, 50) == False
    assert grid.get_cost(56, 50) > 0.0

    # Cell far away (> 0.50m) has zero cost
    assert grid.get_cost(80, 80) == 0.0


def test_unknown_space_handling():
    # Policy 1: allow_unknown=False (unknown is forbidden/lethal)
    grid_forbidden = NavigationOccupancyGrid(allow_unknown=False)
    msg = make_test_grid(width=50, height=50, resolution=0.05)
    data = np.full((50, 50), -1, dtype=np.int8)
    data[20:30, 20:30] = 0  # Only small patch is free
    msg.data = data.flatten().tolist()
    grid_forbidden.update_from_msg(msg)

    assert grid_forbidden.is_lethal(0, 0) == True
    assert grid_forbidden.is_lethal(25, 25) == False

    # Policy 2: allow_unknown=True (unknown traversable with penalty)
    grid_allowed = NavigationOccupancyGrid(allow_unknown=True, unknown_cost_penalty=10.0)
    grid_allowed.update_from_msg(msg)

    assert grid_allowed.is_lethal(0, 0) == False
    assert grid_allowed.get_cost(0, 0) >= 10.0
    assert grid_allowed.get_cost(25, 25) == 0.0


def test_mark_blocked_region_and_persistence():
    grid = NavigationOccupancyGrid(inflation_radius_m=0.15)
    msg = make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5)
    grid.update_from_msg(msg)

    assert grid.is_lethal(50, 50) is False

    # Mark blocked patch around (0.0, 0.0) -> cell (50, 50)
    count = grid.mark_blocked_region(0.0, 0.0, radius_m=0.20)
    assert count > 0
    assert grid.is_lethal(50, 50) is True

    # When a new map update arrives, the blocked region persists!
    new_msg = make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5)
    grid.update_from_msg(new_msg)
    assert grid.is_lethal(50, 50) is True

    # Clear blocked regions restores free space
    grid.clear_blocked_regions()
    assert grid.is_lethal(50, 50) is False


def test_evaluate_360_passages():
    grid = NavigationOccupancyGrid(inflation_radius_m=0.10)
    msg = make_test_grid(width=100, height=100, resolution=0.05, origin_x=-2.5, origin_y=-2.5)
    grid.update_from_msg(msg)

    # Robot at (0.0, 0.0), goal at (2.0, 0.0)
    # Add a block in front at (1.0, 0.0) and walls, leaving open corridors at +45 deg and -45 deg
    grid.mark_blocked_region(1.0, 0.0, radius_m=0.35)

    res = grid.evaluate_360_passages(0.0, 0.0, goal_x=2.0, goal_y=0.0, num_sectors=36, max_range_m=2.0, vehicle_width_m=0.48)
    assert res["scan_complete"] is True
    assert res["sectors_evaluated"] == 36
    assert len(res["passages"]) > 0
    # Widest corridor should fit the 0.48m vehicle
    assert res["widest_corridor_m"] >= 0.48

