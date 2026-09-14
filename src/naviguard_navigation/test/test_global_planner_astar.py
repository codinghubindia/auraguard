import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.global_planner import GlobalPlannerAStar


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
    grid = NavigationOccupancyGrid(inflation_radius_m=0.15, proximity_radius_m=0.30)
    grid.update_from_msg(msg)
    return grid


def test_astar_basic_path_open_environment():
    grid = make_grid()
    planner = GlobalPlannerAStar()

    start = (1.0, 1.0)
    goal = (4.0, 4.0)
    path = planner.plan(grid, start, goal)

    assert path is not None
    assert len(path) > 2
    assert pytest.approx(path[0][0], abs=0.01) == 1.0
    assert pytest.approx(path[0][1], abs=0.01) == 1.0
    assert pytest.approx(path[-1][0], abs=0.01) == 4.0
    assert pytest.approx(path[-1][1], abs=0.01) == 4.0


def test_astar_blocked_path_navigates_around_obstacle():
    # Place a vertical wall between x=2.5 and y=1.0 to 4.0
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 60
    meta.height = 60
    meta.origin = Pose()
    msg.info = meta

    data = np.zeros((60, 60), dtype=np.int8)
    # Wall from y=10 to 40 at x=25 (leaving opening at y > 40)
    data[10:40, 25:27] = 100
    msg.data = data.flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.15, proximity_radius_m=0.30)
    grid.update_from_msg(msg)
    planner = GlobalPlannerAStar()

    start = (1.5, 2.5)
    goal = (4.5, 2.5)
    path = planner.plan(grid, start, goal)

    assert path is not None
    # Path must route around the wall (y must exceed wall boundary y > 4.0m or y < 1.0m)
    y_coords = [p[1] for p in path]
    assert max(y_coords) > 4.1 or min(y_coords) < 0.9

    # Verify no points on path hit lethal cells
    for pt in path:
        cell = grid.world_to_map(pt[0], pt[1])
        assert not grid.is_lethal(cell[0], cell[1])


def test_astar_start_equals_goal():
    grid = make_grid()
    planner = GlobalPlannerAStar()
    start = (2.0, 2.0)
    path = planner.plan(grid, start, start)

    assert path is not None
    assert len(path) == 2
    assert path[0] == start
    assert path[1] == start


def test_astar_unreachable_goal():
    # Completely enclose goal inside a solid box
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 50
    meta.height = 50
    meta.origin = Pose()
    msg.info = meta

    data = np.zeros((50, 50), dtype=np.int8)
    # Complete box around (35, 35)
    data[30:42, 30] = 100
    data[30:42, 41] = 100
    data[30, 30:42] = 100
    data[41, 30:42] = 100
    msg.data = data.flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.15)
    grid.update_from_msg(msg)
    planner = GlobalPlannerAStar()

    start = (1.0, 1.0)
    goal = (3.5, 3.5)  # Inside sealed box
    path = planner.plan(grid, start, goal)

    assert path is None
