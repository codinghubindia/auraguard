import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.path_smoother import PathSmoother


def test_path_smoother_shortcuts_open_path():
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 50
    meta.height = 50
    meta.origin = Pose()
    msg.info = meta
    msg.data = np.zeros((50, 50), dtype=np.int8).flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.1)
    grid.update_from_msg(msg)

    # Jagged stair-step path from (0.5, 0.5) to (2.5, 2.5)
    raw_path = [
        (0.5, 0.5), (0.5, 1.0), (1.0, 1.0), (1.0, 1.5),
        (1.5, 1.5), (1.5, 2.0), (2.0, 2.0), (2.5, 2.5)
    ]

    smoother = PathSmoother()
    smoothed = smoother.smooth(raw_path, grid)

    # In wide open space, string-pulling should shortcut directly to endpoints
    assert len(smoothed) < len(raw_path)
    assert smoothed[0] == raw_path[0]
    assert smoothed[-1] == raw_path[-1]


def test_path_smoother_preserves_obstacle_clearance():
    # Place obstacle in the direct shortcut path
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 50
    meta.height = 50
    meta.origin = Pose()
    msg.info = meta

    data = np.zeros((50, 50), dtype=np.int8)
    data[15:25, 15:25] = 100  # Block center (2.0, 2.0)
    msg.data = data.flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.2)
    grid.update_from_msg(msg)

    # Path goes around obstacle
    raw_path = [
        (0.5, 0.5), (0.5, 3.5), (3.5, 3.5)
    ]

    smoother = PathSmoother()
    smoothed = smoother.smooth(raw_path, grid)

    # Direct line (0.5, 0.5) -> (3.5, 3.5) passes right through the obstacle!
    # Smoother MUST NOT take the direct shortcut through the obstacle
    assert len(smoothed) == 3
    assert smoothed[1] == (0.5, 3.5)
