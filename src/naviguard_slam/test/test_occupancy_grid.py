import os
import tempfile
import numpy as np
import pytest

from naviguard_slam.occupancy_grid_mapper import OccupancyGridMapper


def test_grid_coordinate_transforms():
    mapper = OccupancyGridMapper(
        resolution=0.1,
        width_m=20.0,
        height_m=20.0,
        origin_x=-10.0,
        origin_y=-10.0,
    )

    assert mapper.width_cells == 200
    assert mapper.height_cells == 200

    # Center (0, 0) should be at cell (100, 100)
    pt = mapper.world_to_map(0.0, 0.0)
    assert pt is not None
    assert pt == (100, 100)

    # Convert back to world coordinates
    wx, wy = mapper.map_to_world(100, 100)
    assert pytest.approx(wx, abs=0.06) == 0.05
    assert pytest.approx(wy, abs=0.06) == 0.05

    # Out of bounds test
    assert mapper.world_to_map(25.0, 25.0) is None


def test_clearance_and_obstacle_marking():
    mapper = OccupancyGridMapper(
        resolution=0.1,
        width_m=10.0,
        height_m=10.0,
        origin_x=-5.0,
        origin_y=-5.0,
        clearance_radius_m=0.3,
        min_obstacle_height=0.08,
        max_obstacle_height=1.8,
    )

    # Initial state: all -1 (unknown)
    assert np.all(mapper.grid == -1)

    # Clear robot pose at (0, 0)
    mapper.update_robot_clearance(0.0, 0.0)
    center_cell = mapper.world_to_map(0.0, 0.0)
    assert mapper.grid[center_cell[1], center_cell[0]] == 0

    # Test obstacle elevation gating:
    # Z = 0.0 (ground plane) -> rejected (returns False, cell not 100)
    assert mapper.add_obstacle_point(1.0, 1.0, 0.0) is False
    c0 = mapper.world_to_map(1.0, 1.0)
    assert mapper.grid[c0[1], c0[0]] != 100

    # Z = -0.05 (depression / underground) -> rejected
    assert mapper.add_obstacle_point(1.0, 2.0, -0.05) is False
    c_neg = mapper.world_to_map(1.0, 2.0)
    assert mapper.grid[c_neg[1], c_neg[0]] != 100

    # Z = 0.05 (below min_obstacle_height 0.08) -> rejected
    assert mapper.add_obstacle_point(1.0, 3.0, 0.05) is False
    c_low = mapper.world_to_map(1.0, 3.0)
    assert mapper.grid[c_low[1], c_low[0]] != 100

    # Z = 0.08 (exact threshold boundary) -> accepted (returns True, cell is 100)
    assert mapper.add_obstacle_point(2.0, 1.0, 0.08) is True
    c_thresh = mapper.world_to_map(2.0, 1.0)
    assert mapper.grid[c_thresh[1], c_thresh[0]] == 100

    # Z = 0.20 (low obstacle like log/rock) -> accepted
    assert mapper.add_obstacle_point(2.0, 2.0, 0.20) is True
    c_rock = mapper.world_to_map(2.0, 2.0)
    assert mapper.grid[c_rock[1], c_rock[0]] == 100

    # Z = 1.0 (tree trunk / person) -> accepted
    assert mapper.add_obstacle_point(2.0, 3.0, 1.0) is True
    c_tree = mapper.world_to_map(2.0, 3.0)
    assert mapper.grid[c_tree[1], c_tree[0]] == 100

    # Z = 2.5 (overhead canopy / wire above max_obstacle_height) -> rejected
    assert mapper.add_obstacle_point(2.0, 4.0, 2.5) is False
    c_high = mapper.world_to_map(2.0, 4.0)
    assert mapper.grid[c_high[1], c_high[0]] != 100


def test_ray_clearing_and_visual_observations():
    mapper = OccupancyGridMapper(
        resolution=0.1,
        width_m=10.0,
        height_m=10.0,
        origin_x=-5.0,
        origin_y=-5.0,
        clearance_radius_m=0.3,
        min_obstacle_height=0.08,
        max_obstacle_height=1.8,
    )

    # Initial state: all -1 (unknown)
    assert np.all(mapper.grid == -1)

    # Observation 1: Line of sight to an elevated obstacle at (3.0, 0.0, Z=0.5m)
    # The cells along the ray (0 to 2.9m) must be cleared as free (0).
    # The cell at (3.0, 0.0) must be marked as obstacle (100).
    # Cells away from the ray (e.g. at (0, 3.0)) must remain unknown (-1).
    mapper.update_visual_observation(robot_x=0.0, robot_y=0.0, landmark_x=3.0, landmark_y=0.0, landmark_z=0.5)

    ray_mid = mapper.world_to_map(1.5, 0.0)
    obs_end = mapper.world_to_map(3.0, 0.0)
    unseen = mapper.world_to_map(0.0, 3.0)

    assert mapper.grid[ray_mid[1], ray_mid[0]] == 0, "Line of sight must be marked as free space"
    assert mapper.grid[obs_end[1], obs_end[0]] == 100, "Elevated obstacle must be marked as 100"
    assert mapper.grid[unseen[1], unseen[0]] == -1, "Unobserved space must remain unknown (-1)"

    # Observation 2: Obstacle must NOT be overwritten by subsequent ray clearing
    # Cast another ray crossing or pointing at the existing obstacle
    mapper.clear_ray(0.0, 0.0, 4.0, 0.0)
    assert mapper.grid[obs_end[1], obs_end[0]] == 100, "Ray clearing must not overwrite confirmed obstacles"

    # Observation 3: Line of sight to a ground feature (e.g. shadow edge at Z=0.0)
    # Traversed ray must be free (0) AND endpoint must also be free ground (0), NOT an obstacle!
    mapper.update_visual_observation(robot_x=0.0, robot_y=0.0, landmark_x=0.0, landmark_y=-2.0, landmark_z=0.0)
    ground_ray_mid = mapper.world_to_map(0.0, -1.0)
    ground_end = mapper.world_to_map(0.0, -2.0)

    assert mapper.grid[ground_ray_mid[1], ground_ray_mid[0]] == 0
    assert mapper.grid[ground_end[1], ground_end[0]] == 0, "Ground-level feature must be marked as free ground"


def test_pgm_and_yaml_export():
    mapper = OccupancyGridMapper(resolution=0.1, width_m=4.0, height_m=4.0, origin_x=-2.0, origin_y=-2.0)
    mapper.update_robot_clearance(0.0, 0.0)
    mapper.add_obstacle_point(1.0, 0.0, 0.2)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = os.path.join(tmpdir, "test_map")
        pgm_p, yaml_p = mapper.save_map_files(base_path)

        assert os.path.exists(pgm_p)
        assert os.path.exists(yaml_p)

        with open(pgm_p, 'rb') as f:
            header = f.readline().decode('ascii')
            assert header.strip() == "P5"
