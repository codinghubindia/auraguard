"""Unit tests for RELLIS terrain and world builders."""

import numpy as np
import pytest
from naviguard_rellis.terrain_builder import TerrainBuilder, TerrainSurfaceType
from naviguard_rellis.world_builder import RellisWorldBuilder


def test_terrain_builder_elevation_grid():
    builder = TerrainBuilder(resolution_m=0.50, bounds_m=(-5.0, 5.0, -5.0, 5.0))
    assert builder.width_cells == 20
    assert builder.height_cells == 20

    # 3D points
    pts = np.array([
        [0.0, 0.0, 1.5],
        [0.1, 0.1, 2.5],
        [-4.0, -4.0, 0.5],
    ], dtype=np.float32)

    grid = builder.build_elevation_grid(pts, fill_value=0.0)
    assert grid.shape == (20, 20)
    # Average of (1.5 + 2.5) / 2 = 2.0 at center cell (10, 10)
    assert pytest.approx(grid[10, 10], abs=0.01) == 2.0
    # Point at (-4, -4) -> cell (2, 2)
    assert pytest.approx(grid[2, 2], abs=0.01) == 0.5


def test_terrain_builder_surface_classification():
    builder = TerrainBuilder(resolution_m=0.50, bounds_m=(-10.0, 10.0, -10.0, 10.0))
    grid = np.zeros((builder.height_cells, builder.width_cells), dtype=np.float32)

    # Place a steep elevation spike (obstacle) at (5, 5)
    grid[15:17, 15:17] = 2.0

    mud_patches = [(0.0, -4.0, 1.5)]
    surfaces = builder.classify_surfaces(grid, trail_corridor_width_m=3.0, mud_patches=mud_patches)

    # Trail corridor along center (cy = 20)
    cy = builder.height_cells // 2
    assert surfaces[cy, 20] == TerrainSurfaceType.TRAIL

    # Off-trail is grass
    assert surfaces[5, 5] == TerrainSurfaceType.GRASS

    # Mud patch at (0, -4) -> around cy - 8
    assert surfaces[cy - 8, 20] == TerrainSurfaceType.MUD

    # Friction coefficients
    mu_trail, _ = builder.get_friction_coefficients(TerrainSurfaceType.TRAIL)
    mu_mud, _ = builder.get_friction_coefficients(TerrainSurfaceType.MUD)
    assert mu_trail > 0.80
    assert mu_mud < 0.30


def test_world_builder_sdf_generation():
    wb = RellisWorldBuilder("test_rellis_world")
    wb.add_trail_segment(5.0, 0.0, 0.002, length=10.0, width=3.0)
    wb.add_mud_patch(3.0, 0.0, radius=1.0)
    wb.add_fallen_log(8.0, 0.0, 0.15)

    sdf_str = wb.generate_sdf()
    assert "<world name=\"test_rellis_world\">" in sdf_str
    assert "trail_segment_1" in sdf_str
    assert "mud_patch_1" in sdf_str
    assert "fallen_log_1" in sdf_str
    assert "<mu>0.85</mu>" in sdf_str
    assert "<mu>0.22</mu>" in sdf_str
