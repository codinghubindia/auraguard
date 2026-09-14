import math
import pytest
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter


def test_grid_world_bijective():
    conv = MapCoordinateConverter(
        resolution=0.10,
        width_cells=400,
        height_cells=400,
        origin_x=-20.0,
        origin_y=-20.0,
        canvas_width=800,
        canvas_height=800
    )

    # Pick an arbitrary test point
    test_x, test_y = 5.23, -3.47
    col, row = conv.world_to_grid(test_x, test_y)
    rec_x, rec_y = conv.grid_to_world(col, row)

    # Recovered coordinate should be within half a resolution step (center of cell)
    assert abs(rec_x - test_x) <= conv.resolution
    assert abs(rec_y - test_y) <= conv.resolution


def test_canvas_world_roundtrip():
    conv = MapCoordinateConverter(
        resolution=0.05,
        width_cells=500,
        height_cells=500,
        origin_x=-10.0,
        origin_y=-10.0,
        canvas_width=600,
        canvas_height=600
    )

    # Pick an arbitrary canvas pixel click
    u_click, v_click = 350.0, 220.0
    wx, wy = conv.canvas_to_world(u_click, v_click)
    rec_u, rec_v = conv.world_to_canvas(wx, wy)

    # Canvas coordinates should match within 2 pixels
    assert abs(rec_u - u_click) <= 2.0
    assert abs(rec_v - v_click) <= 2.0


def test_bounds_checking():
    conv = MapCoordinateConverter(
        resolution=0.10,
        width_cells=200,
        height_cells=200,
        origin_x=0.0,
        origin_y=0.0
    )
    # Inside bounds
    assert conv.is_in_world_bounds(5.0, 5.0) is True
    assert conv.is_in_world_bounds(19.9, 19.9) is True

    # Outside bounds
    assert conv.is_in_world_bounds(-1.0, 5.0) is False
    assert conv.is_in_world_bounds(25.0, 5.0) is False
    assert conv.is_in_world_bounds(5.0, -0.5) is False


def test_coordinate_inversion_y():
    """Verify REP-103 (+Y North) maps to lower Canvas V (towards top of screen)."""
    conv = MapCoordinateConverter(
        resolution=0.05,
        width_cells=600,
        height_cells=600,
        origin_x=-15.0,
        origin_y=-15.0,
        canvas_width=600,
        canvas_height=600,
    )
    # y = +5m is North, y = -5m is South
    _, v_north = conv.world_to_canvas(0.0, 5.0)
    _, v_south = conv.world_to_canvas(0.0, -5.0)
    assert v_north < v_south, f"Expected v_north ({v_north}) < v_south ({v_south})"


def test_basemap_generator():
    """Verify RELLIS basemap generates valid 600x600 3-channel BGR image."""
    from naviguard_dashboard.basemap_generator import generate_rellis_basemap
    img = generate_rellis_basemap(600, 600, 0.05, -15.0, -15.0)
    assert img is not None
    assert img.shape == (600, 600, 3)
    assert img.dtype.name == 'uint8'

