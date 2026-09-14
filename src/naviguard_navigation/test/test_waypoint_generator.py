import math
import pytest
from naviguard_navigation.waypoint_generator import WaypointGenerator


def test_waypoint_generator_uniform_spacing():
    gen = WaypointGenerator(target_spacing_m=0.30, min_spacing_m=0.10, max_spacing_m=0.50)
    # Straight path of 3 meters
    path = [(0.0, 0.0), (3.0, 0.0)]
    wps = gen.generate(path, final_yaw=0.0)

    assert len(wps) >= 9
    assert wps[0].x == 0.0
    assert pytest.approx(wps[-1].x, abs=0.01) == 3.0

    # Verify headings are ~0 rad along positive X
    for wp in wps:
        assert pytest.approx(wp.yaw, abs=0.01) == 0.0


def test_waypoint_generator_turn_headings():
    gen = WaypointGenerator(target_spacing_m=0.50)
    # Right-angle path: forward along X, then turn along Y
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    wps = gen.generate(path)

    # First segment headings should be ~0
    assert pytest.approx(wps[0].yaw, abs=0.05) == 0.0

    # Second segment headings should be ~pi/2
    assert pytest.approx(wps[-1].yaw, abs=0.05) == math.pi / 2.0
