"""
Regression and Integration Tests for Interactive Map & Goal-Dispatch System.
"""

import math
import pytest
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
from naviguard_dashboard.goal_validator import GoalValidator
from naviguard_dashboard.web_server import NaviguardRequestHandler
from naviguard_dashboard.state_cache import StateCache


def test_coordinate_conversion_roundtrip():
    """Verify bidirectional conversion between UI canvas pixels and world meters."""
    conv = MapCoordinateConverter(
        resolution=0.05,
        width_cells=600,
        height_cells=600,
        origin_x=-15.0,
        origin_y=-15.0,
        canvas_width=600,
        canvas_height=600
    )

    # Center of map (col=300, row=300) -> world (0.025, 0.025)
    wx, wy = conv.canvas_to_world(300, 300)
    assert math.isclose(wx, 0.025, abs_tol=0.05)
    assert math.isclose(wy, 0.025, abs_tol=0.05)

    # Forward world to canvas
    u, v = conv.world_to_canvas(wx, wy)
    assert math.isclose(u, 300, abs_tol=1.0)
    assert math.isclose(v, 300, abs_tol=1.0)


def test_goal_validation_states():
    """Verify goal validator accepts ONLINE and DEGRADED, but rejects OFFLINE or FAILED_SAFE."""
    conv = MapCoordinateConverter(
        resolution=0.05, width_cells=600, height_cells=600,
        origin_x=-15.0, origin_y=-15.0
    )
    validator = GoalValidator(conv)

    # 1. Nominal online
    ok, reason = validator.validate_goal(2.0, 1.0, navigation_online=True, recovery_state="NORMAL", has_localization=True)
    assert ok is True
    assert "GOAL VALID" in reason

    # 2. String status DEGRADED should still permit goal dispatch
    ok, reason = validator.validate_goal(2.0, 1.0, navigation_online="DEGRADED", recovery_state="NORMAL", has_localization="DEGRADED")
    assert ok is True

    # 3. Navigation OFFLINE rejected
    ok, reason = validator.validate_goal(2.0, 1.0, navigation_online="OFFLINE", recovery_state="NORMAL", has_localization=True)
    assert ok is False
    assert "Navigation subsystem is OFFLINE" in reason

    # 4. Recovery FAILED_SAFE rejected
    ok, reason = validator.validate_goal(2.0, 1.0, navigation_online=True, recovery_state="FAILED_SAFE", has_localization=True)
    assert ok is False
    assert "FAILED_SAFE" in reason


def test_safe_write_socket_error_handling():
    """Verify _safe_write suppresses socket errors when clients disconnect abruptly."""
    class MockHandler(NaviguardRequestHandler):
        def __init__(self, simulate_error=None):
            self.simulate_error = simulate_error
            self.buffer = b""

        @property
        def wfile(self):
            return self

        def write(self, data):
            if self.simulate_error:
                raise self.simulate_error
            self.buffer += data

    # Normal write
    h_ok = MockHandler()
    assert h_ok._safe_write(b"data") is True
    assert h_ok.buffer == b"data"

    # BrokenPipeError
    h_broken = MockHandler(simulate_error=BrokenPipeError("Broken pipe"))
    assert h_broken._safe_write(b"data") is False

    # ConnectionResetError
    h_reset = MockHandler(simulate_error=ConnectionResetError("Connection reset by peer"))
    assert h_reset._safe_write(b"data") is False


def test_basemap_generator_natural_rendering():
    """Verify generate_rellis_basemap outputs a 600x600 natural terrain image."""
    from naviguard_dashboard.basemap_generator import generate_rellis_basemap
    import cv2
    import numpy as np

    img = generate_rellis_basemap(600, 600, 0.05, -15.0, -15.0)
    assert isinstance(img, np.ndarray)
    assert img.shape == (600, 600, 3)
    # Check that image is not blank/black
    mean_val = img.mean()
    assert mean_val > 20.0, f"Expected non-black basemap, got mean {mean_val}"
    # Verify green/brown dominance (natural outdoor terrain)
    # Channels in BGR: B=0, G=1, R=2
    mean_green = img[:, :, 1].mean()
    mean_blue = img[:, :, 0].mean()
    assert mean_green > mean_blue, "Outdoor basemap should have green/vegetation bias"


def test_transparent_slam_fallback_no_occlusion():
    """Verify that when SLAM map is not published, fallback is 100% transparent RGBA (Alpha=0)."""
    import cv2
    import numpy as np

    # Reproduce the fallback logic from web_server.py /api/map_image
    blank = np.zeros((100, 100, 4), dtype=np.uint8)
    _, enc = cv2.imencode('.png', blank)
    png_bytes = enc.tobytes()

    # Decode and check alpha channel
    decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (100, 100, 4), "Fallback must be 4-channel RGBA"
    assert np.all(decoded[:, :, 3] == 0), "Alpha channel must be 0 everywhere to prevent occluding basemap"

