import pytest
import cv2
import numpy as np
from naviguard_dashboard.state_cache import StateCache


def test_subsystem_health_12_nodes():
    cache = StateCache()
    expected_subsystems = [
        "Gazebo", "Camera", "IMU", "Odometry", "Perception",
        "Visual Odometry", "State Estimation", "SLAM",
        "Confidence", "Recovery", "Navigation", "Dashboard"
    ]
    snap = cache.get_snapshot()
    assert len(snap["subsystems"]) == 12
    for s in expected_subsystems:
        assert s in snap["subsystems"]

    # Test Dashboard is active by default
    assert snap["subsystems"]["Dashboard"]["status"] == "ONLINE"
    assert snap["subsystems"]["Dashboard"]["rate"] == 1.0


def test_independent_stream_isolation():
    cache = StateCache()
    # Create distinct test JPEG frames
    raw_img = np.zeros((100, 100, 3), dtype=np.uint8)
    raw_img[:] = (255, 0, 0)
    _, enc_raw = cv2.imencode('.jpg', raw_img)
    cache.set_jpeg_frame("raw", enc_raw.tobytes())

    # Raw is present, others should be None (no cross-contamination or false fallback)
    assert cache.get_jpeg_frame("raw") == enc_raw.tobytes()
    assert cache.get_jpeg_frame("perception") is None
    assert cache.get_jpeg_frame("segmentation") is None
    assert cache.get_jpeg_frame("vo") is None
    assert cache.get_jpeg_frame("chase") is None

    # Now set segmentation
    seg_img = np.zeros((100, 100, 3), dtype=np.uint8)
    seg_img[:] = (0, 255, 0)
    _, enc_seg = cv2.imencode('.jpg', seg_img)
    cache.set_jpeg_frame("segmentation", enc_seg.tobytes())

    assert cache.get_jpeg_frame("segmentation") == enc_seg.tobytes()
    assert cache.get_jpeg_frame("chase") is None


def test_replan_diagnostics_cache():
    cache = StateCache()
    snap = cache.get_snapshot()
    assert "replan_diagnostics" in snap
    assert snap["replan_diagnostics"]["status"] == "NOMINAL"

    # Update replan diagnostics
    cache.replan_diagnostics.update({
        "replan_reason": "PATH_BLOCKED_AT_WAYPOINT_1_TO_2",
        "status": "REPLAN_SUCCEEDED",
        "old_path_length_m": 8.50,
        "new_path_length_m": 9.20,
        "path_difference_m": 0.70,
        "blocked_regions_count": 1,
    })

    snap2 = cache.get_snapshot()
    assert snap2["replan_diagnostics"]["replan_reason"] == "PATH_BLOCKED_AT_WAYPOINT_1_TO_2"
    assert snap2["replan_diagnostics"]["old_path_length_m"] == 8.50
    assert snap2["replan_diagnostics"]["new_path_length_m"] == 9.20
    assert snap2["replan_diagnostics"]["path_difference_m"] == 0.70
    assert snap2["replan_diagnostics"]["blocked_regions_count"] == 1


def test_safe_write_broken_pipe_suppression():
    from naviguard_dashboard.web_server import NaviguardRequestHandler

    class DummyHandler(NaviguardRequestHandler):
        def __init__(self, raise_on_write=False):
            self.wfile = self
            self.raise_on_write = raise_on_write
            self.written_bytes = b""

        def write(self, data):
            if self.raise_on_write:
                raise BrokenPipeError("Client disconnected prematurely")
            self.written_bytes += data

    # 1. Normal write succeeds
    handler_ok = DummyHandler(raise_on_write=False)
    assert handler_ok._safe_write(b"hello world") is True
    assert handler_ok.written_bytes == b"hello world"

    # 2. Broken pipe does not raise exception, returns False
    handler_broken = DummyHandler(raise_on_write=True)
    assert handler_broken._safe_write(b"data stream") is False

