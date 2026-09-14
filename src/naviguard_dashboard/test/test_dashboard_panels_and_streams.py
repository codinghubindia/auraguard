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


def test_heatmap_stream_and_trajectory_status():
    cache = StateCache()
    # Check heatmap stream is initialized
    assert "heatmap" in cache.streams
    
    # Set heatmap JPEG frame
    hm_img = np.zeros((100, 100, 3), dtype=np.uint8)
    hm_img[:] = (200, 50, 20)
    _, enc_hm = cv2.imencode('.jpg', hm_img)
    cache.set_jpeg_frame("heatmap", enc_hm.tobytes())
    assert cache.get_jpeg_frame("heatmap") == enc_hm.tobytes()

    # Test trajectory status
    cache.set_trajectory_status({
        "can_pass": True,
        "status": "PASS",
        "min_clearance_m": 0.85,
        "in_small_gap": True,
        "corridor_width_m": 0.62,
    })
    snap = cache.get_snapshot()
    assert "trajectory_status" in snap
    assert snap["trajectory_status"]["can_pass"] is True
    assert snap["trajectory_status"]["status"] == "PASS"
    assert snap["trajectory_status"]["in_small_gap"] is True
    assert snap["trajectory_status"]["corridor_width_m"] == 0.62


def test_lidar_telemetry_cache():
    cache = StateCache()
    snap = cache.get_snapshot()
    assert "lidar_telemetry" in snap
    assert snap["lidar_telemetry"]["closest_distance_m"] == 99.0

    cache.set_lidar_telemetry({
        "num_obstacles": 2,
        "closest_distance_m": 1.45,
        "critical_hazard": False,
        "corridor_clearance": {
            "left_clearance_m": 0.45,
            "right_clearance_m": 0.52,
            "available_gap_m": 0.97,
            "centering_offset_m": -0.035,
        },
    })
    snap2 = cache.get_snapshot()
    assert snap2["lidar_telemetry"]["closest_distance_m"] == 1.45
    assert snap2["lidar_telemetry"]["corridor_clearance"]["available_gap_m"] == 0.97


def test_recovery_checkpoints_marker_parsing():
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point

    cache = StateCache()
    marker_arr = MarkerArray()
    m = Marker()
    m.ns = "recovery_checkpoints"
    m.type = Marker.SPHERE_LIST
    for x, y in [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = 0.0
        m.points.append(p)
    marker_arr.markers.append(m)

    ckpts = []
    for mark in marker_arr.markers:
        if mark.ns in ("recovery_checkpoints", "checkpoints"):
            if mark.points:
                for idx, pt in enumerate(mark.points):
                    ckpts.append({"x": round(float(pt.x), 2), "y": round(float(pt.y), 2), "id": idx + 1})
    cache.set_checkpoints(ckpts)

    snap = cache.get_snapshot()
    assert len(snap["checkpoints"]) == 3
    assert snap["checkpoints"][0] == {"x": 1.0, "y": 2.0, "id": 1}
    assert snap["checkpoints"][2] == {"x": 5.0, "y": 6.0, "id": 3}


