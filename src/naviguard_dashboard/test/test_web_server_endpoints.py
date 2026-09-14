import json
import os
import threading
import time
import urllib.request
import urllib.parse
from http.server import ThreadingHTTPServer
import pytest

from naviguard_dashboard.state_cache import StateCache
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
from naviguard_dashboard.goal_validator import GoalValidator
from naviguard_dashboard.web_server import NaviguardRequestHandler


@pytest.fixture(scope="module")
def running_test_server():
    """Spin up a live NaviguardRequestHandler server on an ephemeral OS port for integration testing."""
    cache = StateCache()
    conv = MapCoordinateConverter(
        resolution=0.05, width_cells=600, height_cells=600,
        origin_x=-15.0, origin_y=-15.0,
    )
    validator = GoalValidator(conv)

    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'naviguard_dashboard', 'static'))

    NaviguardRequestHandler.state_cache = cache
    NaviguardRequestHandler.coord_converter = conv
    NaviguardRequestHandler.goal_validator = validator
    NaviguardRequestHandler.static_dir = static_dir

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(('127.0.0.1', 0), NaviguardRequestHandler)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    yield f"http://127.0.0.1:{port}", cache

    server.shutdown()
    server.server_close()


def test_http_index_html_contains_all_11_layers_and_mission_status(running_test_server):
    """Verify index.html serves with 11-layer map controls and mission status panel."""
    base_url, _ = running_test_server
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        html = resp.read().decode('utf-8')

        # Check for mission status panel
        assert "missionStatusBox" in html
        assert "MISSION STATUS" in html

        # Check for all layer controls
        assert "BASE" in html
        assert "SLAM" in html
        assert "TRAV" in html
        assert "PATH" in html
        assert "TRAJ" in html
        assert "OBST" in html
        assert "CKPT" in html
        assert "EVENTS" in html
        assert "RESET LAYERS" in html

        # Verify raw video panel is replaced by heatmap panel
        assert "boxHeatmap" in html
        assert "camHeatmap" in html
        assert "boxRaw" not in html


def test_http_api_status_snapshot(running_test_server):
    """Verify /api/status returns complete telemetry dictionary."""
    base_url, cache = running_test_server
    req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert "system_status" in data
        assert "subsystems" in data
        assert "mission_state" in data
        assert "stream_metrics" in data
        assert "raw" in data["stream_metrics"]
        assert "heatmap" in data["stream_metrics"]
        assert "perception" in data["stream_metrics"]
        assert "segmentation" in data["stream_metrics"]
        assert "vo" in data["stream_metrics"]
        assert "chase" in data["stream_metrics"]
        assert "trajectory_status" in data
        assert data["trajectory_status"]["can_pass"] is True


def test_http_api_camera_headers_and_status(running_test_server):
    """Verify /api/camera provides latency and frame age headers."""
    base_url, cache = running_test_server
    # Push a known frame
    cache.set_jpeg_frame("raw", b"\xff\xd8\xff\xe0\x00\x10JFIFdummy", time.time(), 4.5, 15.0)

    req = urllib.request.Request(f"{base_url}/api/camera?type=raw")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/jpeg"
        assert "X-Frame-Timestamp" in resp.headers
        assert "X-Frame-Age-Ms" in resp.headers
        assert "X-Stream-Status" in resp.headers


def test_http_api_reset_mission(running_test_server):
    """Verify /api/reset_mission resets mission and clears failure records."""
    base_url, cache = running_test_server

    # Set failure
    cache.set_failure_record({
        "failure_code": "NO_SAFE_PATH",
        "human_reason": "No safe collision-free path exists",
    })
    assert cache.get_snapshot().get("failure_record") is not None

    # Test GET reset
    req = urllib.request.Request(f"{base_url}/api/reset_mission")
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data.get("success") is True

    # Check that cache cleared
    assert cache.get_snapshot().get("failure_record") is None
