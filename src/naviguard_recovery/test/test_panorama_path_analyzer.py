import math
import numpy as np
import pytest
from naviguard_recovery.panorama_path_analyzer import PanoramaPathAnalyzer


def test_panorama_analyzer_clear_field():
    analyzer = PanoramaPathAnalyzer(camera_height_m=0.26, camera_pitch_rad=0.05, num_sectors=36)
    
    # Create synthetic clear panorama image (1024x320, 3 channels)
    img = np.full((320, 1024, 3), 128, dtype=np.uint8)
    
    # Analyze with robot at (0.0, 0.0, 0.0) and goal ahead at (3.0, 0.0)
    res = analyzer.analyze_frame(
        panorama_bgr=img,
        robot_pose=(0.0, 0.0, 0.0),
        goal_point=(3.0, 0.0),
    )
    
    assert res["panorama_available"] is True
    assert res["clear_route_found"] is True
    assert len(res["passages"]) > 0
    assert res["escape_point"] is not None
    # Best heading should be roughly aligned with goal (near 0 radians)
    assert abs(res["best_heading_rad"]) < 0.40
    # Escape point should be in forward direction (x > 1.0)
    assert res["escape_point"][0] > 1.0


def test_panorama_analyzer_blocked_front_side_escape():
    analyzer = PanoramaPathAnalyzer(camera_height_m=0.26, camera_pitch_rad=0.05, num_sectors=36)
    
    # Create image with a large obstacle right in the center (forward view)
    img = np.full((320, 1024, 3), 200, dtype=np.uint8)
    # Center is cols 400 to 624, draw high-contrast dark obstacle from y=100 to y=280
    img[100:280, 400:624] = 20
    
    # Robot at (0.0, 0.0, 0.0), goal at (4.0, 1.0) (slightly left)
    res = analyzer.analyze_frame(
        panorama_bgr=img,
        robot_pose=(0.0, 0.0, 0.0),
        goal_point=(4.0, 1.0),
    )
    
    assert res["panorama_available"] is True
    assert res["clear_route_found"] is True
    # The selected sector should divert away from center (center is 0 rad)
    assert abs(res["best_heading_rad"]) > 0.15
    assert res["escape_point"] is not None


def test_panorama_render_annotated():
    analyzer = PanoramaPathAnalyzer(camera_height_m=0.26, camera_pitch_rad=0.05, num_sectors=36)
    img = np.full((320, 1024, 3), 100, dtype=np.uint8)
    
    res = analyzer.analyze_frame(
        panorama_bgr=img,
        robot_pose=(1.0, 2.0, 0.5),
        goal_point=(5.0, 2.0),
    )
    annotated = analyzer.render_panorama_overlay(img, res)
    assert annotated is not None
    assert annotated.shape == img.shape
    assert annotated.dtype == np.uint8

