"""Unit tests for LidarObstacleDetector."""

import math
import numpy as np
import pytest
from sensor_msgs.msg import LaserScan
from naviguard_perception.lidar_obstacle_detector import LidarObstacleDetector


def make_scan_msg(ranges, angle_min=-math.pi, angle_max=math.pi, r_min=0.10, r_max=25.0):
    msg = LaserScan()
    msg.header.frame_id = "lidar_link"
    msg.angle_min = angle_min
    msg.angle_max = angle_max
    n = len(ranges)
    msg.angle_increment = (angle_max - angle_min) / float(max(1, n))
    msg.range_min = r_min
    msg.range_max = r_max
    msg.ranges = [float(r) for r in ranges]
    return msg


def test_empty_scan():
    detector = LidarObstacleDetector()
    msg = make_scan_msg([])
    res = detector.process_scan(msg)
    assert len(res["valid_points"]) == 0
    assert len(res["obstacles"]) == 0
    assert res["critical_hazard"] is False


def test_obstacle_clustering_and_position():
    detector = LidarObstacleDetector()
    # 360 rays covering [-pi, pi].
    # Place an obstacle directly in front at 0 deg (x=1.5m, y=0.0m)
    n = 360
    ranges = [20.0] * n
    # Ray at index 180 is angle 0 (forward)
    center_idx = 180
    for i in range(center_idx - 5, center_idx + 6):
        ranges[i] = 1.50

    msg = make_scan_msg(ranges)
    res = detector.process_scan(msg)

    assert len(res["obstacles"]) >= 1
    obs = res["obstacles"][0]
    # Centroid x should be approximately 1.5m
    assert 1.40 <= obs["x_base"] <= 1.60
    assert abs(obs["y_base"]) < 0.15
    assert obs["zone"] == "FRONT_CENTER"
    assert obs["confirmed"] is True
    assert obs["traversable"] is False


def test_chassis_self_filtering():
    detector = LidarObstacleDetector()
    # Points inside chassis envelope (-0.28 to +0.28, -0.24 to +0.24) should be filtered
    assert detector.is_inside_chassis(0.10, 0.05) is True
    assert detector.is_inside_chassis(0.50, 0.0) is False
    assert detector.is_inside_chassis(-0.60, 0.0) is False


def test_critical_collision_hazard():
    detector = LidarObstacleDetector(critical_stop_distance_m=0.38)
    n = 360
    ranges = [20.0] * n
    # Obstacle very close ahead: at 0.45m forward (front bumper at 0.28m, so 0.17m in front of bumper!)
    for i in range(175, 185):
        ranges[i] = 0.45

    msg = make_scan_msg(ranges)
    res = detector.process_scan(msg)
    assert res["critical_hazard"] is True
    assert len(res["obstacles"]) >= 1
    assert res["obstacles"][0]["emergency"] is True


def test_corridor_clearance_and_centering():
    detector = LidarObstacleDetector()
    n = 360
    ranges = [20.0] * n
    # Gateway at x=1.5m: tree on left at y=0.40m, tree on right at y=-0.35m
    # Left tree: angle = atan2(0.40, 1.5) ~ +15 deg
    # Right tree: angle = atan2(-0.35, 1.5) ~ -13 deg
    deg_per_index = 360.0 / n
    left_idx = int(180 + 15.0 / deg_per_index)
    right_idx = int(180 - 13.0 / deg_per_index)

    for i in range(left_idx - 3, left_idx + 4):
        ranges[i] = math.hypot(1.5, 0.40)
    for i in range(right_idx - 3, right_idx + 4):
        ranges[i] = math.hypot(1.5, -0.35)

    msg = make_scan_msg(ranges)
    res = detector.process_scan(msg)
    corridor = res["corridor_clearance"]
    assert corridor["left_clearance_m"] < 1.0
    assert corridor["right_clearance_m"] < 1.0
    # Left clearance ~0.40m, right clearance ~0.35m
    # centering_offset = (0.40 - 0.35) / 2 = +0.025m (robot should shift slightly left away from right obstacle)
    assert corridor["centering_offset_m"] > 0.0
