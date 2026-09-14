import numpy as np
import pytest

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.pose_graph_optimizer import PoseGraphOptimizer
from naviguard_slam.place_recognition import LoopClosureConstraint


def test_pose_graph_odometry_chain():
    optimizer = PoseGraphOptimizer()
    kpts = np.empty((0, 2), dtype=np.float32)

    # 3 Keyframes in a straight line
    kf0 = Keyframe(0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kpts)
    kf1 = Keyframe(1, 2.0, (1.1, 0.0, 0.0), (1.0, 0.0, 0.0), kpts)  # slight odom error
    kf2 = Keyframe(2, 3.0, (2.2, 0.0, 0.0), (2.0, 0.0, 0.0), kpts)

    # Odometry edges
    optimizer.add_odometry_edge(0, 1, dx=1.0, dy=0.0, dtheta=0.0)
    optimizer.add_odometry_edge(1, 2, dx=1.0, dy=0.0, dtheta=0.0)

    keyframes = [kf0, kf1, kf2]
    success, cost = optimizer.optimize(keyframes)

    assert success is True
    assert pytest.approx(kf0.pose_map[0], abs=1e-3) == 0.0
    assert pytest.approx(kf1.pose_map[0], abs=1e-2) == 1.0
    assert pytest.approx(kf2.pose_map[0], abs=1e-2) == 2.0


def test_pose_graph_loop_closure():
    optimizer = PoseGraphOptimizer()
    kpts = np.empty((0, 2), dtype=np.float32)

    # Create 4 keyframes forming a square loop: (0,0) -> (2,0) -> (2,2) -> (0,2) -> back to (0,0)
    # Drift is introduced: final keyframe drifts to (0.3, 2.3)
    kf0 = Keyframe(0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kpts)
    kf1 = Keyframe(1, 2.0, (2.0, 0.0, 0.0), (2.0, 0.0, 0.0), kpts)
    kf2 = Keyframe(2, 3.0, (2.0, 2.0, np.pi/2), (2.0, 2.0, np.pi/2), kpts)
    kf3 = Keyframe(3, 4.0, (0.3, 2.3, np.pi), (0.3, 2.3, np.pi), kpts)

    optimizer.add_odometry_edge(0, 1, 2.0, 0.0, 0.0)
    optimizer.add_odometry_edge(1, 2, 0.0, 2.0, np.pi/2)
    optimizer.add_odometry_edge(2, 3, 0.0, 2.0, np.pi/2)

    # Add loop closure constraint between kf3 and kf0 (closing the square loop)
    loop_edge = LoopClosureConstraint(
        from_kf_id=3,
        to_kf_id=0,
        relative_pose=(0.0, 2.0, -np.pi),
        inlier_count=30,
        confidence_score=0.9,
    )
    optimizer.add_loop_closure_edge(loop_edge)

    keyframes = [kf0, kf1, kf2, kf3]
    success, cost = optimizer.optimize(keyframes)

    assert success is True
    # Anchor kf0 remains at origin
    assert pytest.approx(kf0.pose_map[0], abs=1e-2) == 0.0
    assert pytest.approx(kf0.pose_map[1], abs=1e-2) == 0.0
    # kf3 should be pulled closer to (0.0, 2.0)
    assert abs(kf3.pose_map[0] - 0.0) < 0.2
    assert abs(kf3.pose_map[1] - 2.0) < 0.2
