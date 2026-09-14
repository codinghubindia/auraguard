"""Unit tests for FrameValidator and camera optical convention."""

import numpy as np
import pytest
from naviguard_sensor_sync.frame_validator import FrameValidator


def test_camera_optical_convention_math():
    """Verify quaternion math correctly represents REP-103 optical convention."""
    # Joint rpy in URDF: roll = -pi/2, pitch = 0, yaw = -pi/2
    # In quaternion: q = [-0.5, 0.5, -0.5, 0.5]
    qx, qy, qz, qw = -0.5, 0.5, -0.5, 0.5
    R = FrameValidator._quat_to_rot_matrix(qx, qy, qz, qw)

    # In optical frame:
    # +Z is forward -> in camera body should be +X: [1, 0, 0]
    z_opt_in_cam = R @ np.array([0, 0, 1])
    assert z_opt_in_cam == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)

    # +X is right -> in camera body should be -Y: [0, -1, 0]
    x_opt_in_cam = R @ np.array([1, 0, 0])
    assert x_opt_in_cam == pytest.approx([0.0, -1.0, 0.0], abs=1e-5)

    # +Y is down -> in camera body should be -Z: [0, 0, -1]
    y_opt_in_cam = R @ np.array([0, 1, 0])
    assert y_opt_in_cam == pytest.approx([0.0, 0.0, -1.0], abs=1e-5)
