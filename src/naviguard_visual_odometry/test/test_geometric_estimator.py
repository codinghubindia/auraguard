"""Unit tests for GeometricMotionEstimator, Essential matrix, and pose recovery."""

import cv2
import numpy as np
import pytest

from naviguard_visual_odometry.geometric_estimator import (
    GeometricEstimatorConfig,
    GeometricMotionEstimate,
    GeometricMotionEstimator,
)


@pytest.fixture
def camera_matrix() -> np.ndarray:
    """Standard camera intrinsic matrix (640x480, f=381.36)."""
    return np.array([
        [381.36, 0.0, 320.0],
        [0.0, 381.36, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def synthetic_3d_points() -> np.ndarray:
    """Generate 60 3D scene points in front of the camera (Z in [2.0, 8.0] meters)."""
    np.random.seed(123)
    xs = np.random.uniform(-1.5, 1.5, 60)
    ys = np.random.uniform(-0.8, 0.8, 60)
    zs = np.random.uniform(2.5, 6.0, 60)
    return np.column_stack([xs, ys, zs])


def project_points(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points into camera image plane using [R | t]."""
    # Transform: P_cam = R @ P_world + t
    pts_cam = (R @ pts3d.T).T + t.reshape(1, 3)
    x_proj = pts_cam[:, 0] / pts_cam[:, 2]
    y_proj = pts_cam[:, 1] / pts_cam[:, 2]
    u = K[0, 0] * x_proj + K[0, 2]
    v = K[1, 1] * y_proj + K[1, 2]
    return np.column_stack([u, v]).astype(np.float32)


def test_degenerate_small_baseline(camera_matrix):
    """Verify that near-zero baseline returns DEGENERATE_SMALL_BASELINE."""
    cfg = GeometricEstimatorConfig(min_baseline_disp_px=0.6)
    estimator = GeometricMotionEstimator(cfg)

    pts0 = np.random.uniform(100, 400, (30, 2)).astype(np.float32)
    pts1 = pts0 + 0.05  # sub-pixel displacement (0.07px)

    est = estimator.estimate(pts0, pts1, camera_matrix, median_displacement_px=0.07)
    assert est.is_valid is False
    assert est.status == "DEGENERATE_SMALL_BASELINE"
    assert est.translation_scale_status == "UNKNOWN"
    assert est.rot_angle_deg == 0.0


def test_insufficient_correspondences(camera_matrix):
    """Verify that fewer than min_inliers returns INSUFFICIENT_CORRESPONDENCES."""
    cfg = GeometricEstimatorConfig(min_inliers=15)
    estimator = GeometricMotionEstimator(cfg)

    pts0 = np.random.uniform(100, 400, (8, 2)).astype(np.float32)
    pts1 = pts0 + 2.0

    est = estimator.estimate(pts0, pts1, camera_matrix, median_displacement_px=2.0)
    assert est.is_valid is False
    assert est.status == "INSUFFICIENT_CORRESPONDENCES"
    assert est.translation_scale_status == "UNKNOWN"


def test_pure_forward_translation(camera_matrix, synthetic_3d_points):
    """Verify forward camera motion yields t_z ~= 1.0, R ~= I, and scale is UNKNOWN."""
    cfg = GeometricEstimatorConfig(min_inliers=15, min_baseline_disp_px=0.5)
    estimator = GeometricMotionEstimator(cfg)

    # Frame 0: Camera at origin
    R0 = np.eye(3)
    t0 = np.zeros(3)
    pts0 = project_points(synthetic_3d_points, camera_matrix, R0, t0)

    # Frame 1: Camera moves forward along +Z by 0.3 meters
    # Transformation from cam0 to cam1: P_cam1 = P_cam0 - [0, 0, 0.3]
    R_rel = np.eye(3)
    t_rel = np.array([0.0, 0.0, -0.3])  # Camera moved +Z, so scene shifts -Z
    pts1 = project_points(synthetic_3d_points, camera_matrix, R_rel, t_rel)

    # Median displacement in pixels
    disp = float(np.median(np.hypot(pts1[:, 0] - pts0[:, 0], pts1[:, 1] - pts0[:, 1])))

    est = estimator.estimate(pts0, pts1, camera_matrix, median_displacement_px=disp)

    assert est.is_valid is True
    assert est.status == "GEOMETRIC_MOTION_OK"
    assert est.translation_scale_status == "UNKNOWN"
    assert est.num_inliers >= 30

    # Rotation should be negligible (< 1 degree)
    assert est.rot_angle_deg == pytest.approx(0.0, abs=1.0)

    # Unit translation direction should be primarily forward (Z-axis in camera frame)
    # Note: recoverPose returns camera displacement t such that x2 ~ R*x1 + t
    assert abs(est.unit_tz) > 0.85
    # Unit norm verification
    norm_t = np.hypot(np.hypot(est.unit_tx, est.unit_ty), est.unit_tz)
    assert norm_t == pytest.approx(1.0, abs=1e-5)


def test_pure_yaw_rotation(camera_matrix, synthetic_3d_points):
    """Verify camera rotation around optical Y-axis yields accurate yaw angle."""
    cfg = GeometricEstimatorConfig(min_inliers=15, min_baseline_disp_px=0.5)
    estimator = GeometricMotionEstimator(cfg)

    # Frame 0: Camera at origin
    R0 = np.eye(3)
    t0 = np.zeros(3)
    pts0 = project_points(synthetic_3d_points, camera_matrix, R0, t0)

    # Frame 1: Rotate 2.5 degrees about Y-axis with small translation (to provide epipolar baseline)
    theta = np.radians(2.5)
    R_yaw = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    t_rel = np.array([0.05, 0.0, 0.1])
    pts1 = project_points(synthetic_3d_points, camera_matrix, R_yaw, t_rel)

    disp = float(np.median(np.hypot(pts1[:, 0] - pts0[:, 0], pts1[:, 1] - pts0[:, 1])))

    est = estimator.estimate(pts0, pts1, camera_matrix, median_displacement_px=disp)

    assert est.is_valid is True
    assert est.status == "GEOMETRIC_MOTION_OK"
    assert est.translation_scale_status == "UNKNOWN"
    # Recovered rotation angle should be close to 2.5 degrees
    assert est.rot_angle_deg == pytest.approx(2.5, abs=0.8)
    assert est.yaw_deg == pytest.approx(2.5, abs=0.8)
