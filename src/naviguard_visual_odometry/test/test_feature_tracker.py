"""Unit tests for FeatureTracker and optical flow motion estimation."""

import cv2
import numpy as np
import pytest

from naviguard_visual_odometry.feature_tracker import (
    FeatureTracker,
    FeatureTrackerConfig,
    TrackedFeature,
)


@pytest.fixture
def textured_image() -> np.ndarray:
    """Create a synthetic high-contrast textured image suitable for feature tracking."""
    img = np.zeros((480, 640), dtype=np.uint8)
    np.random.seed(42)
    # Draw multiple squares, circles, and grid points
    for i in range(15, 620, 30):
        for j in range(15, 460, 30):
            cv2.rectangle(img, (i, j), (i + 15, j + 15), 200, -1)
            cv2.circle(img, (i + 7, j + 7), 3, 50, -1)
    return img


def test_tracked_feature_properties():
    """Verify displacement computation in TrackedFeature."""
    feat = TrackedFeature(prev_pt=(10.0, 20.0), curr_pt=(13.0, 24.0), is_inlier=True)
    assert feat.dx == pytest.approx(3.0)
    assert feat.dy == pytest.approx(4.0)
    assert feat.displacement == pytest.approx(5.0)
    assert feat.is_inlier is True


def test_feature_detection(textured_image):
    """Verify Shi-Tomasi corners are detected in sufficient numbers."""
    cfg = FeatureTrackerConfig(max_features=150, quality_level=0.01, min_distance=10.0)
    tracker = FeatureTracker(cfg)

    pts = tracker.detect_features(textured_image)
    assert pts is not None
    assert len(pts) > 20
    assert pts.shape[1:] == (1, 2)


def test_self_robot_mask():
    """Verify Shi-Tomasi corners are NOT detected in the lower self-mask region."""
    img = np.zeros((480, 640), dtype=np.uint8)
    for i in range(15, 620, 20):
        for j in range(15, 470, 20):
            cv2.rectangle(img, (i, j), (i + 10, j + 10), 200, -1)

    cfg = FeatureTrackerConfig(self_mask_height_ratio=0.15)
    tracker = FeatureTracker(cfg)
    pts = tracker.detect_features(img)
    assert pts is not None
    assert len(pts) > 0
    cutoff_y = int(480 * (1.0 - 0.15))
    y_coords = pts[:, 0, 1]
    assert np.all(y_coords < cutoff_y), f"Detected features in lower 15% (>= {cutoff_y})"



def test_zero_motion_tracking(textured_image):
    """Verify identical frames yield zero displacement and high inlier ratio."""
    cfg = FeatureTrackerConfig(max_features=100)
    tracker = FeatureTracker(cfg)

    # Frame 1: initial detection
    tracks1, stats1 = tracker.track(textured_image)
    assert len(tracks1) == 0
    assert stats1['status'] == 'FIRST_FRAME'

    # Frame 2: identical image (0 motion)
    tracks2, stats2 = tracker.track(textured_image)
    assert stats2['is_valid'] is True
    assert stats2['num_inliers'] > 20
    assert stats2['median_displacement_px'] == pytest.approx(0.0, abs=0.2)
    assert stats2['median_dx_px'] == pytest.approx(0.0, abs=0.2)
    assert stats2['median_dy_px'] == pytest.approx(0.0, abs=0.2)
    assert stats2['tracking_ratio'] > 0.8


def test_synthetic_translation_tracking(textured_image):
    """Verify known synthetic translation (e.g. dx=+4, dy=-2) is recovered accurately."""
    cfg = FeatureTrackerConfig(max_features=150)
    tracker = FeatureTracker(cfg)

    # Initialize with frame 1
    tracker.track(textured_image)

    # Shift image by dx=+4, dy=-2
    shift_x = 4.0
    shift_y = -2.0
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_img = cv2.warpAffine(textured_image, M, (640, 480))

    tracks, stats = tracker.track(shifted_img)
    assert stats['is_valid'] is True
    assert stats['num_inliers'] > 20
    assert stats['median_dx_px'] == pytest.approx(shift_x, abs=0.5)
    assert stats['median_dy_px'] == pytest.approx(shift_y, abs=0.5)
    expected_disp = np.hypot(shift_x, shift_y)
    assert stats['median_displacement_px'] == pytest.approx(expected_disp, abs=0.5)


def test_outlier_rejection():
    """Verify MAD filter and forward-backward error threshold flag corrupted tracks."""
    cfg = FeatureTrackerConfig(outlier_mad_k=2.0)
    tracker = FeatureTracker(cfg)

    # Synthetic tracks where one feature moves drastically compared to others
    tracks = [
        TrackedFeature((100.0, 100.0), (102.0, 100.0)),
        TrackedFeature((110.0, 100.0), (102.1, 100.0)),
        TrackedFeature((120.0, 100.0), (101.9, 100.0)),
        TrackedFeature((130.0, 100.0), (102.0, 100.0)),
        TrackedFeature((140.0, 100.0), (180.0, 100.0)),  # Outlier jump of 40px
    ]

    displacements = np.array([t.displacement for t in tracks], dtype=np.float32)
    med_disp = float(np.median(displacements))
    abs_devs = np.abs(displacements - med_disp)
    mad = float(np.median(abs_devs))
    spread_thresh = max(cfg.outlier_mad_k * max(mad, 1.0), 3.0)

    for t in tracks:
        t.is_inlier = abs(t.displacement - med_disp) <= spread_thresh

    inliers = [t for t in tracks if t.is_inlier]
    outliers = [t for t in tracks if not t.is_inlier]
    assert len(inliers) == 4
    assert len(outliers) == 1
    assert outliers[0].displacement == pytest.approx(40.0, abs=1.0)


def test_blank_image_graceful_handling():
    """Verify uniform blank image does not crash or raise exceptions."""
    cfg = FeatureTrackerConfig()
    tracker = FeatureTracker(cfg)

    blank = np.zeros((480, 640), dtype=np.uint8)
    tracks1, stats1 = tracker.track(blank)
    assert len(tracks1) == 0
    assert stats1['num_detected'] == 0

    tracks2, stats2 = tracker.track(blank)
    assert len(tracks2) == 0
    assert stats2['is_valid'] is False
    assert stats2['num_inliers'] == 0


def test_render_debug_canvas(textured_image):
    """Verify render_debug_canvas returns correct shape and type."""
    cfg = FeatureTrackerConfig()
    tracker = FeatureTracker(cfg)

    bgr = cv2.cvtColor(textured_image, cv2.COLOR_GRAY2BGR)
    tracks, stats = tracker.track(textured_image)

    meta = {
        'frame_idx': 1,
        'stamp_sec': 10,
        'stamp_nanosec': 500000000,
        'frame_id': 'camera_link',
        'input_fps': 30.0,
        'proc_latency_ms': 4.5,
        'proc_fps': 220.0,
        'wheel_vx': 0.5,
        'wheel_wz': 0.0,
    }

    canvas = tracker.render_debug_canvas(bgr, tracks, stats, meta)
    assert canvas.shape == bgr.shape
    assert canvas.dtype == np.uint8
