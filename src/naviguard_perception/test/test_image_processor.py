"""Unit tests for NAVIGUARD image processor."""

import numpy as np
import pytest
from naviguard_perception.image_processor import ImageProcessor, ImageProcessorConfig


def test_processor_initialization():
    """Verify ImageProcessor initializes with default and custom configs."""
    processor = ImageProcessor()
    assert processor.config.display_mode == 'overlay'
    assert processor.config.enable_clahe is True

    custom_cfg = ImageProcessorConfig(display_mode='side_by_side', canny_low_threshold=30)
    custom_proc = ImageProcessor(custom_cfg)
    assert custom_proc.config.display_mode == 'side_by_side'
    assert custom_proc.config.canny_low_threshold == 30


def test_process_frame_overlay_mode():
    """Verify standard frame processing in overlay mode."""
    processor = ImageProcessor(ImageProcessorConfig(display_mode='overlay'))
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    metadata = {
        'frame_idx': 1,
        'frame_id': 'camera_link',
        'stamp_sec': 10,
        'stamp_nanosec': 500000000,
        'input_fps': 30.0,
        'proc_latency_ms': 5.0,
        'proc_fps': 200.0,
        'fx': 381.36,
        'fy': 381.36,
        'cx': 320.0,
        'cy': 240.0,
    }

    # Warm up processor
    processor.process_frame(dummy_img, metadata)

    out_img, diagnostics = processor.process_frame(dummy_img, metadata)

    assert out_img is not None
    assert out_img.shape == (480, 640, 3)
    assert out_img.dtype == np.uint8
    assert 'processing_latency_ms' in diagnostics
    assert diagnostics['processing_latency_ms'] < 100.0  # Must be fast (< 100ms)
    assert 'ground_stats' in diagnostics
    assert 'mean_brightness' in diagnostics['ground_stats']


def test_process_frame_side_by_side_mode():
    """Verify side-by-side diagnostic canvas generation."""
    processor = ImageProcessor(ImageProcessorConfig(display_mode='side_by_side'))
    dummy_img = np.full((480, 640, 3), 120, dtype=np.uint8)

    metadata = {
        'frame_idx': 2,
        'frame_id': 'camera_link',
        'stamp_sec': 11,
        'stamp_nanosec': 0,
    }

    out_img, diagnostics = processor.process_frame(dummy_img, metadata)

    assert out_img is not None
    # Side-by-side mode doubles the width (640 * 2 = 1280)
    assert out_img.shape == (480, 1280, 3)
    assert diagnostics['image_width'] == 640


def test_edge_cases():
    """Verify processor handles all-black, all-white, and varied images without error."""
    processor = ImageProcessor()

    # All black (CLAHE may slightly lift black level to ~4)
    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
    out_b, diag_b = processor.process_frame(black_img)
    assert out_b.shape == (480, 640, 3)
    assert diag_b['ground_stats']['mean_brightness'] <= 5.0

    # All white
    white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
    out_w, diag_w = processor.process_frame(white_img)
    assert out_w.shape == (480, 640, 3)
    assert diag_w['ground_stats']['mean_brightness'] > 200.0


def test_ground_roi_polygon():
    """Verify ground ROI polygon geometry is correctly formed."""
    processor = ImageProcessor()
    pts = processor.get_ground_roi_polygon(480, 640)
    assert pts.shape == (4, 2)
    # Check that y coordinates are in the lower portion of the image
    assert all(pt[1] >= 200 for pt in pts)


def test_self_robot_mask():
    """Verify self-robot mask completely excludes bumper/hood features from edges and ground stats."""
    cfg = ImageProcessorConfig(self_mask_height_ratio=0.15)
    processor = ImageProcessor(cfg)

    h, w = 480, 640
    mask_y = int(h * (1.0 - 0.15))  # 408

    # Create image with high-contrast checkerboard/noise in the bottom 15%
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    test_img[mask_y:, :] = np.random.randint(0, 255, (h - mask_y, w, 3), dtype=np.uint8)

    # Extract edges
    edges = processor.extract_structural_edges(test_img)
    # Bottom 15% must have ZERO detected edges despite noisy pattern
    assert np.count_nonzero(edges[mask_y:, :]) == 0, "Self-robot mask must suppress all edges in lower 15%"

    # Ground ROI polygon must not exceed mask boundary
    pts = processor.get_ground_roi_polygon(h, w)
    assert all(pt[1] <= mask_y for pt in pts), "Ground corridor must not extend into self-mask area"

    # Analyze ground region: edge density must be zero since bottom noise is masked
    stats = processor.analyze_ground_region(test_img, edges, pts)
    assert stats['edge_density'] == 0.0, "Masked vehicle noise must not contribute to ground edge density"


def test_generate_segmentation_view():
    """Verify classical traversability & edge segmentation view generation."""
    processor = ImageProcessor()
    h, w = 480, 640
    test_img = np.full((h, w, 3), 100, dtype=np.uint8)

    metadata = {'input_fps': 25.0}
    seg_img, seg_diag = processor.generate_segmentation_view(test_img, metadata)

    assert seg_img is not None
    assert seg_img.shape == (h, w, 3)
    assert seg_img.dtype == np.uint8
    assert 'traversability_pct' in seg_diag
    assert 'confidence_score' in seg_diag
    assert seg_diag['confidence_score'] > 0.0
    assert 'corridor_pixels' in seg_diag
    assert seg_diag['corridor_pixels'] > 0
    assert 'min_clearance_m' in seg_diag
    assert 'can_pass' in seg_diag
    assert 'trajectory_status' in seg_diag


def test_trajectory_projector():
    """Verify TrajectoryProjector produces accurate ground-to-pixel projection and clearance."""
    from naviguard_perception.trajectory_projector import TrajectoryProjector

    proj = TrajectoryProjector(camera_height_m=0.26, camera_pitch_rad=0.05, hfov_deg=80.0)

    # 1. Forward straight projection: (x=1.5m, y=0.0m) should be near center-lower frame
    pix = proj.project_ground_to_pixel(1.5, 0.0, 640, 480)
    assert pix is not None
    assert abs(pix[0] - 320) < 10  # Centered horizontally
    assert 240 < pix[1] < 480      # In lower ground half

    # 2. Pixel to ground round-trip
    gx, gy = proj.project_pixel_to_ground(pix[0], pix[1], 640, 480)
    assert abs(gx - 1.5) < 0.25
    assert abs(gy - 0.0) < 0.20

    # 3. Dynamic Trajectory Points with non-zero curvature (steering turn)
    traj = proj.generate_trajectory_points(cmd_vx=0.20, cmd_wz=0.20, max_dist_m=3.0)
    assert "center" in traj
    assert "left" in traj
    assert "right" in traj
    assert len(traj["center"]) >= 10
    # Turning left should have positive y offset for center points
    assert traj["center"][-1][1] > 0.0

    # 4. Clearance check: obstacle directly on path -> BLOCKED
    obs_blocking = [{"x_m": 1.2, "y_m": 0.0, "radius_m": 0.20}]
    traj_straight = proj.generate_trajectory_points(cmd_vx=0.20, cmd_wz=0.0, max_dist_m=3.0)
    eval_blocked = proj.evaluate_trajectory_clearance(traj_straight, obs_blocking)
    assert eval_blocked["status"] == "BLOCKED"
    assert eval_blocked["can_pass"] is False
    assert eval_blocked["collision_dist_m"] is not None

    # 5. Clearance check: obstacle far to side -> SAFE
    obs_far = [{"x_m": 1.5, "y_m": 1.2, "radius_m": 0.20}]
    eval_safe = proj.evaluate_trajectory_clearance(traj_straight, obs_far)
    assert eval_safe["status"] == "SAFE"
    assert eval_safe["can_pass"] is True


def test_generate_heatmap_view():
    """Verify real-time Distance & Proximity Heatmap generation."""
    processor = ImageProcessor()
    h, w = 480, 640
    test_img = np.full((h, w, 3), 90, dtype=np.uint8)

    metadata = {'input_fps': 20.0, 'cmd_vx': 0.15, 'cmd_wz': 0.05}
    obstacles = [{'x_m': 1.1, 'y_m': 0.1, 'radius_m': 0.25}]

    heat_img, heat_diag = processor.generate_heatmap_view(test_img, metadata, obstacles)

    assert heat_img is not None
    assert heat_img.shape == (h, w, 3)
    assert heat_img.dtype == np.uint8
    assert 'nearest_obstacle_dist_m' in heat_diag
    assert 'min_clearance_m' in heat_diag
    assert 'can_pass' in heat_diag
    assert 'trajectory_status' in heat_diag
    assert heat_diag['nearest_obstacle_dist_m'] < 2.0


def test_generate_unified_perception_view():
    """Verify Unified Multi-Spectral Perception view (Segmentation + YOLO + Perception + Trajectory)."""
    processor = ImageProcessor()
    h, w = 480, 640
    test_img = np.full((h, w, 3), 100, dtype=np.uint8)

    metadata = {'input_fps': 25.0, 'cmd_vx': 0.18, 'cmd_wz': -0.05}
    yolo_dets = [
        {'class_name': 'person', 'confidence': 0.91, 'bbox': [150, 200, 40, 80], 'traversable': False},
        {'class_name': 'car', 'confidence': 0.88, 'bbox': [350, 180, 90, 60], 'traversable': False},
    ]

    unified_img, diag = processor.generate_unified_perception_view(test_img, metadata, yolo_detections=yolo_dets)

    assert unified_img is not None
    assert unified_img.shape == (h, w, 3)
    assert unified_img.dtype == np.uint8
    assert diag['yolo_detections_count'] == 2
    assert 'min_clearance_m' in diag
    assert 'can_pass' in diag
    assert 'trajectory_status' in diag

