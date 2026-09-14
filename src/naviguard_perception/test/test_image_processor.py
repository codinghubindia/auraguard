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
