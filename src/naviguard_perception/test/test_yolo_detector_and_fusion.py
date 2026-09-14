"""Unit and integration tests for YOLO detector, honest hardware reporting, and perception fusion."""

import numpy as np
import pytest

from naviguard_perception.yolo_detector import YoloDetector
from naviguard_perception.perception_fusion import PerceptionFusion, FusedObstacle


def test_yolo_detector_initialization_and_device():
    """Verify YOLO detector initializes and honestly reports hardware device (CPU vs CUDA)."""
    detector = YoloDetector(conf_threshold=0.35, nms_threshold=0.45)
    assert detector.conf_threshold == 0.35
    assert detector.nms_threshold == 0.45
    # In standard CPU/container environments without CUDA GPU, detector must report CPU honestly
    assert detector.active_device in ("CPU", "CUDA")
    diag = detector.get_diagnostics()
    assert "active_device" in diag
    assert "model_path" in diag
    assert "fps" in diag
    assert "backend" in diag


def test_outdoor_ontology_coverage():
    """Verify outdoor navigation ontology covers required classes and traversability flags."""
    detector = YoloDetector()
    ontology = detector.OUTDOOR_CLASSES
    for cls in ("person", "vehicle", "tree", "rock", "log", "barrier", "pole", "mud", "bush"):
        assert cls in ontology
    # Verify non-traversable obstacle classes
    assert ontology["tree"]["traversable"] is False
    assert ontology["rock"]["traversable"] is False
    assert ontology["barrier"]["traversable"] is False
    assert ontology["person"]["traversable"] is False


def test_yolo_detector_inference_on_synthetic_frame():
    """Verify inference runs cleanly on synthetic image and produces structured detections."""
    detector = YoloDetector(conf_threshold=0.20)
    # Create synthetic image with a dark rectangle simulating an obstacle on ground
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[250:380, 260:380] = 30  # High contrast obstacle

    detections, annotated_img, metrics = detector.detect(frame)
    assert isinstance(detections, list)
    assert metrics["latency_ms"] >= 0.0
    assert annotated_img.shape == frame.shape

    # Draw overlay on frame
    vis = detector.draw_detections(frame, detections)
    assert vis.shape == frame.shape
    assert vis.dtype == np.uint8


def test_perception_fusion_temporal_persistence():
    """Verify multi-frame temporal tracking requires >= 3 hits before confirming dynamic obstacle."""
    fusion = PerceptionFusion(min_hits_to_confirm=3, max_misses_to_drop=3)

    # Initial frame: obstacle detected at (x=2.5, y=0.2)
    det1 = {
        "class_name": "rock",
        "confidence": 0.85,
        "box": [280, 200, 80, 80],
        "center": (320, 240),
        "traversable": False,
        "is_edge": False,
    }
    _, confirmed1, _ = fusion.update(detections=[det1], timestamp=1.0)
    tracked1 = fusion.get_tracked_obstacles()
    # After 1 detection, obstacle is tracked but NOT yet confirmed
    assert len(tracked1) == 1
    assert tracked1[0].hits == 1
    assert tracked1[0].confirmed is False
    assert len(confirmed1) == 0

    # Second frame (same location)
    _, confirmed2, _ = fusion.update(detections=[det1], timestamp=1.1)
    tracked2 = fusion.get_tracked_obstacles()
    assert len(tracked2) == 1
    assert tracked2[0].hits == 2
    assert tracked2[0].confirmed is False
    assert len(confirmed2) == 0

    # Third frame (same location) -> confirmed!
    _, confirmed3, _ = fusion.update(detections=[det1], timestamp=1.2)
    tracked3 = fusion.get_tracked_obstacles()
    assert len(tracked3) == 1
    assert tracked3[0].hits == 3
    assert tracked3[0].confirmed is True
    assert len(confirmed3) == 1


def test_perception_fusion_near_field_emergency_bypass():
    """Verify obstacle < 1.0m directly in path immediately triggers confirmed emergency bypass."""
    fusion = PerceptionFusion(min_hits_to_confirm=3)

    # Place detection very low in camera frame (y=450px -> close ground distance ~0.7m)
    det_close = {
        "class_name": "barrier",
        "confidence": 0.90,
        "box": [300, 440, 50, 40],
        "center": (320, 460),
        "traversable": False,
        "is_edge": False,
    }
    _, confirmed, _ = fusion.update(detections=[det_close], timestamp=2.0)
    tracked = fusion.get_tracked_obstacles()
    assert len(tracked) == 1
    obs = tracked[0]
    # If distance is < 1.0m, emergency flag is set and confirmed is True on hit 1
    if obs.x_base < 1.0:
        assert obs.emergency is True
        assert obs.confirmed is True
        assert len(confirmed) == 1


def test_camera_edge_suppression():
    """Verify false-positive suppression for detections clipped at camera image border."""
    detector = YoloDetector()
    # Detection flush against the left border (x=0)
    det_edge = {"box": [0, 100, 50, 80], "class_name": "rock", "confidence": 0.70}
    is_edge = detector.is_border_clipped(det_edge["box"], img_width=640, img_height=480, margin=5)
    assert is_edge is True

    # Detection in the center of the image
    det_center = {"box": [200, 100, 50, 80], "class_name": "rock", "confidence": 0.70}
    is_not_edge = detector.is_border_clipped(det_center["box"], img_width=640, img_height=480, margin=5)
    assert is_not_edge is False
