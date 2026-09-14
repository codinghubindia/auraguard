"""Unit tests for Dashboard Vehicle Geometry, YOLO Diagnostics, and 11-Layer Architecture panels."""

import json
import os
import pytest
from naviguard_dashboard.state_cache import StateCache


def test_state_cache_vehicle_and_yolo_diagnostics():
    """Verify state cache holds default vehicle and YOLO diagnostics in snapshot."""
    cache = StateCache()
    snap = cache.get_snapshot()

    assert "vehicle_diagnostics" in snap
    vdiag = snap["vehicle_diagnostics"]
    assert "vehicle" in vdiag
    assert vdiag["vehicle"]["length_m"] == 0.56
    assert vdiag["vehicle"]["width_m"] == 0.48
    assert vdiag["vehicle"]["nominal_passage_m"] == 0.68
    assert vdiag["vehicle"]["tight_passage_limit_m"] == 0.58
    assert vdiag["status"] in ("SAFE", "TIGHT", "BLOCKED", "UNKNOWN")

    assert "yolo_diagnostics" in snap
    ydiag = snap["yolo_diagnostics"]
    assert "model" in ydiag
    assert "device" in ydiag
    assert ydiag["device"] in ("CPU", "CUDA")
    assert "detections" in ydiag
    assert "fps" in ydiag


def test_state_cache_yolo_stream_buffer():
    """Verify state cache maintains a bounded stream buffer for YOLO overlay frames."""
    cache = StateCache()
    assert "yolo" in cache.streams

    dummy_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb"
    cache.set_jpeg_frame("yolo", dummy_jpeg, timestamp=100.0, encode_latency_ms=12.5, source_fps=10.0)

    frame_bytes, ts, age_ms, status = cache.get_stream_frame("yolo")
    assert frame_bytes == dummy_jpeg
    assert ts == 100.0
    assert status in ("LIVE", "DEGRADED")


def test_state_cache_updates():
    """Verify state cache updates properly with external ROS telemetry."""
    cache = StateCache()
    cache.set_vehicle_diagnostics({
        "status": "TIGHT",
        "available_clear_width_m": 0.62,
        "clearance_margin_m": 0.14,
        "can_fit": True,
    })
    snap = cache.get_snapshot()
    assert snap["vehicle_diagnostics"]["status"] == "TIGHT"
    assert snap["vehicle_diagnostics"]["available_clear_width_m"] == 0.62
    assert snap["vehicle_diagnostics"]["clearance_margin_m"] == 0.14

    cache.set_yolo_diagnostics({
        "model": "yolov8n.onnx",
        "device": "CPU",
        "fps": 11.2,
        "latency_ms": 35.4,
        "num_detections": 2,
        "detections": [
            {"class_name": "rock", "confidence": 0.88, "traversable": False},
            {"class_name": "trail", "confidence": 0.95, "traversable": True},
        ]
    })
    snap2 = cache.get_snapshot()
    assert snap2["yolo_diagnostics"]["fps"] == 11.2
    assert snap2["yolo_diagnostics"]["num_detections"] == 2
    assert len(snap2["yolo_diagnostics"]["detections"]) == 2


def test_index_html_panel_and_layer_elements():
    """Verify index.html contains DOM structures for Vehicle Geometry, YOLO, and 11 Layers."""
    html_path = os.path.join(
        os.path.dirname(__file__), "..", "naviguard_dashboard", "static", "index.html"
    )
    assert os.path.exists(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Vehicle Geometry Panel elements
    assert "VEHICLE GEOMETRY &amp; PASSAGE CLEARANCE" in content or "VEHICLE GEOMETRY & PASSAGE CLEARANCE" in content
    assert "0.56 m" in content
    assert "0.48 m" in content
    assert "0.68 m" in content
    assert "0.58 m" in content
    assert "badgePassageStatus" in content

    # 2. YOLO Panel elements
    assert "YOLOv8 OUTDOOR PERCEPTION &amp; DETECTIONS" in content or "YOLOv8 OUTDOOR PERCEPTION & DETECTIONS" in content
    assert "badgeYoloStatus" in content
    assert "yoloDevice" in content
    assert "yoloDetectionsList" in content
    assert "camYolo" in content

    # 3. 11-Layer Spatial Model
    assert "11-LAYER SPATIAL ARCHITECTURE SPECIFICATION" in content
    assert "L1 — RELLIS Basemap" in content
    assert "L2 — SLAM Occupancy" in content
    assert "L4 — UGV Footprint" in content
    assert "L10 — Recovery Checkpoints" in content
    assert "L11 — Telemetry HUD" in content
