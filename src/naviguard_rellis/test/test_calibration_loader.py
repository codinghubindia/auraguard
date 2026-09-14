import os
import tempfile
import pytest
import numpy as np
from naviguard_rellis.calibration_loader import (
    parse_rellis_camera_info,
    create_camera_info_msg,
    parse_k_matrix_from_text
)


def test_parse_k_matrix_from_text():
    text = "[100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0]"
    k = parse_k_matrix_from_text(text)
    assert k is not None
    assert k.shape == (3, 3)
    assert k[0, 0] == 100.0
    assert k[0, 2] == 320.0


def test_parse_rellis_camera_info_synthetic():
    sample_content = """
    # Camera calibration
    width: 1280
    height: 720
    camera_matrix: [1050.0, 0.0, 640.0, 0.0, 1050.0, 360.0, 0.0, 0.0, 1.0]
    distortion_coefficients: [-0.1, 0.05, 0.001, 0.001, 0.0]
    """
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(sample_content)
        temp_path = f.name

    try:
        params = parse_rellis_camera_info(temp_path)
        assert params['width'] == 1280
        assert params['height'] == 720
        assert params['k'][0, 0] == 1050.0
        assert params['k'][0, 2] == 640.0
        assert len(params['d']) == 5
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_create_camera_info_msg():
    params = {
        'width': 640,
        'height': 480,
        'k': np.eye(3, dtype=np.float64),
        'd': [0.0] * 5
    }
    msg = create_camera_info_msg(params, frame_id='test_camera')
    assert msg.width == 640
    assert msg.height == 480
    assert msg.header.frame_id == 'test_camera'
    assert len(msg.k) == 9
    assert len(msg.d) == 5


def test_parse_official_rellis_camera_info():
    # Official RELLIS-3D camera_info.txt format: fx fy cx cy
    official_content = "2813.643275 2808.326079 969.285772 624.049972\n"
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(official_content)
        temp_path = f.name

    try:
        params = parse_rellis_camera_info(temp_path)
        assert pytest.approx(params['k'][0, 0], abs=1e-3) == 2813.643
        assert pytest.approx(params['k'][1, 1], abs=1e-3) == 2808.326
        assert pytest.approx(params['k'][0, 2], abs=1e-3) == 969.286
        assert pytest.approx(params['k'][1, 2], abs=1e-3) == 624.050
        assert params['p'][0, 0] == params['k'][0, 0]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

