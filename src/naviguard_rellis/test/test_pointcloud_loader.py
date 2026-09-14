import os
import tempfile
import numpy as np
from naviguard_rellis.pointcloud_loader import PointCloudLoader, parse_pcd_header


def test_parse_pcd_header():
    lines = [
        "VERSION 0.7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 2",
        "HEIGHT 1",
        "POINTS 2",
        "DATA ascii"
    ]
    header, count = parse_pcd_header(lines)
    assert header['version'] == '0.7'
    assert header['fields'] == ['x', 'y', 'z']
    assert header['points'] == 2
    assert header['data'] == 'ascii'


def test_load_pcd_ascii():
    content = """VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 3
HEIGHT 1
POINTS 3
DATA ascii
1.0 2.0 3.0
4.0 5.0 6.0
7.0 8.0 9.0
"""
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        pts = PointCloudLoader.load_pcd(temp_path)
        assert pts is not None
        assert pts.shape == (3, 3)
        assert np.allclose(pts[0], [1.0, 2.0, 3.0])

        msg = PointCloudLoader.points_to_pointcloud2_msg(pts, frame_id='test_frame')
        assert msg.width == 3
        assert msg.height == 1
        assert msg.header.frame_id == 'test_frame'
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
