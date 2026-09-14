import os
import math
import tempfile
import numpy as np
from naviguard_rellis.pose_loader import (
    rot_matrix_to_quaternion,
    parse_kitti_pose_line,
    PoseLoader
)


def test_rot_matrix_to_quaternion_identity():
    R = np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = rot_matrix_to_quaternion(R)
    assert math.isclose(qw, 1.0, abs_tol=1e-5)
    assert math.isclose(qx, 0.0, abs_tol=1e-5)
    assert math.isclose(qy, 0.0, abs_tol=1e-5)
    assert math.isclose(qz, 0.0, abs_tol=1e-5)


def test_parse_kitti_pose_line():
    line = "1 0 0 10.5 0 1 0 2.3 0 0 1 -0.4"
    res = parse_kitti_pose_line(line)
    assert res is not None
    R, t = res
    assert np.allclose(R, np.eye(3))
    assert math.isclose(t[0], 10.5)
    assert math.isclose(t[1], 2.3)
    assert math.isclose(t[2], -0.4)


def test_pose_loader_file():
    content = """
    1 0 0 0 0 1 0 0 0 0 1 0
    1 0 0 1.0 0 1 0 0.1 0 0 1 0
    1 0 0 2.0 0 1 0 0.2 0 0 1 0
    """
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(content.strip())
        temp_path = f.name

    try:
        loader = PoseLoader(temp_path)
        assert len(loader) == 3
        p1 = loader.get_pose(1)
        assert p1 is not None
        assert math.isclose(p1['position'][0], 1.0)
        assert math.isclose(p1['position'][1], 0.1)

        msg = loader.to_pose_msg(2)
        assert msg is not None
        assert math.isclose(msg.position.x, 2.0)

        path = loader.to_path_msg()
        assert len(path.poses) == 3
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
