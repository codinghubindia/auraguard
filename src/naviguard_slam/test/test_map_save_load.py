import os
import tempfile
import numpy as np
import pytest

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.landmark_database import LandmarkDatabase
from naviguard_slam.occupancy_grid_mapper import OccupancyGridMapper
from naviguard_slam.slam_node import NaviguardSlamNode
import rclpy


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_map_save_and_load_cycle(rclpy_init):
    node = NaviguardSlamNode()

    # Populate node with test keyframes and landmarks
    kpts = np.array([[100.0, 100.0]], dtype=np.float32)
    descs = np.full((1, 32), 128, dtype=np.uint8)

    kf0 = Keyframe(0, 1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kpts, descs)
    kf1 = Keyframe(1, 2.0, (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kpts, descs)
    node.keyframes = [kf0, kf1]

    node.landmark_db.add_landmark((1.5, 0.5, 0.2), descs[0], keyframe_id=0)
    node.map_to_odom_offset = (0.2, -0.1, 0.05)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file = os.path.join(tmpdir, "test_map.json")
        success, msg = node.save_map_to_disk(save_file)
        assert success is True
        assert os.path.exists(save_file)
        assert os.path.exists(os.path.join(tmpdir, "test_map.yaml"))
        assert os.path.exists(os.path.join(tmpdir, "test_map.pgm"))

        # Create fresh node and load map
        node_loader = NaviguardSlamNode()
        load_success, load_msg = node_loader.load_map_from_disk(save_file)
        assert load_success is True
        assert len(node_loader.keyframes) == 2
        assert len(node_loader.landmark_db) == 1
        assert node_loader.mode == "localization"
        assert pytest.approx(node_loader.map_to_odom_offset[0], abs=1e-4) == 0.2

        node_loader.destroy_node()

    node.destroy_node()
