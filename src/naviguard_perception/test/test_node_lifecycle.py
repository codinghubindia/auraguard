"""Integration tests for NAVIGUARD perception node lifecycle and conversions."""

import cv_bridge
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import CameraInfo, Image

from naviguard_perception.perception_node import NaviguardPerceptionNode


@pytest.fixture
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_instantiation_and_parameters(rclpy_context):
    """Verify node initializes with correct parameters and topics."""
    node = NaviguardPerceptionNode()
    assert node.get_parameter('image_topic').value == '/camera/image_raw'
    assert node.get_parameter('debug_image_topic').value == '/perception/debug_image'
    assert node.camera_info_received is False
    assert node.total_received == 0
    node.destroy_node()


def test_camera_info_handling(rclpy_context):
    """Verify camera_info callback extracts and stores intrinsic calibration parameters."""
    node = NaviguardPerceptionNode()
    cam_info = CameraInfo()
    cam_info.header.frame_id = 'camera_link'
    cam_info.width = 640
    cam_info.height = 480
    # K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    cam_info.k = [381.36, 0.0, 320.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]

    node.camera_info_callback(cam_info)
    assert node.camera_info_received is True
    assert node.cam_fx == 381.36
    assert node.cam_fy == 381.36
    assert node.cam_cx == 320.0
    assert node.cam_cy == 240.0
    node.destroy_node()


def test_image_callback_and_publish(rclpy_context):
    """Verify image_callback converts, processes, and publishes debug image."""
    node = NaviguardPerceptionNode()
    bridge = cv_bridge.CvBridge()

    dummy_rgb = np.full((480, 640, 3), 100, dtype=np.uint8)
    msg = bridge.cv2_to_imgmsg(dummy_rgb, encoding='rgb8')
    msg.header.frame_id = 'camera_link'
    msg.header.stamp.sec = 42
    msg.header.stamp.nanosec = 123456789

    # Track published message
    published_msgs = []
    sub = node.create_subscription(
        Image,
        '/perception/debug_image',
        lambda m: published_msgs.append(m),
        10
    )

    node.image_callback(msg)

    assert node.total_received == 1
    assert node.total_processed == 1
    assert node.total_dropped == 0

    node.destroy_node()


def test_malformed_image_resilience(rclpy_context):
    """Verify node does not crash when receiving an empty or invalid image."""
    node = NaviguardPerceptionNode()

    # Empty image
    empty_msg = Image()
    empty_msg.header.frame_id = 'camera_link'
    empty_msg.data = b''
    empty_msg.encoding = 'rgb8'

    # Should not raise exception
    node.image_callback(empty_msg)
    assert node.total_received == 1
    assert node.total_dropped == 1
    assert node.total_processed == 0

    node.destroy_node()
