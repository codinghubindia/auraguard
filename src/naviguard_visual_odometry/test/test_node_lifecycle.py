"""Integration tests for VisualOdometryNode lifecycle, topics, and message handling."""

import cv2
import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image

from naviguard_visual_odometry.visual_odometry_node import VisualOdometryNode


@pytest.fixture
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_initialization(rclpy_context):
    """Verify VisualOdometryNode sets default parameters and attributes correctly."""
    node = VisualOdometryNode()

    assert node.get_parameter('image_topic').value == '/camera/image_raw'
    assert node.get_parameter('camera_info_topic').value == '/camera/camera_info'
    assert node.get_parameter('odom_topic').value == '/odom'
    assert node.get_parameter('debug_image_topic').value == '/visual_odometry/debug_image'
    assert node.get_parameter('telemetry_topic').value == '/visual_odometry/telemetry'
    assert node.get_parameter('max_features').value == 200
    assert node.get_parameter('fb_err_threshold').value == 1.0

    assert node.frame_idx == 0
    assert node.camera_info is None
    assert node.latest_odom_vx is None

    node.destroy_node()


def test_camera_info_callback(rclpy_context):
    """Verify camera intrinsics matrix is parsed and stored."""
    node = VisualOdometryNode()

    info = CameraInfo()
    info.header.frame_id = 'camera_link'
    info.width = 640
    info.height = 480
    info.k = [381.36, 0.0, 320.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]

    node.camera_info_callback(info)

    assert node.camera_info is not None
    assert node.fx == pytest.approx(381.36)
    assert node.fy == pytest.approx(381.36)
    assert node.cx == pytest.approx(320.0)
    assert node.cy == pytest.approx(240.0)

    node.destroy_node()


def test_odom_callback(rclpy_context):
    """Verify reference odometry is captured and stored."""
    node = VisualOdometryNode()

    odom = Odometry()
    odom.header.stamp.sec = 10
    odom.header.stamp.nanosec = 200000000
    odom.twist.twist.linear.x = 0.45
    odom.twist.twist.angular.z = -0.15

    node.odom_callback(odom)

    assert node.latest_odom_vx == pytest.approx(0.45)
    assert node.latest_odom_wz == pytest.approx(-0.15)
    assert node.latest_odom_time == pytest.approx(10.2)

    node.destroy_node()


def test_image_pipeline_and_telemetry_publication(rclpy_context):
    """Verify image_callback processes frame, updates frame index, and publishes outputs."""
    node = VisualOdometryNode()
    bridge = CvBridge()

    # Create synthetic textured image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(20, 600, 40):
        for j in range(20, 440, 40):
            cv2.rectangle(img, (i, j), (i + 20, j + 20), (200, 200, 200), -1)

    img_msg1 = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    img_msg1.header.frame_id = 'camera_link'
    img_msg1.header.stamp.sec = 1
    img_msg1.header.stamp.nanosec = 0

    published_debug = []
    published_telemetry = []

    orig_debug_pub = node.debug_image_pub.publish
    node.debug_image_pub.publish = lambda m: (published_debug.append(m), orig_debug_pub(m))

    orig_telemetry_pub = node.telemetry_pub.publish
    node.telemetry_pub.publish = lambda m: (published_telemetry.append(m), orig_telemetry_pub(m))

    # Frame 1
    node.image_callback(img_msg1)
    assert node.frame_idx == 1

    # Shift image slightly for Frame 2
    M = np.float32([[1, 0, 3], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (640, 480))
    img_msg2 = bridge.cv2_to_imgmsg(shifted, encoding='bgr8')
    img_msg2.header.frame_id = 'camera_link'
    img_msg2.header.stamp.sec = 1
    img_msg2.header.stamp.nanosec = 100000000  # 100ms later (10 Hz)

    node.image_callback(img_msg2)
    assert node.frame_idx == 2

    # Check published telemetry has geometric fields
    assert len(published_telemetry) >= 1
    last_diag = published_telemetry[-1]
    diag_keys = {kv.key: kv.value for kv in last_diag.status[0].values}
    assert 'geom_status' in diag_keys
    assert 'translation_scale_status' in diag_keys
    assert diag_keys['translation_scale_status'] == 'UNKNOWN'
    assert 'unit_tz' in diag_keys
    assert 'rot_yaw_deg' in diag_keys

    node.destroy_node()
