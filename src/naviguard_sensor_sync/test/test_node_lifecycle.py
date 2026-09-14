"""Integration tests for SensorSyncNode lifecycle and diagnostic generation."""

import pytest
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu

from naviguard_sensor_sync.sensor_sync_node import SensorSyncNode


@pytest.fixture
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_initialization(rclpy_context):
    """Verify SensorSyncNode initializes with default parameters."""
    node = SensorSyncNode()

    assert node.get_parameter('camera_image_topic').value == '/camera/image_raw'
    assert node.get_parameter('camera_info_topic').value == '/camera/camera_info'
    assert node.get_parameter('imu_topic').value == '/imu'
    assert node.get_parameter('odom_topic').value == '/odom'
    assert node.get_parameter('sync_tolerance_ms').value == 20.0
    assert node.get_parameter('expected_camera_rate').value == 30.0

    node.destroy_node()


def test_sensor_callbacks_and_diagnostics(rclpy_context):
    """Verify sensor callbacks update internal monitors and diagnostic publisher fires."""
    node = SensorSyncNode()

    published_diags = []
    orig_pub = node.diag_pub.publish
    node.diag_pub.publish = lambda m: (published_diags.append(m), orig_pub(m))

    # Send Camera Image
    img = Image()
    img.header.stamp.sec = 10
    img.header.stamp.nanosec = 0
    img.header.frame_id = "camera_link"
    img.width = 640
    img.height = 480
    node._cam_callback(img)

    # Send Camera Info
    info = CameraInfo()
    info.header.stamp.sec = 10
    info.header.stamp.nanosec = 0
    info.header.frame_id = "camera_optical_link"
    info.width = 640
    info.height = 480
    info.k = [381.36, 0.0, 320.0, 0.0, 381.36, 240.0, 0.0, 0.0, 1.0]
    node._info_callback(info)

    # Send IMU
    imu = Imu()
    imu.header.stamp.sec = 10
    imu.header.stamp.nanosec = 2000000  # 2ms offset
    imu.header.frame_id = "imu_link"
    imu.linear_acceleration.z = 9.81
    node._imu_callback(imu)

    # Send Odometry
    odom = Odometry()
    odom.header.stamp.sec = 10
    odom.header.stamp.nanosec = 5000000  # 5ms offset
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"
    node._odom_callback(odom)

    # Trigger diagnostics callback directly
    node._diagnostics_timer_callback()

    assert len(published_diags) == 1
    diag = published_diags[0]
    assert len(diag.status) >= 2

    # Check overall system status block
    overall_ds = diag.status[0]
    assert overall_ds.name == "naviguard_sensor_sync: Overall System"
    vals = {kv.key: kv.value for kv in overall_ds.values}
    assert 'phase_5a_status' in vals
    assert 'camera_rate_hz' in vals
    assert 'imu_rate_hz' in vals
    assert 'odom_rate_hz' in vals

    node.destroy_node()
