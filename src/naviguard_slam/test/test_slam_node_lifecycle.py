import pytest
import rclpy
from naviguard_slam.slam_node import NaviguardSlamNode


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_slam_node_initialization_and_topics(rclpy_init):
    node = NaviguardSlamNode()

    assert node.mode == "mapping"
    assert node.map_frame == "map"
    assert node.odom_frame == "odom"

    # Publishers verified
    assert node.map_pub is not None
    assert node.traj_pub is not None
    assert node.pose_pub is not None
    assert node.marker_pub is not None
    assert node.diag_pub is not None

    # Services verified
    assert node.save_srv is not None
    assert node.load_srv is not None

    # Initial state
    assert node.tracking_state == "INITIALIZING"

    # Clean shutdown
    node.destroy_node()


def test_slam_node_transition_to_ok(rclpy_init):
    import numpy as np
    from sensor_msgs.msg import CameraInfo, Image
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge

    node = NaviguardSlamNode()
    bridge = CvBridge()

    # 1. Provide CameraInfo
    cam_info = CameraInfo()
    cam_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    node.camera_info_cb(cam_info)
    assert node.camera_matrix is not None

    # 2. Provide Odometry via fallback or primary
    odom_msg = Odometry()
    odom_msg.header.stamp.sec = 10
    odom_msg.header.stamp.nanosec = 0
    odom_msg.pose.pose.orientation.w = 1.0
    node.fallback_odom_cb(odom_msg)
    assert node.latest_odom_pose is not None

    # 3. Provide rich visual Image (random texture with high gradient)
    np.random.seed(42)
    img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    img_msg = bridge.cv2_to_imgmsg(img, encoding='mono8')
    img_msg.header.stamp.sec = 10
    img_msg.header.stamp.nanosec = 0

    node.image_cb(img_msg)

    # 4. Verify transition to OK
    assert node.tracking_state == "OK"
    assert node.failure_reason == ""
    assert len(node.keyframes) == 1

    node.destroy_node()


def test_map_publisher_qos_transient_local(rclpy_init):
    """Verify /slam/map publisher is TRANSIENT_LOCAL to match all subscribers."""
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    node = NaviguardSlamNode()
    assert node.map_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert node.map_pub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    node.destroy_node()


