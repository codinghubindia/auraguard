import pytest
import rclpy
from naviguard_confidence.confidence_node import NaviguardConfidenceNode
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, OccupancyGrid


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_confidence_node_initialization_and_topics(rclpy_init):
    node = NaviguardConfidenceNode()

    # Verify Publishers
    assert node.decision_pub is not None
    assert node.diagnostics_pub is not None
    assert node.marker_pub is not None

    # Verify Subscriptions
    assert node.vo_sub is not None
    assert node.slam_sub is not None
    assert node.sync_sub is not None
    assert node.state_est_sub is not None
    assert node.imu_sub is not None
    assert node.wheel_sub is not None
    assert node.map_sub is not None

    # Verify default state
    assert node.latest_decision is not None

    node.destroy_node()


def test_confidence_node_callback_processing(rclpy_init):
    node = NaviguardConfidenceNode()

    # 1. Provide VO Telemetry
    vo_msg = DiagnosticArray()
    vo_msg.header.stamp.sec = 10
    vo_msg.header.stamp.nanosec = 0
    ds_vo = DiagnosticStatus()
    ds_vo.name = "naviguard_vo: telemetry"
    ds_vo.values = [
        KeyValue(key="geom_num_inliers", value="48"),
        KeyValue(key="geom_is_valid", value="true"),
        KeyValue(key="geom_inlier_ratio", value="0.75"),
    ]
    vo_msg.status.append(ds_vo)
    node.vo_telemetry_cb(vo_msg)
    assert node.vo_dict is not None
    assert node.vo_dict.get("geom_num_inliers") == "48"

    # 2. Provide IMU
    imu_msg = Imu()
    imu_msg.header.stamp.sec = 10
    imu_msg.linear_acceleration.z = 9.81
    node.imu_cb(imu_msg)
    assert node.imu_accel == (0.0, 0.0, 9.81)

    # 3. Provide Wheel Odom
    odom_msg = Odometry()
    odom_msg.header.stamp.sec = 10
    odom_msg.twist.twist.linear.x = 0.4
    node.wheel_cb(odom_msg)
    assert node.wheel_twist == (0.4, 0.0, 0.0)

    # 4. Trigger evaluation loop
    node.evaluation_loop()
    assert node.latest_decision.scores.overall > 0.0

    node.destroy_node()
