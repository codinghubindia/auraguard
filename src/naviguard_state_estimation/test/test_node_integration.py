import pytest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray

from naviguard_state_estimation.state_estimation_node import StateEstimationNode


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_node_lifecycle(rclpy_init):
    node = StateEstimationNode()
    assert node.publish_rate_hz == 20.0
    assert node.odom_pub is not None
    assert node.diag_pub is not None

    # Verify initial state is uninitialized before first wheel odom
    assert node.estimator.is_initialized is False

    # Publish dummy wheel odom to initialize
    odom_msg = Odometry()
    odom_msg.header.stamp.sec = 1
    odom_msg.header.stamp.nanosec = 0
    odom_msg.header.frame_id = "odom"
    odom_msg.child_frame_id = "base_link"
    odom_msg.pose.pose.position.x = 1.5
    odom_msg.pose.pose.position.y = 2.5
    odom_msg.pose.pose.orientation.w = 1.0
    odom_msg.twist.twist.linear.x = 0.2
    odom_msg.twist.twist.angular.z = 0.05

    node.wheel_odom_callback(odom_msg)
    assert node.estimator.is_initialized is True
    assert node.estimator.state.x == 1.5
    assert node.estimator.state.y == 2.5

    # Run timer callback once
    node.timer_callback()

    # Clean shutdown
    node.destroy_node()
