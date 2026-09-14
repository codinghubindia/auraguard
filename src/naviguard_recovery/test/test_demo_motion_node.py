import json
import pytest
import rclpy
from naviguard_recovery.demo_motion_node import NaviguardDemoMotionNode
from std_msgs.msg import String


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_demo_motion_initialization_and_topics(rclpy_init):
    node = NaviguardDemoMotionNode()

    assert node.cmd_vel_pub is not None
    assert node.status_pub is not None
    assert node.recovery_sub is not None
    assert node.is_active is True
    assert node.recovery_state == 'NORMAL'

    node.destroy_node()


def test_demo_motion_yields_on_non_normal_state(rclpy_init):
    node = NaviguardDemoMotionNode()

    # Simulate SAFE_STOP message from recovery
    msg = String()
    msg.data = json.dumps({'recovery_state': 'SAFE_STOP'})
    node._recovery_state_callback(msg)

    assert node.recovery_state == 'SAFE_STOP'
    assert node.is_active is False

    # Simulate control loop iteration when yielding
    node._control_loop()
    assert node.current_vx == 0.0
    assert node.current_wz == 0.0

    # Simulate return to NORMAL
    msg.data = json.dumps({'recovery_state': 'NORMAL'})
    node._recovery_state_callback(msg)
    assert node.recovery_state == 'NORMAL'
    assert node.is_active is True

    node.destroy_node()
