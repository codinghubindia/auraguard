import json
import pytest
import rclpy
from naviguard_recovery.recovery_node import NaviguardRecoveryNode
from naviguard_recovery.recovery_state_machine import RecoveryState
from std_msgs.msg import String
from std_srvs.srv import Trigger


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_recovery_node_initialization_and_topics(rclpy_init):
    node = NaviguardRecoveryNode()

    # Verify Publishers
    assert node.cmd_vel_pub is not None
    assert node.state_pub is not None
    assert node.diag_pub is not None
    assert node.marker_pub is not None

    # Verify Subscriptions
    assert node.decision_sub is not None
    assert node.pose_sub is not None
    assert node.slam_diag_sub is not None
    assert node.odom_sub is not None
    assert node.map_sub is not None

    # Verify Services
    assert node.trigger_srv is not None
    assert node.reset_srv is not None

    assert node.fsm.state == RecoveryState.NORMAL

    node.destroy_node()


def test_recovery_node_manual_trigger_service(rclpy_init):
    node = NaviguardRecoveryNode()

    req = Trigger.Request()
    resp = Trigger.Response()

    resp = node.manual_trigger_callback(req, resp)
    assert resp.success is True
    assert node.fsm.state == RecoveryState.SAFE_STOP

    # Reset service
    resp_reset = Trigger.Response()
    resp_reset = node.reset_budget_callback(req, resp_reset)
    assert resp_reset.success is True
    assert node.fsm.state == RecoveryState.NORMAL

    node.destroy_node()


def test_recovery_node_decision_callback(rclpy_init):
    node = NaviguardRecoveryNode()

    dec_msg = String()
    dec_msg.data = json.dumps({
        "state": "RECOVER",
        "primary_reason": "SLAM_TRACKING_LOST",
        "scores": {
            "overall": 0.25,
            "localization": 0.10,
            "visual": 0.30,
        },
    })

    node.decision_callback(dec_msg)
    assert node.phase7_state_str == "RECOVER"
    assert node.phase7_primary_reason == "SLAM_TRACKING_LOST"
    assert node.phase7_overall_conf == 0.25

    # Run one timer step to process input
    node.timer_callback()
    assert node.fsm.state == RecoveryState.SAFE_STOP

    node.destroy_node()
