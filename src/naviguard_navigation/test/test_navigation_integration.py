import json
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np

from naviguard_navigation.navigation_node import NavigationNode
from naviguard_navigation.mission_manager import MissionState


@pytest.fixture
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_navigation_node_lifecycle_and_topics(rclpy_init):
    node = NavigationNode()

    # Publishers
    assert node.cmd_vel_pub is not None
    assert node.state_pub is not None
    assert node.path_pub is not None
    assert node.waypoints_pub is not None
    assert node.marker_pub is not None

    # Subscribers
    assert node.map_sub is not None
    assert node.pose_sub is not None
    assert node.odom_sub is not None
    assert node.goal_sub is not None
    assert node.set_goal_sub is not None
    assert node.decision_sub is not None
    assert node.recovery_sub is not None

    # Initial state
    assert node.mission_mgr.state == MissionState.IDLE

    node.destroy_node()


def test_navigation_goal_reception_and_cancel_service(rclpy_init):
    node = NavigationNode()

    # Send goal
    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "map"
    goal_msg.pose.position.x = 2.0
    goal_msg.pose.position.y = 1.0
    node._goal_callback(goal_msg)

    assert node.goal_mgr.has_goal() is True
    assert node.mission_mgr.state == MissionState.GOAL_SET

    # Test cancel service
    req = Trigger.Request()
    resp = Trigger.Response()
    resp = node._cancel_goal_callback(req, resp)

    assert resp.success is True
    assert node.goal_mgr.has_goal() is False
    assert node.mission_mgr.state == MissionState.IDLE

    node.destroy_node()


def test_command_ownership_and_recovery_yield(rclpy_init):
    node = NavigationNode()

    # Set map and pose
    grid_msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 50
    meta.height = 50
    grid_msg.info = meta
    grid_msg.data = np.zeros((50, 50), dtype=np.int8).flatten().tolist()
    node._map_callback(grid_msg)

    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = 1.0
    pose_msg.pose.position.y = 1.0
    node._pose_callback(pose_msg)

    goal_msg = PoseStamped()
    goal_msg.header.frame_id = "map"
    goal_msg.pose.position.x = 3.0
    goal_msg.pose.position.y = 1.0
    node._goal_callback(goal_msg)

    # Trigger control loop to plan and enter active navigation
    node._control_loop()  # GOAL_SET -> PLANNING
    node._control_loop()  # PLANNING -> NAVIGATING (path computed)
    node._control_loop()  # NAVIGATING (cmd_vel active)
    assert node.mission_mgr.state == MissionState.NAVIGATING
    assert node.cmd_ownership == "NAVIGATING_ACTIVE"

    # Inject recovery trigger from Phase 8
    rec_msg = String()
    rec_msg.data = json.dumps({"recovery_state": "SAFE_STOP"})
    node._recovery_callback(rec_msg)

    # Next control cycle must immediately yield to recovery
    node._control_loop()
    assert node.mission_mgr.state == MissionState.RECOVERY_WAIT
    assert node.cmd_ownership == "YIELDED_TO_RECOVERY"

    # Recovery finishes and transitions back to NORMAL
    rec_msg.data = json.dumps({"recovery_state": "NORMAL"})
    node._recovery_callback(rec_msg)

    # Control cycle resumes navigation by replanning
    node._control_loop()
    assert node.mission_mgr.state == MissionState.REPLANNING

    node.destroy_node()
