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


def test_panoramic_route_assistance_on_blocked_path(rclpy_init):
    node = NavigationNode()

    # 40x40 grid, res 0.1m, origin (0.0, 0.0)
    grid_msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 40
    meta.height = 40
    grid_msg.info = meta
    arr = np.zeros((40, 40), dtype=np.int8)
    # Wall from y=0.5 to y=1.5 at x=2.0 (blocking direct path from (1.0, 1.0) to (3.0, 1.0))
    arr[5:16, 20] = 100
    grid_msg.data = arr.flatten().tolist()
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

    # Provide panoramic analysis with clear escape waypoint at (1.0, 2.5)
    pano_msg = String()
    pano_msg.data = json.dumps({
        "panorama_available": True,
        "clear_route_found": True,
        "escape_point": [1.0, 2.5],
        "best_heading_deg": 60.0
    })
    node._panorama_analysis_callback(pano_msg)

    # Test path planning
    ok = node._compute_and_set_path(now_sec=10.0)
    assert ok is True
    assert len(node.raw_planned_path) > 0

    node.destroy_node()


def test_lidar_scan_integration_and_collision_guard(rclpy_init):
    """Verify LiDAR scan returns update forward obstacle proximity and trigger collision guard."""
    from sensor_msgs.msg import LaserScan
    node = NavigationNode()

    # Create scan with obstacle directly ahead at 0.40m
    scan = LaserScan()
    scan.header.frame_id = "lidar_link"
    scan.angle_min = -3.14159
    scan.angle_max = 3.14159
    n = 360
    scan.angle_increment = (2 * 3.14159) / n
    scan.range_min = 0.10
    scan.range_max = 25.0
    ranges = [10.0] * n
    # Direct forward beam at index 180 (angle 0)
    ranges[178] = 0.40
    ranges[179] = 0.40
    ranges[180] = 0.40
    ranges[181] = 0.40
    scan.ranges = ranges

    node._scan_callback(scan)
    assert node.min_forward_lidar_distance_m <= 0.44

    # Test fused obstacles with x_base, y_base, and radius
    fused_msg = String()
    fused_msg.data = json.dumps({
        "obstacles": [
            {
                "x_base": 1.2,
                "y_base": 0.0,
                "radius": 0.25,
                "confirmed": True,
                "traversable": False,
            }
        ]
    })
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = 0.0
    pose_msg.pose.position.y = 0.0
    node._pose_callback(pose_msg)

    # Initialize grid
    grid_msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = 0.1
    meta.width = 30
    meta.height = 30
    grid_msg.info = meta
    grid_msg.data = [0] * (30 * 30)
    node._map_callback(grid_msg)

    node._fused_obstacles_callback(fused_msg)
    assert len(node.occ_grid.persistent_blocked_regions) >= 1

    node.destroy_node()

