"""Master navigation and mission control node for NAVIGUARD autonomous outdoor UGV."""

import json
import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, PoseArray, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import String, Header
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from diagnostic_msgs.msg import DiagnosticArray

from naviguard_navigation.mission_manager import MissionManager, MissionState, FailureCode
from naviguard_navigation.goal_manager import GoalManager, NavigationGoal
from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.global_planner import GlobalPlannerAStar
from naviguard_navigation.path_smoother import PathSmoother
from naviguard_navigation.waypoint_generator import WaypointGenerator, Waypoint
from naviguard_navigation.path_follower import PathFollower
from naviguard_navigation.goal_checker import GoalChecker
from naviguard_navigation.replanner import Replanner
from naviguard_navigation.navigation_diagnostics import NavigationDiagnostics
from naviguard_navigation.vehicle_geometry import VehicleGeometry


class NavigationNode(Node):
    """Integrates mission management, multi-cost A* global planning, and adaptive closed-loop path following."""

    def __init__(self, node_name: str = 'navigation_node') -> None:
        super().__init__(node_name)

        # 1. Declare Parameters
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('inflation_radius_m', 0.34)
        self.declare_parameter('proximity_radius_m', 1.0)
        self.declare_parameter('safe_clearance_m', 1.20)
        self.declare_parameter('allow_unknown', True)
        self.declare_parameter('unknown_cost_penalty', 8.0)
        self.declare_parameter('max_linear_velocity', 0.25)
        self.declare_parameter('min_linear_velocity', 0.05)
        self.declare_parameter('max_angular_velocity', 0.35)
        self.declare_parameter('lookahead_distance_m', 0.40)
        self.declare_parameter('xy_goal_tolerance_m', 0.25)
        self.declare_parameter('yaw_goal_tolerance_rad', 0.30)
        self.declare_parameter('required_dwell_sec', 1.0)
        self.declare_parameter('max_replan_retries', 5)
        self.declare_parameter('min_replan_interval_sec', 1.0)
        self.declare_parameter('max_path_deviation_m', 0.60)
        self.declare_parameter('map_frame', 'map')

        # Extract parameter values
        control_rate = float(self.get_parameter('control_rate_hz').value)
        inflation_radius = float(self.get_parameter('inflation_radius_m').value)
        proximity_radius = float(self.get_parameter('proximity_radius_m').value)
        safe_clearance = float(self.get_parameter('safe_clearance_m').value)
        allow_unknown = bool(self.get_parameter('allow_unknown').value)
        unknown_penalty = float(self.get_parameter('unknown_cost_penalty').value)
        max_v = float(self.get_parameter('max_linear_velocity').value)
        min_v = float(self.get_parameter('min_linear_velocity').value)
        max_w = float(self.get_parameter('max_angular_velocity').value)
        lookahead = float(self.get_parameter('lookahead_distance_m').value)
        xy_tol = float(self.get_parameter('xy_goal_tolerance_m').value)
        yaw_tol = float(self.get_parameter('yaw_goal_tolerance_rad').value)
        dwell_sec = float(self.get_parameter('required_dwell_sec').value)
        max_retries = int(self.get_parameter('max_replan_retries').value)
        replan_interval = float(self.get_parameter('min_replan_interval_sec').value)
        max_deviation = float(self.get_parameter('max_path_deviation_m').value)
        self.map_frame = str(self.get_parameter('map_frame').value)

        # 2. Subsystem Components
        self.mission_mgr = MissionManager(max_replan_retries=max_retries)
        self.goal_mgr = GoalManager(default_frame=self.map_frame)
        self.occ_grid = NavigationOccupancyGrid(
            inflation_radius_m=inflation_radius,
            proximity_radius_m=proximity_radius,
            safe_clearance_m=safe_clearance,
            allow_unknown=allow_unknown,
            unknown_cost_penalty=unknown_penalty,
        )
        self.planner = GlobalPlannerAStar(
            heuristic_weight=1.0,
            turn_penalty_weight=0.5,
            clearance_weight=0.15,
            terrain_weight=0.15,
            slope_weight=0.20,
        )
        self.smoother = PathSmoother()
        self.wp_generator = WaypointGenerator(target_spacing_m=0.35)
        self.follower = PathFollower(
            lookahead_distance_m=lookahead,
            max_linear_velocity=max_v,
            min_linear_velocity=min_v,
            max_angular_velocity=max_w,
        )
        self.goal_checker = GoalChecker(
            xy_tolerance_m=xy_tol,
            yaw_tolerance_rad=yaw_tol,
            required_dwell_sec=dwell_sec,
        )
        self.replanner = Replanner(
            min_replan_interval_sec=replan_interval,
            max_path_deviation_m=max_deviation,
        )

        # 3. State & Sensor Variables
        self.robot_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)
        self.raw_planned_path: List[Tuple[float, float]] = []
        self.waypoints: List[Waypoint] = []
        self.current_lookahead: Optional[Waypoint] = None
        self.cross_track_error: float = 0.0

        # Integration states
        self.confidence_decision: str = "CONTINUE"
        self.confidence_overall: float = 1.0
        self.confidence_visual: float = 1.0
        self.confidence_localization: float = 1.0
        self.recovery_state: str = "NORMAL"
        self.recovery_attempts_count: int = 0
        self.cmd_ownership: str = "YIELDED"
        self.last_cmd_vx: float = 0.0
        self.last_cmd_wz: float = 0.0
        # Guard to prevent re-triggering 360° recovery every tick while waiting for response
        self._recovery_pending: bool = False

        # Environmental state explanations
        self.nav_reason: str = "Nominal path following"
        self.current_clearance_m: float = 1.0
        self.current_terrain_cost: float = 0.0
        self.current_speed_scale: float = 1.0
        self.last_passage_eval: dict = {
            "passage_status": "UNKNOWN",
            "available_width_m": 0.0,
            "required_width_m": VehicleGeometry.NOMINAL_PASSAGE_WIDTH_M,
            "clearance_margin_m": 0.0,
            "can_fit": True,
            "can_turn": True,
            "turning_diameter_available_m": 0.0,
        }

        # 4. QoS Profiles
        qos_map = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # 5. Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/navigation/state', 10)
        self.path_pub = self.create_publisher(Path, '/navigation/path', 10)
        self.waypoints_pub = self.create_publisher(PoseArray, '/navigation/waypoints', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/navigation/markers', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/navigation/diagnostics', 10)
        self.replan_diag_pub = self.create_publisher(String, '/navigation/replan_diagnostics', 10)
        self.vehicle_diag_pub = self.create_publisher(String, '/navigation/vehicle_diagnostics', 10)

        # 6. Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/slam/map', self._map_callback, qos_map
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/slam/pose', self._pose_callback, qos_reliable
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/state_estimation/odom', self._odom_callback, qos_reliable
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self._goal_callback, qos_reliable
        )
        self.set_goal_sub = self.create_subscription(
            PoseStamped, '/navigation/set_goal', self._goal_callback, qos_reliable
        )
        self.decision_sub = self.create_subscription(
            String, '/naviguard/decision', self._decision_callback, qos_reliable
        )
        self.recovery_sub = self.create_subscription(
            String, '/recovery/state', self._recovery_callback, qos_reliable
        )
        self.fused_obstacles_sub = self.create_subscription(
            String, '/perception/fused_obstacles', self._fused_obstacles_callback, qos_reliable
        )
        self.perception_conf_sub = self.create_subscription(
            String, '/perception/confidence', self._perception_confidence_callback, qos_reliable
        )

        # 7. Services
        self.cancel_srv = self.create_service(
            Trigger, '/navigation/cancel_goal', self._cancel_goal_callback
        )

        # 8. Recovery trigger client — calls recovery_node to start 360° lookaround scan
        self.fault_cli = self.create_client(Trigger, '/recovery/trigger_manual_recovery')

        # 9. Control Timer
        timer_period = 1.0 / max(1.0, control_rate)
        self.timer = self.create_timer(timer_period, self._control_loop)

        self.get_logger().info(
            f"Navigation Node initialized: rate={control_rate}Hz, v_max={max_v}m/s, w_max={max_w}rad/s, R_inflate={inflation_radius}m"
        )

    def _map_callback(self, msg: OccupancyGrid) -> None:
        self.occ_grid.update_from_msg(msg)

    def _pose_callback(self, msg) -> None:
        if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
        elif hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
            p = msg.pose.position
            q = msg.pose.orientation
        else:
            return
        px = float(p.x)
        py = float(p.y)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose = (px, py, yaw)
        self.mission_mgr.update_odometry_distance(px, py)

    def _odom_callback(self, msg: Odometry) -> None:
        # Fallback pose if SLAM pose hasn't arrived yet
        if self.robot_pose is None:
            px = float(msg.pose.pose.position.x)
            py = float(msg.pose.pose.position.y)
            q = msg.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.robot_pose = (px, py, yaw)
            self.mission_mgr.update_odometry_distance(px, py)

    def _goal_callback(self, msg: PoseStamped) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        goal = self.goal_mgr.set_goal_from_msg(msg, now_sec)
        if goal is not None:
            self.get_logger().info(f"New navigation goal received: ({goal.x:.2f}, {goal.y:.2f})")
            self.mission_mgr.start_mission(now_sec)
            self._recovery_pending = False  # Reset for new mission
            if self.mission_mgr.state in (MissionState.IDLE, MissionState.GOAL_REACHED, MissionState.MISSION_FAILED):
                self.mission_mgr.transition_to(MissionState.GOAL_SET, now_sec)
            elif self.mission_mgr.state == MissionState.NAVIGATING:
                self.mission_mgr.transition_to(MissionState.PLANNING, now_sec, "NEW_GOAL_PREEMPTION")

    def _cancel_goal_callback(self, req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.goal_mgr.clear_goal()
        self.occ_grid.clear_blocked_regions()
        self.mission_mgr.reset(now_sec)
        self._stop_robot()
        resp.success = True
        resp.message = "Navigation goal canceled. Robot returned to IDLE."
        self.get_logger().info("Mission canceled by operator.")
        return resp

    def _fused_obstacles_callback(self, msg: String) -> None:
        """Integrate perception fused obstacles into navigation occupancy grid."""
        try:
            data = json.loads(msg.data)
            obstacles = data.get("obstacles", [])
            if self.robot_pose is None or not self.occ_grid.is_initialized:
                return
            rx, ry, ryaw = self.robot_pose
            cos_yaw = math.cos(ryaw)
            sin_yaw = math.sin(ryaw)
            for obs in obstacles:
                # Insert non-traversable confirmed obstacles or emergency obstacles
                if not obs.get("traversable", False) and (obs.get("confirmed", False) or obs.get("emergency", False)):
                    xb = float(obs.get("x_base", 0.0))
                    yb = float(obs.get("y_base", 0.0))
                    radius = float(obs.get("radius", 0.25))
                    # Transform from base_link to map
                    xm = rx + xb * cos_yaw - yb * sin_yaw
                    ym = ry + xb * sin_yaw + yb * cos_yaw
                    self.occ_grid.mark_blocked_region(xm, ym, radius_m=max(0.20, radius))
        except Exception as e:
            self.get_logger().warn(f"Failed parsing fused obstacles: {e}")

    def _perception_confidence_callback(self, msg: String) -> None:
        """Update confidence metrics from perception confidence pipeline."""
        try:
            data = json.loads(msg.data)
            vis_conf = data.get("visual_confidence") or data.get("confidence")
            if vis_conf is not None:
                self.confidence_visual = float(vis_conf)
        except Exception:
            pass

    def _decision_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.confidence_decision = data.get("state", data.get("decision", "CONTINUE"))
            scores = data.get("scores", {})
            self.confidence_overall = float(scores.get("overall", 1.0))
            self.confidence_visual = float(scores.get("visual", 1.0))
            self.confidence_localization = float(scores.get("localization", 1.0))
        except Exception:
            self.confidence_decision = msg.data.strip()

    def _recovery_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.recovery_state = data.get("recovery_state", "NORMAL")
            self.recovery_attempts_count = int(data.get("attempts", data.get("budget", {}).get("attempt_count", 0)))
        except Exception:
            self.recovery_state = msg.data.strip()

    def _build_context(self, now_sec: float) -> dict:
        """Construct rich telemetry context for failure explanation."""
        goal = self.goal_mgr.get_goal()
        rx, ry, ryaw = self.robot_pose if self.robot_pose else (0.0, 0.0, 0.0)
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry) if goal else 0.0
        return {
            "recovery_state": self.recovery_state,
            "confidence": self.confidence_overall,
            "visual_confidence": self.confidence_visual,
            "localization_confidence": self.confidence_localization,
            "goal": goal.to_dict() if goal else None,
            "robot_pose": {"x": round(rx, 3), "y": round(ry, 3), "yaw": round(ryaw, 3)},
            "distance_to_goal": round(dist_to_goal, 3),
            "last_valid_pose": {"x": round(rx, 3), "y": round(ry, 3), "yaw": round(ryaw, 3)},
            "recovery_attempts": self.recovery_attempts_count,
            "recovery_max_attempts": 3,
            "blocked_regions": len(self.occ_grid.persistent_blocked_regions),
            "path_status": "BLOCKED" if len(self.occ_grid.persistent_blocked_regions) > 0 else "NO_SAFE_PATH",
            "passage": self.last_passage_eval,
            "sensor_status": {
                "SLAM": "ONLINE" if self.robot_pose is not None else "OFFLINE",
                "Map": "ONLINE" if self.occ_grid.is_initialized else "OFFLINE",
                "Confidence": "ONLINE" if self.confidence_overall > 0.3 else "DEGRADED",
            },
        }

    def _control_loop(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Evaluate vehicle footprint passage and clearance at current pose
        if self.robot_pose is not None and self.occ_grid.is_initialized:
            rx, ry, ryaw = self.robot_pose
            _, _, diag = self.occ_grid.evaluate_passage_at(rx, ry, ryaw)
            self.last_passage_eval = diag

        # Handle Mission State Machine
        if self.mission_mgr.state == MissionState.IDLE:
            self.cmd_ownership = "IDLE_STOP"
            self.nav_reason = "Awaiting operator goal"
            if self.goal_mgr.has_goal():
                self.mission_mgr.transition_to(MissionState.GOAL_SET, now_sec)

        elif self.mission_mgr.state == MissionState.GOAL_SET:
            self.cmd_ownership = "PLANNING_WAIT"
            self.nav_reason = "Initializing navigation map and pose"
            if self.robot_pose is not None and self.occ_grid.is_initialized:
                self.mission_mgr.transition_to(MissionState.PLANNING, now_sec)

        elif self.mission_mgr.state == MissionState.PLANNING:
            self.cmd_ownership = "PLANNING"
            self.nav_reason = "Computing optimal multi-cost path"
            if self.robot_pose is None or not self.occ_grid.is_initialized:
                self.cmd_ownership = "PLANNING_WAIT"
                self.nav_reason = "Waiting for SLAM occupancy map and robot pose before planning"
                self._publish_telemetry(now_sec)
                return

            # Only attempt to plan if we are not already waiting for a recovery scan
            if self._recovery_pending:
                self.cmd_ownership = "YIELDED_TO_RECOVERY"
                self.nav_reason = "Waiting for 360° recovery scan to complete before retrying plan"
                self._publish_telemetry(now_sec)
                return

            success = self._compute_and_set_path(now_sec)
            if success:
                self._recovery_pending = False
                self.mission_mgr.transition_to(MissionState.NAVIGATING, now_sec)
            else:
                # Quick 2-D passage snapshot for telemetry
                if self.robot_pose is not None and self.occ_grid.is_initialized:
                    rx, ry, ryaw = self.robot_pose
                    goal = self.goal_mgr.get_goal()
                    gx = goal.x if goal else None
                    gy = goal.y if goal else None
                    lookaround_res = self.occ_grid.evaluate_360_passages(rx, ry, gx, gy)
                    passages = lookaround_res.get("passages", [])
                    if passages:
                        best_deg = lookaround_res.get("best_heading_deg", 0.0)
                        widest_m = lookaround_res.get("widest_corridor_m", 0.0)
                        self.nav_reason = (
                            f"360° Survey: {len(passages)} passages found "
                            f"(best: {best_deg:+.1f}°, width {widest_m:.2f}m)"
                        )

                # Trigger 360° lookaround recovery — ONCE per block (guard prevents re-fire each tick)
                if self.mission_mgr.replan_count < self.mission_mgr.max_replan_retries:
                    self.get_logger().warn(
                        "Initial plan blocked — triggering 360° lookaround recovery scan "
                        f"(replan_count={self.mission_mgr.replan_count}/{self.mission_mgr.max_replan_retries})..."
                    )
                    self._recovery_pending = True
                    if self.fault_cli.service_is_ready():
                        self.fault_cli.call_async(Trigger.Request())
                    self.mission_mgr.transition_to(MissionState.RECOVERY_WAIT, now_sec, "PLAN_BLOCKED_WAIT_360_LOOKAROUND")
                    self.cmd_ownership = "YIELDED_TO_RECOVERY"
                else:
                    ctx = self._build_context(now_sec)
                    can_fit = self.last_passage_eval.get("can_fit", True)
                    fail_code = FailureCode.INSUFFICIENT_CLEARANCE if not can_fit else FailureCode.NO_SAFE_PATH
                    self.mission_mgr.record_planning_failure(now_sec, fail_code, context=ctx)

        elif self.mission_mgr.state == MissionState.NAVIGATING:
            # 1. Check recovery integration & mutual exclusion
            if self.recovery_state not in ("NORMAL", "VERIFY") or self.confidence_decision == "RECOVER":
                self.get_logger().warn(
                    f"Yielding control to Recovery (recovery_state={self.recovery_state}, confidence={self.confidence_decision})."
                )
                if self.robot_pose is not None:
                    rx, ry, ryaw = self.robot_pose
                    ox = rx + 0.60 * math.cos(ryaw)
                    oy = ry + 0.60 * math.sin(ryaw)
                    self.occ_grid.mark_blocked_region(ox, oy, radius_m=0.45)
                self._recovery_pending = True  # Recovery already active (not started by us — just yield)
                self.mission_mgr.transition_to(MissionState.RECOVERY_WAIT, now_sec, "YIELD_TO_RECOVERY")
                self.cmd_ownership = "YIELDED_TO_RECOVERY"
                self.nav_reason = "Recovery active: yielding control to recovery state machine"
                self._publish_telemetry(now_sec)
                return

            # 2. Check if goal is reached
            goal = self.goal_mgr.get_goal()
            rx, ry, ryaw = self.robot_pose
            dist_to_goal = math.hypot(goal.x - rx, goal.y - ry) if goal else 0.0

            if self.goal_checker.is_goal_reached(rx, ry, ryaw, goal, now_sec):
                self.get_logger().info(f"Goal successfully reached at ({rx:.2f}, {ry:.2f})!")
                path_len = sum(
                    math.hypot(self.raw_planned_path[i+1][0] - self.raw_planned_path[i][0],
                               self.raw_planned_path[i+1][1] - self.raw_planned_path[i][1])
                    for i in range(len(self.raw_planned_path) - 1)
                ) if len(self.raw_planned_path) > 1 else 0.0
                self.mission_mgr.trigger_success(now_sec, final_error_m=dist_to_goal, path_length_m=path_len)
                self._stop_robot()
                self.cmd_ownership = "GOAL_REACHED_STOP"
                self.nav_reason = "Goal reached within tolerance"
                self._publish_telemetry(now_sec)
                return

            # 3. Check dynamic replanning triggers
            should_replan, reason = self.replanner.should_replan_due_to_obstacle(
                self.waypoints, self.follower.current_waypoint_idx, self.occ_grid, now_sec
            )
            if should_replan:
                blocked = self.replanner.find_blocked_segment(
                    self.waypoints, self.follower.current_waypoint_idx, self.occ_grid
                )
                if blocked is not None:
                    self.occ_grid.mark_blocked_region(blocked[0], blocked[1], radius_m=0.45)
                self.get_logger().warn(f"Replanning triggered by obstacle: {reason}")
                self.mission_mgr.transition_to(MissionState.REPLANNING, now_sec, reason)
                self._publish_telemetry(now_sec)
                return

            if not should_replan:
                should_replan, reason = self.replanner.should_replan_due_to_deviation(
                    self.cross_track_error, now_sec
                )
                if should_replan:
                    self.get_logger().warn(f"Replanning triggered by deviation: {reason}")
                    self.mission_mgr.transition_to(MissionState.REPLANNING, now_sec, reason)
                    self._publish_telemetry(now_sec)
                    return

            # 4. Multi-Cost Clearance and Terrain Evaluation at Robot Pose
            pt = self.occ_grid.world_to_map(rx, ry)
            clearance_m = self.occ_grid.get_clearance(pt[0], pt[1]) if pt else 1.0
            cell_cost = self.occ_grid.get_cost(pt[0], pt[1]) if pt else 0.0
            self.current_clearance_m = clearance_m
            self.current_terrain_cost = cell_cost

            # Continuous Clearance factor [0.45 .. 1.0]
            span = max(0.1, self.occ_grid.proximity_radius_m - self.occ_grid.inflation_radius_m)
            clearance_factor = min(1.0, max(0.45, (clearance_m - self.occ_grid.inflation_radius_m) / span))

            # Continuous Terrain factor [0.40 .. 1.0]
            terrain_factor = max(0.40, 1.0 - (cell_cost / 100.0) * 0.60)

            # Confidence factor [0.0, 0.5, 1.0]
            speed_scale = 0.50 if self.confidence_decision == "VERIFY" else (0.0 if self.confidence_decision == "RECOVER" else 1.0)
            self.current_speed_scale = speed_scale * clearance_factor * terrain_factor

            # Execute Adaptive Path Following
            vx, wz, lookahead_wp, cross_err = self.follower.compute_commands(
                rx, ry, ryaw, self.waypoints,
                speed_scale=speed_scale,
                terrain_factor=terrain_factor,
                clearance_factor=clearance_factor,
            )
            self.current_lookahead = lookahead_wp
            self.cross_track_error = cross_err
            self.last_cmd_vx = vx
            self.last_cmd_wz = wz
            self.cmd_ownership = "NAVIGATING_ACTIVE"

            # Formulate clear operator explanation
            if speed_scale < 1.0:
                self.nav_reason = "Visual confidence degraded: slow for verification"
            elif terrain_factor < 0.70:
                self.nav_reason = f"High terrain cost: reduced speed ({vx:.2f} m/s)"
            elif clearance_factor < 0.70:
                self.nav_reason = f"Narrow obstacle clearance: cautious speed ({vx:.2f} m/s)"
            else:
                self.nav_reason = "Following safest available route"

            twist = Twist()
            twist.linear.x = float(vx)
            twist.angular.z = float(wz)
            self.cmd_vel_pub.publish(twist)

        elif self.mission_mgr.state == MissionState.REPLANNING:
            self.cmd_ownership = "REPLANNING"
            self.nav_reason = "Computing alternate detour around detected blockage"

            # Skip replanning if already waiting for a recovery scan to finish
            if self._recovery_pending:
                self.cmd_ownership = "YIELDED_TO_RECOVERY"
                self.nav_reason = "Waiting for 360° recovery scan before retrying replan"
                self._publish_telemetry(now_sec)
                return

            # 360-degree look-around snapshot for telemetry
            if self.robot_pose is not None and self.occ_grid.is_initialized:
                rx, ry, ryaw = self.robot_pose
                goal = self.goal_mgr.get_goal()
                gx = goal.x if goal else None
                gy = goal.y if goal else None
                lookaround_res = self.occ_grid.evaluate_360_passages(rx, ry, gx, gy)
                passages = lookaround_res.get("passages", [])
                if passages:
                    best_deg = lookaround_res.get("best_heading_deg", 0.0)
                    widest_m = lookaround_res.get("widest_corridor_m", 0.0)
                    self.nav_reason = (
                        f"360° Lookaround: {len(passages)} passages found "
                        f"(best: {best_deg:+.1f}°, width {widest_m:.2f}m)"
                    )

            success = self._compute_and_set_path(now_sec, replan_reason="OBSTACLE_OR_DEVIATION_REPLAN")
            if success:
                self._recovery_pending = False
                self.mission_mgr.transition_to(MissionState.NAVIGATING, now_sec)
            else:
                if self.mission_mgr.replan_count < self.mission_mgr.max_replan_retries:
                    self.get_logger().warn(
                        "Replan blocked — triggering 360° lookaround recovery scan "
                        f"(replan_count={self.mission_mgr.replan_count}/{self.mission_mgr.max_replan_retries})..."
                    )
                    self._recovery_pending = True
                    if self.fault_cli.service_is_ready():
                        self.fault_cli.call_async(Trigger.Request())
                    self.mission_mgr.transition_to(MissionState.RECOVERY_WAIT, now_sec, "REPLAN_BLOCKED_WAIT_360_LOOKAROUND")
                    self.cmd_ownership = "YIELDED_TO_RECOVERY"
                else:
                    ctx = self._build_context(now_sec)
                    can_fit = self.last_passage_eval.get("can_fit", True)
                    fail_code = FailureCode.INSUFFICIENT_CLEARANCE if not can_fit else FailureCode.PATH_BLOCKED
                    self.mission_mgr.record_planning_failure(now_sec, fail_code, context=ctx)

        elif self.mission_mgr.state == MissionState.RECOVERY_WAIT:
            # Strictly do not publish /cmd_vel; yield to recovery node
            self.cmd_ownership = "YIELDED_TO_RECOVERY"
            rs = self.recovery_state
            if "360" in rs or "LOOKAROUND" in rs:
                self.nav_reason = "Retry mechanism: waiting for 360° panoramic path lookaround scan"
            elif self._recovery_pending:
                self.nav_reason = f"Recovery in progress: {rs} (waiting for 360° path scan to complete)"
            else:
                self.nav_reason = f"Recovery in progress: {rs}"

            # When recovery returns NORMAL *after* a scan was actually triggered, resume replanning.
            # Guard with _recovery_pending to avoid spurious exit on first tick (recovery_state starts as "NORMAL").
            if self._recovery_pending and rs == "NORMAL" and self.confidence_decision in ("CONTINUE", "VERIFY"):
                self.get_logger().info(
                    "360° lookaround recovery completed — clearing recovery guard and replanning."
                )
                self._recovery_pending = False
                # Reset replan counter so the post-recovery plan attempt is not pre-failed
                self.mission_mgr.replan_count = 0
                self.mission_mgr.transition_to(MissionState.REPLANNING, now_sec, "RECOVERY_RESUME")
            elif rs == "FAILED_SAFE":
                ctx = self._build_context(now_sec)
                self.mission_mgr.trigger_failure(FailureCode.RECOVERY_BUDGET_EXHAUSTED, now_sec, context=ctx)


        elif self.mission_mgr.state in (MissionState.GOAL_REACHED, MissionState.MISSION_FAILED):
            self.cmd_ownership = "TERMINAL_STOP"
            if self.mission_mgr.state == MissionState.MISSION_FAILED:
                rec = self.mission_mgr.failure_record
                self.nav_reason = f"FAILED: {rec.get('human_reason', 'No safe route')}" if rec else "Mission failed"
            else:
                self.nav_reason = "Mission completed successfully"


        self._publish_telemetry(now_sec)

    def _compute_and_set_path(self, now_sec: float, replan_reason: str = "INITIAL_PLAN") -> bool:
        """Run multi-cost A* planner, smooth path, and generate waypoints."""
        goal = self.goal_mgr.get_goal()
        if goal is None or self.robot_pose is None or not self.occ_grid.is_initialized:
            return False

        old_path_len = 0.0
        for i in range(len(self.raw_planned_path) - 1):
            old_path_len += math.hypot(
                self.raw_planned_path[i+1][0] - self.raw_planned_path[i][0],
                self.raw_planned_path[i+1][1] - self.raw_planned_path[i][1]
            )

        rx, ry, _ = self.robot_pose
        raw_path = self.planner.plan(self.occ_grid, (rx, ry), (goal.x, goal.y))
        if not raw_path:
            self._publish_replan_diagnostics(now_sec, replan_reason, old_path_len, 0.0, 0.0, False)
            return False

        smoothed = self.smoother.smooth(raw_path, self.occ_grid)
        waypoints = self.wp_generator.generate(smoothed, final_yaw=goal.yaw)

        new_path_len = 0.0
        for i in range(len(smoothed) - 1):
            new_path_len += math.hypot(
                smoothed[i+1][0] - smoothed[i][0],
                smoothed[i+1][1] - smoothed[i][1]
            )
        path_diff = abs(new_path_len - old_path_len)

        self.raw_planned_path = smoothed
        self.waypoints = waypoints
        self.follower.reset()
        self.goal_checker.reset()
        self.replanner.record_replan(now_sec)

        # Publish visualizations
        self._publish_path_and_waypoints()
        self._publish_replan_diagnostics(now_sec, replan_reason, old_path_len, new_path_len, path_diff, True)
        return True

    def _publish_replan_diagnostics(
        self,
        now_sec: float,
        reason: str,
        old_length: float,
        new_length: float,
        path_diff: float,
        success: bool,
    ) -> None:
        goal = self.goal_mgr.get_goal()
        dist_to_goal = 0.0
        if goal is not None and self.robot_pose is not None:
            dist_to_goal = math.hypot(goal.x - self.robot_pose[0], goal.y - self.robot_pose[1])
        diag_data = {
            "timestamp": now_sec,
            "replan_reason": reason,
            "status": "REPLAN_SUCCEEDED" if success else "REPLAN_FAILED",
            "old_path_length_m": round(old_length, 3),
            "new_path_length_m": round(new_length, 3),
            "path_difference_m": round(path_diff, 3),
            "blocked_regions_count": len(self.occ_grid.persistent_blocked_regions),
            "robot_pose": [round(v, 3) for v in self.robot_pose] if self.robot_pose else None,
            "dist_to_goal_m": round(dist_to_goal, 3),
        }
        msg = String()
        msg.data = json.dumps(diag_data)
        self.replan_diag_pub.publish(msg)

    def _stop_robot(self) -> None:
        """Send zero velocity command."""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.last_cmd_vx = 0.0
        self.last_cmd_wz = 0.0

    def _publish_path_and_waypoints(self) -> None:
        """Publish nav_msgs/msg/Path and PoseArray for RViz and Dashboard."""
        stamp = self.get_clock().now().to_msg()
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.map_frame

        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.map_frame

        for wp in self.waypoints:
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = float(wp.x)
            ps.pose.position.y = float(wp.y)
            ps.pose.position.z = 0.0
            ps.pose.orientation = self._yaw_to_quaternion(wp.yaw)
            path_msg.poses.append(ps)
            pose_array.poses.append(ps.pose)

        self.path_pub.publish(path_msg)
        self.waypoints_pub.publish(pose_array)

    def _publish_telemetry(self, now_sec: float) -> None:
        """Publish JSON state and diagnostic messages."""
        stamp = self.get_clock().now().to_msg()
        goal = self.goal_mgr.get_goal()

        dist_to_goal = 0.0
        if goal is not None and self.robot_pose is not None:
            dist_to_goal = math.hypot(goal.x - self.robot_pose[0], goal.y - self.robot_pose[1])

        path_length = 0.0
        for i in range(len(self.raw_planned_path) - 1):
            p1 = self.raw_planned_path[i]
            p2 = self.raw_planned_path[i + 1]
            path_length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        # 1. State JSON with Structured Failure & Success Records
        state_msg = String()
        state_data = {
            "mission_state": self.mission_mgr.state.to_string(),
            "mission_state_code": int(self.mission_mgr.state),
            "goal": goal.to_dict() if goal else None,
            "dist_to_goal_m": round(dist_to_goal, 3),
            "current_waypoint_idx": self.follower.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "replan_count": self.mission_mgr.replan_count,
            "failure_reason": self.mission_mgr.failure_reason,
            "failure_record": self.mission_mgr.failure_record,
            "success_record": self.mission_mgr.success_record,
            "nav_reason": self.nav_reason,
            "clearance_m": round(self.current_clearance_m, 2),
            "terrain_cost": round(self.current_terrain_cost, 1),
            "speed_scale": round(self.current_speed_scale, 2),
            "cmd_vx": round(self.last_cmd_vx, 3),
            "cmd_wz": round(self.last_cmd_wz, 3),
            "confidence_decision": self.confidence_decision,
            "recovery_state": self.recovery_state,
            "cmd_ownership": self.cmd_ownership,
            "passage": self.last_passage_eval,
            "persistent_blocked_regions": [
                {"x": round(bx, 2), "y": round(by, 2), "radius_m": round(br, 2)}
                for bx, by, br in self.occ_grid.persistent_blocked_regions
            ],
        }
        state_msg.data = json.dumps(state_data)
        self.state_pub.publish(state_msg)

        # 2. Vehicle & Passage Diagnostics
        v_diag_msg = String()
        v_diag_data = {
            "timestamp": now_sec,
            "vehicle": {
                "length_m": VehicleGeometry.TOTAL_LENGTH_M,
                "width_m": VehicleGeometry.TOTAL_WIDTH_M,
                "height_m": VehicleGeometry.CHASSIS_HEIGHT_M,
                "inscribed_radius_m": VehicleGeometry.INSCRIBED_RADIUS_M,
                "circumscribed_radius_m": VehicleGeometry.CIRCUMSCRIBED_RADIUS_M,
                "nominal_passage_m": VehicleGeometry.NOMINAL_PASSAGE_WIDTH_M,
                "tight_passage_limit_m": VehicleGeometry.TIGHT_PASSAGE_LIMIT_M,
                "min_turn_diameter_m": VehicleGeometry.MIN_TURN_DIAMETER_M,
                "safety_margin_m": VehicleGeometry.SAFETY_MARGIN_M,
            },
            "passage": self.last_passage_eval,
            "status": self.last_passage_eval.get("passage_status", "SAFE"),
            "can_fit": self.last_passage_eval.get("can_fit", True),
            "can_turn": self.last_passage_eval.get("can_turn", True),
            "available_clear_width_m": round(self.last_passage_eval.get("available_width_m", 0.0), 3),
            "required_clear_width_m": round(self.last_passage_eval.get("required_width_m", VehicleGeometry.NOMINAL_PASSAGE_WIDTH_M), 3),
            "clearance_margin_m": round(self.last_passage_eval.get("clearance_margin_m", 0.0), 3),
        }
        v_diag_msg.data = json.dumps(v_diag_data)
        self.vehicle_diag_pub.publish(v_diag_msg)

        # 3. Diagnostic Array
        diag_msg = NavigationDiagnostics.build_diagnostic_array(
            stamp=stamp,
            mission_state=self.mission_mgr.state.to_string(),
            goal_dict=goal.to_dict() if goal else None,
            dist_to_goal=dist_to_goal,
            current_wp_idx=self.follower.current_waypoint_idx,
            total_wps=len(self.waypoints),
            path_length_m=path_length,
            replan_count=self.mission_mgr.replan_count,
            cmd_vx=self.last_cmd_vx,
            cmd_wz=self.last_cmd_wz,
            confidence_decision=self.confidence_decision,
            recovery_state=self.recovery_state,
            ownership=self.cmd_ownership,
        )
        self.diag_pub.publish(diag_msg)

        # 3. Markers for RViz
        self._publish_markers(goal, stamp)

    def _publish_markers(self, goal: Optional[NavigationGoal], stamp) -> None:
        """Publish RViz markers for target goal, lookahead, and blocked obstacles."""
        marker_arr = MarkerArray()

        # Goal Marker
        if goal is not None:
            m_goal = Marker()
            m_goal.header.stamp = stamp
            m_goal.header.frame_id = self.map_frame
            m_goal.ns = "goal"
            m_goal.id = 0
            m_goal.type = Marker.CYLINDER
            m_goal.action = Marker.ADD
            m_goal.pose.position.x = float(goal.x)
            m_goal.pose.position.y = float(goal.y)
            m_goal.pose.position.z = 0.2
            m_goal.scale.x = 0.4
            m_goal.scale.y = 0.4
            m_goal.scale.z = 0.4
            m_goal.color.r = 0.1
            m_goal.color.g = 0.8
            m_goal.color.b = 0.2
            m_goal.color.a = 0.8
            marker_arr.markers.append(m_goal)

        # Lookahead Marker
        if self.current_lookahead is not None:
            m_look = Marker()
            m_look.header.stamp = stamp
            m_look.header.frame_id = self.map_frame
            m_look.ns = "lookahead"
            m_look.id = 1
            m_look.type = Marker.SPHERE
            m_look.action = Marker.ADD
            m_look.pose.position.x = float(self.current_lookahead.x)
            m_look.pose.position.y = float(self.current_lookahead.y)
            m_look.pose.position.z = 0.1
            m_look.scale.x = 0.2
            m_look.scale.y = 0.2
            m_look.scale.z = 0.2
            m_look.color.r = 1.0
            m_look.color.g = 0.5
            m_look.color.b = 0.0
            m_look.color.a = 0.9
            marker_arr.markers.append(m_look)

        # Blocked Obstacle Markers
        for idx, (bx, by, br) in enumerate(self.occ_grid.persistent_blocked_regions):
            m_block = Marker()
            m_block.header.stamp = stamp
            m_block.header.frame_id = self.map_frame
            m_block.ns = "obstacles"
            m_block.id = 100 + idx
            m_block.type = Marker.CYLINDER
            m_block.action = Marker.ADD
            m_block.pose.position.x = float(bx)
            m_block.pose.position.y = float(by)
            m_block.pose.position.z = 0.15
            m_block.scale.x = float(br * 2)
            m_block.scale.y = float(br * 2)
            m_block.scale.z = 0.3
            m_block.color.r = 0.95
            m_block.color.g = 0.15
            m_block.color.b = 0.15
            m_block.color.a = 0.65
            marker_arr.markers.append(m_block)

        if marker_arr.markers:
            self.marker_pub.publish(marker_arr)

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
