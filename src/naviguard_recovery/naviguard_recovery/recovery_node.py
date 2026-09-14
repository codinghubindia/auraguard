"""NAVIGUARD Autonomous Closed-Loop Recovery ROS 2 Node.

Subscribes to Phase 7 confidence decisions, SLAM pose, and occupancy maps;
executes deterministic recovery maneuvers (Safe Stop, Backtrack, Visual Reacquisition);
and manages relocalization verification.
"""

import json
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from naviguard_recovery.recovery_controller import RecoveryController
from naviguard_recovery.recovery_diagnostics import RecoveryDiagnosticsBuilder
from naviguard_recovery.recovery_planner import RecoveryPlanner, RecoveryStrategy
from naviguard_recovery.recovery_state_machine import (
    RecoveryBudget,
    RecoveryState,
    RecoveryStateMachine,
    RecoveryStateMachineConfig,
)
from naviguard_recovery.relocalization_manager import RelocalizationManager
from naviguard_recovery.trusted_state_manager import (
    TrustedCheckpoint,
    TrustedStateManager,
    TrustedStateManagerConfig,
)


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class NaviguardRecoveryNode(Node):
    """Central Recovery Node managing the closed-loop recovery protocol."""

    def __init__(self) -> None:
        super().__init__('recovery_node')

        # 1. Parameter Declarations
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('safe_stop_dwell_sec', 0.50)
        self.declare_parameter('verification_dwell_sec', 1.00)
        self.declare_parameter('max_attempts', 8)
        self.declare_parameter('max_total_duration_sec', 120.0)
        self.declare_parameter('max_backtrack_dist_m', 5.0)
        self.declare_parameter('max_rotation_deg', 1440.0)
        self.declare_parameter('log_events_json', '')

        rate_hz = float(self.get_parameter('rate_hz').value)
        fsm_cfg = RecoveryStateMachineConfig(
            safe_stop_dwell_sec=float(self.get_parameter('safe_stop_dwell_sec').value),
            verification_dwell_sec=float(self.get_parameter('verification_dwell_sec').value),
        )

        # 2. Components
        self.fsm = RecoveryStateMachine(fsm_cfg)
        self.fsm.budget.max_attempts = int(self.get_parameter('max_attempts').value)
        self.fsm.budget.max_total_duration_sec = float(self.get_parameter('max_total_duration_sec').value)
        self.fsm.budget.max_backtrack_dist_m = float(self.get_parameter('max_backtrack_dist_m').value)
        self.fsm.budget.max_rotation_deg = float(self.get_parameter('max_rotation_deg').value)

        self.trusted_mgr = TrustedStateManager(TrustedStateManagerConfig())
        self.planner = RecoveryPlanner()
        self.controller = RecoveryController()
        self.reloc_mgr = RelocalizationManager(min_dwell_sec=fsm_cfg.verification_dwell_sec)
        self.diag_builder = RecoveryDiagnosticsBuilder()

        # 3. State Variables
        self.current_pose_map: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, yaw)
        self.current_pose_odom: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_vel: Tuple[float, float] = (0.0, 0.0)                   # (vx, wz)

        # Cached Phase 7 confidence & decision
        self.phase7_state_str: str = "CONTINUE"
        self.phase7_overall_conf: float = 1.0
        self.phase7_loc_conf: float = 1.0
        self.phase7_vis_conf: float = 1.0
        self.phase7_primary_reason: str = "NOMINAL_OPERATION"

        # Cached SLAM diagnostics
        self.slam_tracking_state: str = "OK"

        # Cached Occupancy Grid
        self.grid_data: Optional[List[int]] = None
        self.grid_res: float = 0.05
        self.grid_w: int = 0
        self.grid_h: int = 0
        self.grid_ox: float = 0.0
        self.grid_oy: float = 0.0

        # Active action execution state
        self.action_in_progress: bool = False
        self.current_waypoints: List[Tuple[float, float, float]] = []
        self.waypoint_index: int = 0
        self.target_rotation_yaw: float = 0.0
        self.rotation_start_yaw: float = 0.0
        self.active_strategy_enum: RecoveryStrategy = RecoveryStrategy.STOP_AND_RELOCALIZE
        self.lookaround_accum_rad: float = 0.0
        self.lookaround_last_yaw: float = 0.0
        self.lookaround_direction: int = 1
        self.lookaround_360_eval: dict = {}

        self.last_timer_time = time.time()

        # 4. Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/recovery/state', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/recovery/diagnostics', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/recovery/visualization', 10)

        # 5. Subscriptions
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        transient_local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.decision_sub = self.create_subscription(
            String,
            '/naviguard/decision',
            self.decision_callback,
            reliable_qos,
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/slam/pose',
            self.slam_pose_callback,
            reliable_qos,
        )
        self.slam_diag_sub = self.create_subscription(
            DiagnosticArray,
            '/slam/diagnostics',
            self.slam_diagnostics_callback,
            reliable_qos,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            reliable_qos,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/slam/map',
            self.map_callback,
            transient_local_qos,
        )

        # 6. Services
        self.trigger_srv = self.create_service(
            Trigger,
            '/recovery/trigger_manual_recovery',
            self.manual_trigger_callback,
        )
        self.reset_srv = self.create_service(
            Trigger,
            '/recovery/reset_budget',
            self.reset_budget_callback,
        )

        # 7. Lifecycle Timer
        self.timer = self.create_timer(1.0 / rate_hz, self.timer_callback)

        self.get_logger().info(
            f"NaviguardRecoveryNode initialized at {rate_hz:.1f} Hz. "
            f"Max attempts: {self.fsm.budget.max_attempts}, Safe stop dwell: {fsm_cfg.safe_stop_dwell_sec:.2f}s"
        )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def decision_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.phase7_state_str = str(data.get("state", "CONTINUE"))
            self.phase7_primary_reason = str(data.get("primary_reason", "NONE"))
            scores = data.get("scores", {})
            self.phase7_overall_conf = float(scores.get("overall", 1.0))
            self.phase7_loc_conf = float(scores.get("localization", 1.0))
            self.phase7_vis_conf = float(scores.get("visual", 1.0))
        except Exception as e:
            self.get_logger().warn(f"Failed to parse decision payload: {e}")

    def slam_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose_map = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
        )

    def slam_diagnostics_callback(self, msg: DiagnosticArray) -> None:
        for status in msg.status:
            for kv in status.values:
                if kv.key == "tracking_state":
                    self.slam_tracking_state = kv.value

    def odom_callback(self, msg: Odometry) -> None:
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_pose_odom = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
        )
        self.current_vel = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.grid_data = list(msg.data)
        self.grid_res = float(msg.info.resolution)
        self.grid_w = int(msg.info.width)
        self.grid_h = int(msg.info.height)
        self.grid_ox = float(msg.info.origin.position.x)
        self.grid_oy = float(msg.info.origin.position.y)

    def manual_trigger_callback(self, request, response) -> Trigger.Response:
        now_sec = self._get_now_sec()
        self.fsm.transition_to(RecoveryState.SAFE_STOP, now_sec, "OPERATOR_MANUAL_TRIGGER")
        response.success = True
        response.message = "Recovery lifecycle triggered manually by operator."
        return response

    def reset_budget_callback(self, request, response) -> Trigger.Response:
        now_sec = self._get_now_sec()
        self.fsm.reset(now_sec)
        response.success = True
        response.message = "Recovery budget and state machine reset."
        return response

    def _get_now_sec(self) -> float:
        clock_now = self.get_clock().now()
        s = float(clock_now.nanoseconds) * 1e-9
        return s if s > 0.0 else time.time()

    # -------------------------------------------------------------------------
    # Main Timer Lifecycle Loop
    # -------------------------------------------------------------------------

    def timer_callback(self) -> None:
        now_sec = self._get_now_sec()
        dt = max(0.001, min(0.2, now_sec - self.last_timer_time))
        self.last_timer_time = now_sec

        # 1. Update Trusted Checkpoint Buffer during high-confidence CONTINUE
        if self.fsm.state == RecoveryState.NORMAL:
            self.trusted_mgr.maybe_add_checkpoint(
                timestamp_sec=now_sec,
                map_pose=self.current_pose_map,
                odom_pose=self.current_pose_odom,
                conf_overall=self.phase7_overall_conf,
                conf_loc=self.phase7_loc_conf,
                conf_vis=self.phase7_vis_conf,
                velocity=self.current_vel,
                phase7_state=self.phase7_state_str,
            )

        # 2. Feed Phase 7 decision into recovery FSM
        self.fsm.update_phase7_input(self.phase7_state_str, self.phase7_primary_reason, now_sec)

        # 3. Check Relocalization & Verification status
        reloc_ok = (self.slam_tracking_state.upper() == "OK" and self.phase7_overall_conf >= 0.70)
        verify_ok, dwell_verify = self.reloc_mgr.evaluate_verification(
            current_time=now_sec,
            conf_overall=self.phase7_overall_conf,
            conf_loc=self.phase7_loc_conf,
            conf_vis=self.phase7_vis_conf,
            slam_tracking_state=self.slam_tracking_state,
        )

        # 4. Handle State-Specific Actions
        cmd_vx = 0.0
        cmd_wz = 0.0

        if self.fsm.state == RecoveryState.NORMAL:
            self.action_in_progress = False

        elif self.fsm.state == RecoveryState.VERIFY:
            # During VERIFY, robot slows down if needed, no recovery motion
            self.action_in_progress = False

        elif self.fsm.state == RecoveryState.SAFE_STOP:
            cmd_vx, cmd_wz = self.controller.compute_stop()
            self.action_in_progress = False

        elif self.fsm.state == RecoveryState.SELECT_CHECKPOINT:
            cmd_vx, cmd_wz = self.controller.compute_stop()
            # Pick best checkpoint
            ckpt = self.trusted_mgr.select_best_checkpoint(
                current_pose=self.current_pose_map,
                current_time=now_sec,
                grid_data=self.grid_data,
                grid_res=self.grid_res,
                grid_w=self.grid_w,
                grid_h=self.grid_h,
                grid_ox=self.grid_ox,
                grid_oy=self.grid_oy,
            )
            # Pick recovery strategy
            strategy = self.planner.select_strategy(
                failure_reason=self.fsm.failure_reason,
                attempt_number=self.fsm.budget.attempt_count + 1,
                has_checkpoint=ckpt is not None,
            )
            self.active_strategy_enum = strategy
            self.fsm.active_strategy = strategy.to_string()
            self.fsm.budget.record_attempt(
                strategy_name=strategy.to_string(),
                reason=self.fsm.failure_reason,
                timestamp_sec=now_sec,
                conf_before=self.phase7_overall_conf,
            )

            # Initialize chosen strategy
            if strategy == RecoveryStrategy.SHORT_BACKTRACK and ckpt is not None:
                success, wps, reason = self.planner.plan_backtrack(
                    current_pose=self.current_pose_map,
                    target_checkpoint=ckpt,
                    grid_data=self.grid_data,
                    grid_res=self.grid_res,
                    grid_w=self.grid_w,
                    grid_h=self.grid_h,
                    grid_ox=self.grid_ox,
                    grid_oy=self.grid_oy,
                )
                if success and wps:
                    self.current_waypoints = wps
                    self.waypoint_index = 0
                    self.action_in_progress = True
                    self.fsm.transition_to(RecoveryState.RECOVER, now_sec)
                else:
                    # Fallback to 360 lookaround if backtrack path blocked
                    self.active_strategy_enum = RecoveryStrategy.LOOKAROUND_360_SCAN
                    self.fsm.active_strategy = self.active_strategy_enum.to_string()
                    self.lookaround_accum_rad = 0.0
                    self.lookaround_last_yaw = self.current_pose_map[2]
                    self.lookaround_direction = 1
                    self.lookaround_360_eval = {}
                    self.action_in_progress = True
                    self.fsm.transition_to(RecoveryState.LOOKAROUND_360_SCAN, now_sec)

            elif strategy == RecoveryStrategy.LOOKAROUND_360_SCAN:
                self.lookaround_accum_rad = 0.0
                self.lookaround_last_yaw = self.current_pose_map[2]
                self.lookaround_direction = 1
                self.lookaround_360_eval = {}
                self.action_in_progress = True
                self.fsm.transition_to(RecoveryState.LOOKAROUND_360_SCAN, now_sec)

            elif strategy == RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION:
                dyaw, tyaw, scan_name = self.planner.plan_rotation_scan(
                    self.current_pose_map[2],
                    attempt_number=self.fsm.budget.attempt_count,
                )
                self.target_rotation_yaw = tyaw
                self.rotation_start_yaw = self.current_pose_map[2]
                self.action_in_progress = True
                self.fsm.transition_to(RecoveryState.RECOVER, now_sec)

            else:  # STOP_AND_RELOCALIZE
                self.action_in_progress = False
                self.fsm.transition_to(RecoveryState.RELOCALIZE, now_sec)

        elif self.fsm.state in (RecoveryState.RECOVER, RecoveryState.LOOKAROUND_360_SCAN):
            if self.active_strategy_enum == RecoveryStrategy.SHORT_BACKTRACK:
                if self.current_waypoints and self.waypoint_index < len(self.current_waypoints):
                    target_wp = self.current_waypoints[self.waypoint_index]
                    cmd_vx, cmd_wz, reached = self.controller.compute_backtrack_step(
                        self.current_pose_map, target_wp
                    )
                    self.fsm.budget.record_motion(abs(cmd_vx) * dt, 0.0)
                    if reached:
                        self.waypoint_index += 1
                        if self.waypoint_index >= len(self.current_waypoints):
                            self.action_in_progress = False
                else:
                    self.action_in_progress = False

            elif self.active_strategy_enum == RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION:
                cmd_vx, cmd_wz, reached = self.controller.compute_rotation_step(
                    self.current_pose_map[2], self.target_rotation_yaw
                )
                self.fsm.budget.record_motion(0.0, float(np.degrees(abs(cmd_wz) * dt)))
                if reached:
                    self.action_in_progress = False
                # Early success exit if visual confidence has recovered
                if self.phase7_vis_conf >= 0.70 and self.phase7_overall_conf >= 0.75:
                    self.action_in_progress = False

            elif self.active_strategy_enum == RecoveryStrategy.LOOKAROUND_360_SCAN:
                cmd_vx, cmd_wz, self.lookaround_accum_rad, reached = self.controller.compute_lookaround_360_step(
                    current_yaw=self.current_pose_map[2],
                    last_yaw=self.lookaround_last_yaw,
                    accumulated_yaw_rad=self.lookaround_accum_rad,
                    target_total_rad=2.0 * np.pi,
                    direction=self.lookaround_direction,
                )
                self.lookaround_last_yaw = self.current_pose_map[2]
                self.fsm.budget.record_motion(0.0, float(np.degrees(abs(cmd_wz) * dt)))

                # Continuously survey 360-degree clear paths & narrow passages
                self.lookaround_360_eval = self.planner.evaluate_360_passages(
                    current_pose=self.current_pose_map,
                    grid_data=self.grid_data,
                    grid_res=self.grid_res,
                    grid_w=self.grid_w,
                    grid_h=self.grid_h,
                    grid_ox=self.grid_ox,
                    grid_oy=self.grid_oy,
                )

                # Retry mechanism WAITS until full 360-degree sweep is completely performed
                if reached:
                    self.get_logger().info(
                        f"360-degree lookaround complete ({np.degrees(self.lookaround_accum_rad):.1f} deg). "
                        f"Found {len(self.lookaround_360_eval.get('passages', []))} viable passages. "
                        f"Best heading: {self.lookaround_360_eval.get('best_heading_deg', 0.0):.1f} deg."
                    )
                    self.action_in_progress = False
            else:
                self.action_in_progress = False

        elif self.fsm.state in (
            RecoveryState.VISUAL_REACQUISITION_OBSERVATION,
            RecoveryState.RELOCALIZE,
            RecoveryState.VERIFY_RECOVERY,
            RecoveryState.REPLAN,
            RecoveryState.RESUME,
            RecoveryState.FAILED_SAFE,
        ):
            cmd_vx, cmd_wz = self.controller.compute_stop()

        # 5. Step state machine transitions
        self.fsm.step_recovery_lifecycle(
            stamp_sec=now_sec,
            dt=dt,
            action_in_progress=self.action_in_progress,
            relocalization_ok=reloc_ok,
            verification_ok=verify_ok,
        )

        # 6. Publish command velocity if in recovery states
        if self.fsm.state not in (RecoveryState.NORMAL, RecoveryState.VERIFY):
            twist = Twist()
            twist.linear.x = float(cmd_vx)
            twist.angular.z = float(cmd_wz)
            self.cmd_vel_pub.publish(twist)

        # 7. Publish Diagnostics & Visualizations
        self._publish_state_json(now_sec)
        self._publish_diagnostics(now_sec)
        self._publish_markers(now_sec)

    def _publish_state_json(self, now_sec: float) -> None:
        msg = String()
        data = {
            "recovery_state": self.fsm.state.to_string(),
            "recovery_state_code": int(self.fsm.state),
            "active_strategy": self.fsm.active_strategy,
            "failure_reason": self.fsm.failure_reason,
            "dwell_sec": float(now_sec - self.fsm.state_entry_time),
            "attempts": int(self.fsm.budget.attempt_count),
            "max_attempts": int(self.fsm.budget.max_attempts),
            "budget": self.fsm.budget.to_dict(),
            "current_attempt": self.fsm.budget.current_attempt.to_dict() if self.fsm.budget.current_attempt else None,
            "attempt_history": [att.to_dict() for att in self.fsm.budget.attempt_history],
            "selected_checkpoint": self.trusted_mgr.selected_checkpoint.to_dict() if self.trusted_mgr.selected_checkpoint else None,
            "lookaround_360": {
                "progress_deg": round(float(np.degrees(self.lookaround_accum_rad)), 1),
                "is_complete": (not self.action_in_progress) if (self.active_strategy_enum == RecoveryStrategy.LOOKAROUND_360_SCAN) else False,
                "best_heading_deg": self.lookaround_360_eval.get("best_heading_deg", 0.0),
                "passages_count": len(self.lookaround_360_eval.get("passages", [])),
                "widest_corridor_m": self.lookaround_360_eval.get("widest_corridor_m", 0.0),
            },
        }
        msg.data = json.dumps(data)
        self.state_pub.publish(msg)

    def _publish_diagnostics(self, now_sec: float) -> None:
        stamp_msg = self.get_clock().now().to_msg()
        diag_msg = self.diag_builder.build_diagnostic_array(
            stamp_msg=stamp_msg,
            state=self.fsm.state,
            strategy=self.fsm.active_strategy,
            budget=self.fsm.budget,
            failure_reason=self.fsm.failure_reason,
            dwell_sec=now_sec - self.fsm.state_entry_time,
            selected_checkpoint=self.trusted_mgr.selected_checkpoint,
            checkpoints_count=len(self.trusted_mgr.checkpoints),
        )
        self.diag_pub.publish(diag_msg)

    def _publish_markers(self, now_sec: float) -> None:
        stamp_msg = self.get_clock().now().to_msg()
        markers = self.diag_builder.build_markers(
            stamp_msg=stamp_msg,
            state=self.fsm.state,
            strategy=self.fsm.active_strategy,
            checkpoints=list(self.trusted_mgr.checkpoints),
            selected_checkpoint=self.trusted_mgr.selected_checkpoint,
            backtrack_path=self.current_waypoints if self.action_in_progress else [],
            robot_pose=self.current_pose_map,
        )
        self.marker_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NaviguardRecoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Guarantee zero velocity on shutdown
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
