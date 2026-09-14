"""NAVIGUARD Confidence and Autonomous Decision ROS 2 Node.

Subscribes to all telemetry, diagnostics, and sensor streams across the
NAVIGUARD UGV architecture. Calculates multi-dimensional confidence and publishes
actionable decisions (CONTINUE, VERIFY, RECOVER) and RViz overlays.
"""

import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from naviguard_confidence.confidence_dimensions import (
    ConfidenceScores,
    DecisionResult,
    DecisionState,
)
from naviguard_confidence.decision_state_machine import (
    DecisionStateMachine,
    StateMachineConfig,
)
from naviguard_confidence.event_logger import ConfidenceEventLogger
from naviguard_confidence.sensor_monitors import (
    CompositeConfidenceEngine,
    ConfidenceEngineWeights,
)


class NaviguardConfidenceNode(Node):
    """Central Confidence and Decision Layer Node for NAVIGUARD."""

    def __init__(self) -> None:
        super().__init__('confidence_node')

        # 1. Parameter Declarations
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('log_csv_path', '')
        self.declare_parameter('log_json_path', '')

        # FSM Parameters
        self.declare_parameter('continue_threshold', 0.75)
        self.declare_parameter('verify_threshold', 0.65)
        self.declare_parameter('recover_threshold', 0.35)
        self.declare_parameter('recover_to_verify_threshold', 0.50)
        self.declare_parameter('verify_debounce_sec', 0.30)
        self.declare_parameter('recover_debounce_sec', 0.80)
        self.declare_parameter('recover_clear_sec', 1.50)
        self.declare_parameter('continue_clear_sec', 1.00)

        # Retrieve parameters
        rate_hz = float(self.get_parameter('rate_hz').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        csv_path = str(self.get_parameter('log_csv_path').value)
        json_path = str(self.get_parameter('log_json_path').value)

        fsm_cfg = StateMachineConfig(
            continue_threshold=float(self.get_parameter('continue_threshold').value),
            verify_threshold=float(self.get_parameter('verify_threshold').value),
            recover_threshold=float(self.get_parameter('recover_threshold').value),
            recover_to_verify_threshold=float(self.get_parameter('recover_to_verify_threshold').value),
            verify_debounce_sec=float(self.get_parameter('verify_debounce_sec').value),
            recover_debounce_sec=float(self.get_parameter('recover_debounce_sec').value),
            recover_clear_sec=float(self.get_parameter('recover_clear_sec').value),
            continue_clear_sec=float(self.get_parameter('continue_clear_sec').value),
        )

        # 2. Components
        self.engine = CompositeConfidenceEngine(ConfidenceEngineWeights())
        self.fsm = DecisionStateMachine(fsm_cfg)
        self.event_logger = ConfidenceEventLogger(csv_path, json_path)

        # 3. Cached Sensor / Telemetry State
        self.vo_dict: Optional[Dict[str, str]] = None
        self.vo_last_stamp: float = 0.0

        self.slam_dict: Optional[Dict[str, str]] = None
        self.slam_last_stamp: float = 0.0

        self.imu_accel: Tuple[float, float, float] = (0.0, 0.0, 9.81)
        self.imu_gyro: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.imu_last_stamp: float = 0.0

        self.wheel_twist: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (vx, vy, wz)
        self.wheel_last_stamp: float = 0.0

        self.sync_dict: Optional[Dict[str, str]] = None
        self.sync_last_stamp: float = 0.0

        self.state_est_dict: Optional[Dict[str, str]] = None
        self.state_est_last_stamp: float = 0.0

        self.map_data: Optional[List[int]] = None
        self.map_res: float = 0.05
        self.map_w: int = 0
        self.map_h: int = 0
        self.map_ox: float = 0.0
        self.map_oy: float = 0.0
        self.map_last_stamp: float = 0.0

        self.robot_pose_map: Tuple[float, float] = (0.0, 0.0)

        # Latest decision result
        self.latest_decision: DecisionResult = DecisionResult()

        # 4. Publishers
        self.decision_pub = self.create_publisher(String, '/naviguard/decision', 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, '/naviguard/diagnostics', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/naviguard/decision_marker', 10)

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

        # VO telemetry
        self.vo_sub = self.create_subscription(
            DiagnosticArray,
            '/visual_odometry/telemetry',
            self.vo_telemetry_cb,
            reliable_qos,
        )

        # SLAM diagnostics
        self.slam_sub = self.create_subscription(
            DiagnosticArray,
            '/slam/diagnostics',
            self.slam_diagnostics_cb,
            reliable_qos,
        )

        # Sensor sync diagnostics
        self.sync_sub = self.create_subscription(
            DiagnosticArray,
            '/sensor_sync/diagnostics',
            self.sync_diagnostics_cb,
            reliable_qos,
        )

        # State estimation diagnostics
        self.state_est_sub = self.create_subscription(
            DiagnosticArray,
            '/state_estimation/diagnostics',
            self.state_est_diagnostics_cb,
            reliable_qos,
        )

        # IMU stream
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            qos_profile_sensor_data,
        )

        # Wheel odometry stream
        self.wheel_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.wheel_cb,
            reliable_qos,
        )

        # Occupancy grid
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/slam/map',
            self.map_cb,
            transient_local_qos,
        )

        # SLAM pose
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/slam/pose',
            self.pose_cb,
            reliable_qos,
        )

        # 6. Evaluation Timer
        self.timer = self.create_timer(1.0 / rate_hz, self.evaluation_loop)

        self.get_logger().info(
            f"NaviguardConfidenceNode initialized at {rate_hz:.1f} Hz. "
            f"Thresholds: CONTINUE>={fsm_cfg.continue_threshold}, "
            f"VERIFY<{fsm_cfg.verify_threshold}, RECOVER<{fsm_cfg.recover_threshold}"
        )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _msg_stamp_to_sec(self, header) -> float:
        return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9

    def vo_telemetry_cb(self, msg: DiagnosticArray) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        data: Dict[str, str] = {}
        for status in msg.status:
            for kv in status.values:
                data[kv.key] = kv.value
        self.vo_dict = data
        self.vo_last_stamp = stamp if stamp > 0.0 else time.time()

    def slam_diagnostics_cb(self, msg: DiagnosticArray) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        data: Dict[str, str] = {}
        for status in msg.status:
            for kv in status.values:
                data[kv.key] = kv.value
        self.slam_dict = data
        self.slam_last_stamp = stamp if stamp > 0.0 else time.time()

    def sync_diagnostics_cb(self, msg: DiagnosticArray) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        data: Dict[str, str] = {}
        for status in msg.status:
            for kv in status.values:
                data[kv.key] = kv.value
        self.sync_dict = data
        self.sync_last_stamp = stamp if stamp > 0.0 else time.time()

    def state_est_diagnostics_cb(self, msg: DiagnosticArray) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        data: Dict[str, str] = {}
        for status in msg.status:
            for kv in status.values:
                data[kv.key] = kv.value
        self.state_est_dict = data
        self.state_est_last_stamp = stamp if stamp > 0.0 else time.time()

    def imu_cb(self, msg: Imu) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        self.imu_accel = (
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        )
        self.imu_gyro = (
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        )
        self.imu_last_stamp = stamp if stamp > 0.0 else time.time()

    def wheel_cb(self, msg: Odometry) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        self.wheel_twist = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.angular.z),
        )
        self.wheel_last_stamp = stamp if stamp > 0.0 else time.time()

    def map_cb(self, msg: OccupancyGrid) -> None:
        stamp = self._msg_stamp_to_sec(msg.header)
        self.map_data = list(msg.data)
        self.map_res = float(msg.info.resolution)
        self.map_w = int(msg.info.width)
        self.map_h = int(msg.info.height)
        self.map_ox = float(msg.info.origin.position.x)
        self.map_oy = float(msg.info.origin.position.y)
        self.map_last_stamp = stamp if stamp > 0.0 else time.time()

    def pose_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.robot_pose_map = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )

    # -------------------------------------------------------------------------
    # Evaluation Loop
    # -------------------------------------------------------------------------

    def evaluation_loop(self) -> None:
        """Run confidence calculation and state machine evaluation."""
        now = self.get_clock().now()
        now_sec = float(now.nanoseconds) * 1e-9
        if now_sec <= 0.0:
            now_sec = time.time()

        # Step 1: Compute composite multi-dimensional confidence
        scores, primary_reason, secondary_reasons = self.engine.compute(
            current_time=now_sec,
            vo_telemetry=self.vo_dict,
            vo_last_time=self.vo_last_stamp,
            slam_diag=self.slam_dict,
            slam_last_time=self.slam_last_stamp,
            imu_accel=self.imu_accel,
            imu_gyro=self.imu_gyro,
            imu_last_time=self.imu_last_stamp,
            wheel_twist=self.wheel_twist,
            wheel_last_time=self.wheel_last_stamp,
            sync_diag=self.sync_dict,
            sync_last_time=self.sync_last_stamp,
            state_diag=self.state_est_dict,
            state_last_time=self.state_est_last_stamp,
            map_data=self.map_data,
            map_res=self.map_res,
            map_w=self.map_w,
            map_h=self.map_h,
            map_ox=self.map_ox,
            map_oy=self.map_oy,
            robot_pose=self.robot_pose_map,
            map_last_time=self.map_last_stamp,
        )

        # Step 2: Step finite state machine
        decision = self.fsm.update(
            scores=scores,
            primary_reason=primary_reason,
            secondary_reasons=secondary_reasons,
            current_time=now_sec,
        )
        self.latest_decision = decision

        # Step 3: Log step and transition
        self.event_logger.log_step(decision)
        if decision.transition_occurred and decision.previous_state is not None:
            self.get_logger().warn(
                f"[NAVIGUARD TRANSITION] {decision.previous_state.to_string()} -> "
                f"{decision.state.to_string()} | Reason: {decision.primary_reason} "
                f"| Conf: {decision.scores.overall:.2f}"
            )

        # Step 4: Publish ROS outputs
        self._publish_decision_string(decision, now)
        self._publish_diagnostics(decision, now)
        self._publish_markers(decision, now)

    def _publish_decision_string(self, decision: DecisionResult, now) -> None:
        msg = String()
        msg.data = json.dumps(decision.to_dict())
        self.decision_pub.publish(msg)

    def _publish_diagnostics(self, decision: DecisionResult, now) -> None:
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = now.to_msg()
        diag_msg.header.frame_id = self.base_frame

        # Core Decision Status
        ds = DiagnosticStatus()
        ds.name = "naviguard_confidence: decision"
        ds.hardware_id = "NAVIGUARD_DECISION_ENGINE"
        if decision.state == DecisionState.CONTINUE:
            ds.level = DiagnosticStatus.OK
            ds.message = "Autonomous navigation permitted"
        elif decision.state == DecisionState.VERIFY:
            ds.level = DiagnosticStatus.WARN
            ds.message = f"Verify state: {decision.primary_reason}"
        else:
            ds.level = DiagnosticStatus.ERROR
            ds.message = f"Recover state: {decision.primary_reason}"

        ds.values = [
            KeyValue(key="state", value=decision.state.to_string()),
            KeyValue(key="state_code", value=str(int(decision.state))),
            KeyValue(key="overall_confidence", value=f"{decision.scores.overall:.3f}"),
            KeyValue(key="primary_reason", value=decision.primary_reason),
            KeyValue(key="secondary_reasons", value=",".join(decision.secondary_reasons) if decision.secondary_reasons else "NONE"),
            KeyValue(key="dwell_time_sec", value=f"{decision.dwell_time_sec:.2f}"),
            KeyValue(key="visual_confidence", value=f"{decision.scores.visual:.3f}"),
            KeyValue(key="localization_confidence", value=f"{decision.scores.localization:.3f}"),
            KeyValue(key="imu_confidence", value=f"{decision.scores.imu:.3f}"),
            KeyValue(key="wheel_confidence", value=f"{decision.scores.wheel:.3f}"),
            KeyValue(key="temporal_confidence", value=f"{decision.scores.temporal:.3f}"),
            KeyValue(key="cross_sensor_confidence", value=f"{decision.scores.cross_sensor:.3f}"),
            KeyValue(key="map_confidence", value=f"{decision.scores.map:.3f}"),
        ]
        diag_msg.status.append(ds)
        self.diagnostics_pub.publish(diag_msg)

    def _publish_markers(self, decision: DecisionResult, now) -> None:
        marker_arr = MarkerArray()
        stamp_msg = now.to_msg()

        # Marker 0: 3D Status Halo Cylinder around base_link
        m_halo = Marker()
        m_halo.header.stamp = stamp_msg
        m_halo.header.frame_id = self.base_frame
        m_halo.ns = "confidence_halo"
        m_halo.id = 0
        m_halo.type = Marker.CYLINDER
        m_halo.action = Marker.ADD
        m_halo.pose.position.x = 0.0
        m_halo.pose.position.y = 0.0
        m_halo.pose.position.z = -0.05
        m_halo.pose.orientation.w = 1.0
        m_halo.scale.x = 0.70
        m_halo.scale.y = 0.70
        m_halo.scale.z = 0.04

        # Marker 1: Status Banner Text above robot
        m_text = Marker()
        m_text.header.stamp = stamp_msg
        m_text.header.frame_id = self.base_frame
        m_text.ns = "confidence_text"
        m_text.id = 1
        m_text.type = Marker.TEXT_VIEW_FACING
        m_text.action = Marker.ADD
        m_text.pose.position.x = 0.0
        m_text.pose.position.y = 0.0
        m_text.pose.position.z = 0.55
        m_text.pose.orientation.w = 1.0
        m_text.scale.z = 0.12  # Text height

        if decision.state == DecisionState.CONTINUE:
            # Green
            m_halo.color.r = 0.0
            m_halo.color.g = 0.90
            m_halo.color.b = 0.20
            m_halo.color.a = 0.65

            m_text.color.r = 0.10
            m_text.color.g = 1.00
            m_text.color.b = 0.20
            m_text.color.a = 0.95
            m_text.text = f"CONTINUE [{decision.scores.overall:.2f}]"

        elif decision.state == DecisionState.VERIFY:
            # Amber / Yellow
            m_halo.color.r = 1.00
            m_halo.color.g = 0.80
            m_halo.color.b = 0.00
            m_halo.color.a = 0.75

            m_text.color.r = 1.00
            m_text.color.g = 0.85
            m_text.color.b = 0.00
            m_text.color.a = 0.95
            m_text.text = f"VERIFY [{decision.scores.overall:.2f}]\n{decision.primary_reason}"

        else:  # RECOVER
            # Red
            m_halo.color.r = 0.95
            m_halo.color.g = 0.10
            m_halo.color.b = 0.10
            m_halo.color.a = 0.85

            m_text.color.r = 1.00
            m_text.color.g = 0.15
            m_text.color.b = 0.15
            m_text.color.a = 0.98
            m_text.text = f"RECOVER [{decision.scores.overall:.2f}]\n{decision.primary_reason}"

        marker_arr.markers.extend([m_halo, m_text])
        self.marker_pub.publish(marker_arr)


def main(args=None) -> None:
    """Entry point for naviguard_confidence node."""
    rclpy.init(args=args)
    node = NaviguardConfidenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.event_logger.save_json_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
