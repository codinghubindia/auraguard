"""Sensor Synchronization, Rate Monitoring, and Frame Validation Node for NAVIGUARD UGV.

Coordinates timestamp monitors, cross-sensor synchronization analysis,
TF frame verification, and diagnostic publication.
"""

import time
from typing import Dict, List, Optional

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, Imu
import tf2_ros

from naviguard_sensor_sync.data_logger import SensorSyncDataLogger
from naviguard_sensor_sync.frame_validator import FrameValidator
from naviguard_sensor_sync.sensor_validators import (
    validate_camera_info,
    validate_imu,
    validate_odometry,
)
from naviguard_sensor_sync.sync_analyzer import CrossSensorSyncAnalyzer
from naviguard_sensor_sync.timestamp_monitor import SensorTimestampMonitor


class SensorSyncNode(Node):
    """ROS 2 Node monitoring sensor rates, temporal synchronization, and TF frame calibration."""

    def __init__(self) -> None:
        super().__init__('sensor_sync_node')

        # 1. Parameter Declarations
        self.declare_parameter('camera_image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('diagnostics_topic', '/sensor_sync/diagnostics')

        self.declare_parameter('sync_tolerance_ms', 20.0)
        self.declare_parameter('expected_camera_rate', 30.0)
        self.declare_parameter('expected_imu_rate', 100.0)
        self.declare_parameter('expected_odom_rate', 50.0)
        self.declare_parameter('rate_tolerance_pct', 35.0)
        self.declare_parameter('gap_threshold_factor', 2.5)
        self.declare_parameter('window_size', 100)

        self.declare_parameter('csv_logging_enabled', False)
        self.declare_parameter('csv_output_path', '/tmp/sensor_sync_log.csv')
        self.declare_parameter('diag_publish_rate_hz', 2.0)
        self.declare_parameter('report_interval_sec', 5.0)

        # Retrieve parameter values
        camera_image_topic = self.get_parameter('camera_image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        imu_topic = self.get_parameter('imu_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        diagnostics_topic = self.get_parameter('diagnostics_topic').value

        sync_tolerance_ms = float(self.get_parameter('sync_tolerance_ms').value)
        exp_cam_rate = float(self.get_parameter('expected_camera_rate').value)
        exp_imu_rate = float(self.get_parameter('expected_imu_rate').value)
        exp_odom_rate = float(self.get_parameter('expected_odom_rate').value)
        rate_tol_pct = float(self.get_parameter('rate_tolerance_pct').value)
        gap_factor = float(self.get_parameter('gap_threshold_factor').value)
        win_size = int(self.get_parameter('window_size').value)

        csv_enabled = bool(self.get_parameter('csv_logging_enabled').value)
        csv_path = str(self.get_parameter('csv_output_path').value)
        diag_hz = float(self.get_parameter('diag_publish_rate_hz').value)
        self.report_interval_sec = float(self.get_parameter('report_interval_sec').value)

        # 2. Component Initialization
        self.cam_monitor = SensorTimestampMonitor(
            'camera', exp_cam_rate, win_size, gap_factor, rate_tol_pct
        )
        self.cam_info_monitor = SensorTimestampMonitor(
            'camera_info', exp_cam_rate, win_size, gap_factor, rate_tol_pct
        )
        self.imu_monitor = SensorTimestampMonitor(
            'imu', exp_imu_rate, win_size, gap_factor, rate_tol_pct
        )
        self.odom_monitor = SensorTimestampMonitor(
            'odom', exp_odom_rate, win_size, gap_factor, rate_tol_pct
        )

        self.sync_analyzer = CrossSensorSyncAnalyzer(tolerance_ms=sync_tolerance_ms)
        self.data_logger = SensorSyncDataLogger(enabled=csv_enabled, output_path=csv_path)

        # TF components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.frame_validator = FrameValidator(self.tf_buffer)

        # Sensor message caches
        self.latest_image: Optional[Image] = None
        self.latest_info: Optional[CameraInfo] = None
        self.latest_imu: Optional[Imu] = None
        self.latest_odom: Optional[Odometry] = None

        # Timing tracking
        self.node_start_time = time.time()
        self.last_report_time = time.time()

        # 3. Subscriptions (QoS aligned with Gazebo bridge publishers)
        sub_sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        sub_odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Image, camera_image_topic, self._cam_callback, sub_sensor_qos)
        self.create_subscription(CameraInfo, camera_info_topic, self._info_callback, sub_sensor_qos)
        self.create_subscription(Imu, imu_topic, self._imu_callback, sub_sensor_qos)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, sub_odom_qos)

        # 4. Publisher
        self.diag_pub = self.create_publisher(DiagnosticArray, diagnostics_topic, 10)

        # 5. Periodic Timers
        self.diag_timer = self.create_timer(1.0 / diag_hz, self._diagnostics_timer_callback)
        self.report_timer = self.create_timer(self.report_interval_sec, self._report_timer_callback)

        self.get_logger().info(
            f"SensorSyncNode initialized:\n"
            f"  Camera Image:    {camera_image_topic} (Expected: {exp_cam_rate:.1f} Hz)\n"
            f"  Camera Info:     {camera_info_topic} (Expected: {exp_cam_rate:.1f} Hz)\n"
            f"  IMU:             {imu_topic} (Expected: {exp_imu_rate:.1f} Hz)\n"
            f"  Odometry:        {odom_topic} (Expected: {exp_odom_rate:.1f} Hz)\n"
            f"  Sync Tolerance:  {sync_tolerance_ms:.1f} ms\n"
            f"  Diagnostics:     {diagnostics_topic}"
        )

    def _get_current_time_sec(self) -> float:
        """Return current ROS simulation time in seconds."""
        t = self.get_clock().now()
        return float(t.nanoseconds) * 1e-9

    def _cam_callback(self, msg: Image) -> None:
        arrival = self._get_current_time_sec()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_image = msg
        self.cam_monitor.add_sample(stamp, arrival, msg.header.frame_id, 'sensor_msgs/Image')

    def _info_callback(self, msg: CameraInfo) -> None:
        arrival = self._get_current_time_sec()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_info = msg
        self.cam_info_monitor.add_sample(stamp, arrival, msg.header.frame_id, 'sensor_msgs/CameraInfo')

    def _imu_callback(self, msg: Imu) -> None:
        arrival = self._get_current_time_sec()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_imu = msg
        self.imu_monitor.add_sample(stamp, arrival, msg.header.frame_id, 'sensor_msgs/Imu')

    def _odom_callback(self, msg: Odometry) -> None:
        arrival = self._get_current_time_sec()
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        self.latest_odom = msg
        self.odom_monitor.add_sample(stamp, arrival, msg.header.frame_id, 'nav_msgs/Odometry')

    def _diagnostics_timer_callback(self) -> None:
        """Evaluate sensor sync and publish DiagnosticArray message."""
        now_sec = self._get_current_time_sec()

        cam_stats = self.cam_monitor.compute_stats(now_sec)
        info_stats = self.cam_info_monitor.compute_stats(now_sec)
        imu_stats = self.imu_monitor.compute_stats(now_sec)
        odom_stats = self.odom_monitor.compute_stats(now_sec)

        # Cross-sensor sync analysis
        stamps_cam = self.cam_monitor.get_recent_timestamps()
        stamps_info = self.cam_info_monitor.get_recent_timestamps()
        stamps_imu = self.imu_monitor.get_recent_timestamps()
        stamps_odom = self.odom_monitor.get_recent_timestamps()

        sync_cam_imu = self.sync_analyzer.evaluate_pair(stamps_cam, stamps_imu, "camera_imu")
        sync_cam_odom = self.sync_analyzer.evaluate_pair(stamps_cam, stamps_odom, "camera_odom")
        sync_imu_odom = self.sync_analyzer.evaluate_pair(stamps_imu, stamps_odom, "imu_odom")
        sync_cam_info = self.sync_analyzer.evaluate_pair(stamps_cam, stamps_info, "camera_info_sync")

        # Formats and field validations
        cam_calib = (
            validate_camera_info(self.latest_info, self.latest_image)
            if self.latest_info
            else None
        )
        imu_val = validate_imu(self.latest_imu) if self.latest_imu else None
        odom_val = validate_odometry(self.latest_odom) if self.latest_odom else None
        frame_val = self.frame_validator.validate_frames()

        # Overall Status Determination (PASS, WARNING, FAIL)
        overall_status = self._determine_overall_status(
            cam_stats, imu_stats, odom_stats,
            sync_cam_imu, sync_cam_odom,
            cam_calib, imu_val, odom_val, frame_val,
        )

        # Build Diagnostic Message
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()

        # 1. Overall System Status
        ds_overall = DiagnosticStatus()
        ds_overall.name = "naviguard_sensor_sync: Overall System"
        ds_overall.hardware_id = "NAVIGUARD_SENSOR_SUITE"
        ds_overall.level = (
            DiagnosticStatus.OK
            if overall_status == "PASS"
            else (DiagnosticStatus.WARN if overall_status == "WARNING" else DiagnosticStatus.ERROR)
        )
        ds_overall.message = f"Phase 5A Status: {overall_status}"

        ds_overall.values = [
            KeyValue(key="phase_5a_status", value=overall_status),
            KeyValue(key="camera_rate_hz", value=f"{cam_stats.current_rate_hz:.1f}"),
            KeyValue(key="camera_status", value=cam_stats.status),
            KeyValue(key="imu_rate_hz", value=f"{imu_stats.current_rate_hz:.1f}"),
            KeyValue(key="imu_status", value=imu_stats.status),
            KeyValue(key="odom_rate_hz", value=f"{odom_stats.current_rate_hz:.1f}"),
            KeyValue(key="odom_status", value=odom_stats.status),
            KeyValue(key="cam_imu_mean_diff_ms", value=f"{sync_cam_imu.mean_abs_diff_ms:.2f}"),
            KeyValue(key="cam_imu_pct_in_tol", value=f"{sync_cam_imu.pct_within_tolerance:.1f}%"),
            KeyValue(key="cam_odom_mean_diff_ms", value=f"{sync_cam_odom.mean_abs_diff_ms:.2f}"),
            KeyValue(key="cam_odom_pct_in_tol", value=f"{sync_cam_odom.pct_within_tolerance:.1f}%"),
            KeyValue(key="frame_validation_status", value=frame_val.status),
            KeyValue(key="optical_convention_valid", value=str(frame_val.camera_optical_convention_valid).lower()),
            KeyValue(key="camera_calib_status", value=cam_calib.status if cam_calib else "NO_DATA"),
            KeyValue(key="imu_validation_status", value=imu_val.status if imu_val else "NO_DATA"),
            KeyValue(key="odom_validation_status", value=odom_val.status if odom_val else "NO_DATA"),
        ]
        diag_msg.status.append(ds_overall)

        # 2. Timing & Synchronization Status
        ds_sync = DiagnosticStatus()
        ds_sync.name = "naviguard_sensor_sync: Cross-Sensor Synchronization"
        ds_sync.hardware_id = "NAVIGUARD_TIMING"
        ds_sync.level = (
            DiagnosticStatus.OK
            if sync_cam_imu.status == "OK" and sync_cam_odom.status == "OK"
            else DiagnosticStatus.WARN
        )
        ds_sync.message = f"Cam-IMU: {sync_cam_imu.pct_within_tolerance:.1f}%, Cam-Odom: {sync_cam_odom.pct_within_tolerance:.1f}%"
        ds_sync.values = [
            KeyValue(key="cam_period_mean_ms", value=f"{cam_stats.mean_period_sec * 1000.0:.2f}"),
            KeyValue(key="cam_period_std_ms", value=f"{cam_stats.std_period_sec * 1000.0:.2f}"),
            KeyValue(key="imu_period_mean_ms", value=f"{imu_stats.mean_period_sec * 1000.0:.2f}"),
            KeyValue(key="imu_period_std_ms", value=f"{imu_stats.std_period_sec * 1000.0:.2f}"),
            KeyValue(key="odom_period_mean_ms", value=f"{odom_stats.mean_period_sec * 1000.0:.2f}"),
            KeyValue(key="odom_period_std_ms", value=f"{odom_stats.std_period_sec * 1000.0:.2f}"),
            KeyValue(key="cam_imu_median_ms", value=f"{sync_cam_imu.median_abs_diff_ms:.2f}"),
            KeyValue(key="cam_imu_max_ms", value=f"{sync_cam_imu.max_abs_diff_ms:.2f}"),
            KeyValue(key="cam_odom_median_ms", value=f"{sync_cam_odom.median_abs_diff_ms:.2f}"),
            KeyValue(key="cam_odom_max_ms", value=f"{sync_cam_odom.max_abs_diff_ms:.2f}"),
            KeyValue(key="imu_odom_median_ms", value=f"{sync_imu_odom.median_abs_diff_ms:.2f}"),
            KeyValue(key="imu_odom_max_ms", value=f"{sync_imu_odom.max_abs_diff_ms:.2f}"),
        ]
        diag_msg.status.append(ds_sync)

        self.diag_pub.publish(diag_msg)

        # Log to CSV if enabled
        if self.data_logger.enabled:
            log_data = {
                "timestamp_sec": now_sec,
                "camera_rate_hz": cam_stats.current_rate_hz,
                "imu_rate_hz": imu_stats.current_rate_hz,
                "odom_rate_hz": odom_stats.current_rate_hz,
                "camera_mean_period_sec": cam_stats.mean_period_sec,
                "imu_mean_period_sec": imu_stats.mean_period_sec,
                "odom_mean_period_sec": odom_stats.mean_period_sec,
                "cam_imu_median_diff_ms": sync_cam_imu.median_abs_diff_ms,
                "cam_odom_median_diff_ms": sync_cam_odom.median_abs_diff_ms,
                "imu_odom_median_diff_ms": sync_imu_odom.median_abs_diff_ms,
                "cam_imu_pct_within_tol": sync_cam_imu.pct_within_tolerance,
                "cam_odom_pct_within_tol": sync_cam_odom.pct_within_tolerance,
                "imu_odom_pct_within_tol": sync_imu_odom.pct_within_tolerance,
                "frame_status": frame_val.status,
                "camera_calib_status": cam_calib.status if cam_calib else "NO_DATA",
                "imu_status": imu_val.status if imu_val else "NO_DATA",
                "odom_status": odom_val.status if odom_val else "NO_DATA",
                "overall_status": overall_status,
            }
            self.data_logger.log_record(log_data)

    def _determine_overall_status(
        self,
        cam_stats, imu_stats, odom_stats,
        sync_cam_imu, sync_cam_odom,
        cam_calib, imu_val, odom_val, frame_val,
    ) -> str:
        """Classify aggregate health as PASS, WARNING, or FAIL."""
        # Initial warmup check (first 3 seconds)
        uptime = time.time() - self.node_start_time
        if uptime < 3.0:
            return "WARNING"

        # Any critical failures
        has_fatal_sensor = (
            cam_stats.status in ("FAIL", "NO_DATA")
            or imu_stats.status in ("FAIL", "NO_DATA")
            or odom_stats.status in ("FAIL", "NO_DATA")
        )
        has_fatal_calib = (
            (cam_calib and not cam_calib.is_valid)
            or (imu_val and not imu_val.is_valid)
            or (odom_val and not odom_val.is_valid)
        )
        has_fatal_frame = frame_val.status == "FAIL"

        if has_fatal_sensor or has_fatal_calib or has_fatal_frame:
            return "FAIL"

        # Any degraded or warnings
        has_warning = (
            cam_stats.status in ("WARN", "STALE")
            or imu_stats.status in ("WARN", "STALE")
            or odom_stats.status in ("WARN", "STALE")
            or sync_cam_imu.status in ("WARN", "INSUFFICIENT_DATA")
            or sync_cam_odom.status in ("WARN", "INSUFFICIENT_DATA")
            or frame_val.status == "WARN"
            or not frame_val.camera_optical_convention_valid
        )

        if has_warning:
            return "WARNING"

        return "PASS"

    def _report_timer_callback(self) -> None:
        """Output clear, structured terminal report for engineering verification."""
        now_sec = self._get_current_time_sec()

        cam_stats = self.cam_monitor.compute_stats(now_sec)
        imu_stats = self.imu_monitor.compute_stats(now_sec)
        odom_stats = self.odom_monitor.compute_stats(now_sec)

        stamps_cam = self.cam_monitor.get_recent_timestamps()
        stamps_imu = self.imu_monitor.get_recent_timestamps()
        stamps_odom = self.odom_monitor.get_recent_timestamps()

        sync_cam_imu = self.sync_analyzer.evaluate_pair(stamps_cam, stamps_imu, "camera_imu")
        sync_cam_odom = self.sync_analyzer.evaluate_pair(stamps_cam, stamps_odom, "camera_odom")
        sync_imu_odom = self.sync_analyzer.evaluate_pair(stamps_imu, stamps_odom, "imu_odom")

        cam_calib = (
            validate_camera_info(self.latest_info, self.latest_image)
            if self.latest_info
            else None
        )
        imu_val = validate_imu(self.latest_imu) if self.latest_imu else None
        odom_val = validate_odometry(self.latest_odom) if self.latest_odom else None
        frame_val = self.frame_validator.validate_frames()

        overall = self._determine_overall_status(
            cam_stats, imu_stats, odom_stats,
            sync_cam_imu, sync_cam_odom,
            cam_calib, imu_val, odom_val, frame_val,
        )

        calib_str = f"fx={cam_calib.fx:.1f}, fy={cam_calib.fy:.1f}, cx={cam_calib.cx:.1f}, cy={cam_calib.cy:.1f}" if cam_calib else "None"
        imu_orient = imu_val.orientation_status if imu_val else "None"
        opt_conv = "VALID (REP-103)" if frame_val.camera_optical_convention_valid else "INVALID"

        if imu_val:
            imu_str = f"[{imu_orient}] (Angular Vel: {imu_val.angular_vel_norm:.3f} rad/s, Accel: {imu_val.linear_accel_norm:.2f} m/s²)"
        else:
            imu_str = "[NO_DATA]"

        if odom_val:
            odom_str = f"[{odom_val.status}] ('{odom_val.header_frame_id}' -> '{odom_val.child_frame_id}')"
        else:
            odom_str = "[NO_DATA]"

        report_text = (
            f"\n==================== [NAVIGUARD PHASE 5A SENSOR SYNC REPORT] ====================\n"
            f"  OVERALL STATUS: [{overall}]\n"
            f"  1. SENSORS & RATES:\n"
            f"     - Camera (/camera/image_raw):  {cam_stats.current_rate_hz:5.1f} Hz (Period: {cam_stats.mean_period_sec*1000:4.1f}±{cam_stats.std_period_sec*1000:4.1f}ms, Mono: {cam_stats.is_monotonic}, Gaps: {cam_stats.num_gaps})\n"
            f"     - IMU (/imu):                  {imu_stats.current_rate_hz:5.1f} Hz (Period: {imu_stats.mean_period_sec*1000:4.1f}±{imu_stats.std_period_sec*1000:4.1f}ms, Mono: {imu_stats.is_monotonic}, Gaps: {imu_stats.num_gaps})\n"
            f"     - Odometry (/odom):            {odom_stats.current_rate_hz:5.1f} Hz (Period: {odom_stats.mean_period_sec*1000:4.1f}±{odom_stats.std_period_sec*1000:4.1f}ms, Mono: {odom_stats.is_monotonic}, Gaps: {odom_stats.num_gaps})\n"
            f"  2. CROSS-SENSOR SYNCHRONIZATION (Tolerance: {self.sync_analyzer.tolerance_ms:.1f}ms):\n"
            f"     - Camera <-> IMU:     Median={sync_cam_imu.median_abs_diff_ms:4.1f}ms, Mean={sync_cam_imu.mean_abs_diff_ms:4.1f}ms, Max={sync_cam_imu.max_abs_diff_ms:4.1f}ms [{sync_cam_imu.pct_within_tolerance:5.1f}% in tol]\n"
            f"     - Camera <-> Odometry: Median={sync_cam_odom.median_abs_diff_ms:4.1f}ms, Mean={sync_cam_odom.mean_abs_diff_ms:4.1f}ms, Max={sync_cam_odom.max_abs_diff_ms:4.1f}ms [{sync_cam_odom.pct_within_tolerance:5.1f}% in tol]\n"
            f"     - IMU <-> Odometry:    Median={sync_imu_odom.median_abs_diff_ms:4.1f}ms, Mean={sync_imu_odom.mean_abs_diff_ms:4.1f}ms, Max={sync_imu_odom.max_abs_diff_ms:4.1f}ms [{sync_imu_odom.pct_within_tolerance:5.1f}% in tol]\n"
            f"  3. TF & FRAME VALIDATION:\n"
            f"     - Hierarchy Status:    [{frame_val.status}] (All required frames available: {frame_val.all_frames_available})\n"
            f"     - Optical Convention:  [{opt_conv}]\n"
            f"  4. INTRINSIC & FIELD VALIDATIONS:\n"
            f"     - Camera Calibration:  [{cam_calib.status if cam_calib else 'NO_DATA'}] ({calib_str})\n"
            f"     - IMU Validation:      {imu_str}\n"
            f"     - Odometry Validation: {odom_str}\n"
            f"=================================================================================="
        )
        self.get_logger().info(report_text)


def main(args=None) -> None:
    """ROS 2 entry point."""
    rclpy.init(args=args)
    node = SensorSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.data_logger.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
