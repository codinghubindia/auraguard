"""Visual Odometry ROS 2 Node for NAVIGUARD UGV.

Performs Shi-Tomasi feature tracking with Lucas-Kanade optical flow,
bidirectional forward-backward validation, robust MAD outlier rejection,
Essential matrix relative pose recovery (R, unit-scale t), and publishes
diagnostic telemetry alongside annotated debug visualizations.

CRITICAL MONOCULAR SCALE NOTICE:
The camera is monocular. Translation recovered from recoverPose is strictly
represented as a unit-scale translation direction (||t|| = 1.0).
translation_scale_status is strictly UNKNOWN. Metric odometry (/visual_odom)
is NEVER fabricated.
"""

import time
from typing import Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
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
from sensor_msgs.msg import CameraInfo, Image

from naviguard_visual_odometry.feature_tracker import (
    FeatureTracker,
    FeatureTrackerConfig,
)
from naviguard_visual_odometry.geometric_estimator import (
    GeometricEstimatorConfig,
    GeometricMotionEstimate,
    GeometricMotionEstimator,
)


class VisualOdometryNode(Node):
    """ROS 2 Node executing temporal visual tracking and geometric relative motion estimation."""

    def __init__(self) -> None:
        super().__init__('visual_odometry_node')

        # 1. Parameter Declarations
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('debug_image_topic', '/visual_odometry/debug_image')
        self.declare_parameter('telemetry_topic', '/visual_odometry/telemetry')

        # Tracker configuration parameters
        self.declare_parameter('max_features', 200)
        self.declare_parameter('quality_level', 0.01)
        self.declare_parameter('min_distance', 12.0)
        self.declare_parameter('block_size', 3)
        self.declare_parameter('use_harris', False)
        self.declare_parameter('lk_win_size', 21)
        self.declare_parameter('lk_max_level', 3)
        self.declare_parameter('lk_max_iters', 30)
        self.declare_parameter('lk_eps', 0.01)
        self.declare_parameter('fb_err_threshold', 1.0)
        self.declare_parameter('min_features_threshold', 70)
        self.declare_parameter('outlier_mad_k', 3.5)
        self.declare_parameter('max_displacement_px', 80.0)
        self.declare_parameter('self_mask_height_ratio', 0.15)

        # Geometric motion estimator parameters
        self.declare_parameter('geom_ransac_threshold', 1.0)
        self.declare_parameter('geom_ransac_prob', 0.999)
        self.declare_parameter('geom_min_inliers', 15)
        self.declare_parameter('geom_min_baseline_disp_px', 0.6)

        self.declare_parameter('diag_period_sec', 3.0)
        self.declare_parameter('panorama_topic', '/camera/panorama_image')
        self.declare_parameter('panorama_debug_topic', '/visual_odometry/panorama_debug')

        # Retrieve parameter values
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        telemetry_topic = self.get_parameter('telemetry_topic').get_parameter_value().string_value
        panorama_topic = self.get_parameter('panorama_topic').get_parameter_value().string_value
        panorama_debug_topic = self.get_parameter('panorama_debug_topic').get_parameter_value().string_value

        lk_win = self.get_parameter('lk_win_size').get_parameter_value().integer_value
        self.diag_period_sec = self.get_parameter('diag_period_sec').get_parameter_value().double_value

        tracker_config = FeatureTrackerConfig(
            max_features=self.get_parameter('max_features').get_parameter_value().integer_value,
            quality_level=self.get_parameter('quality_level').get_parameter_value().double_value,
            min_distance=self.get_parameter('min_distance').get_parameter_value().double_value,
            block_size=self.get_parameter('block_size').get_parameter_value().integer_value,
            use_harris=self.get_parameter('use_harris').get_parameter_value().bool_value,
            lk_win_size=(lk_win, lk_win),
            lk_max_level=self.get_parameter('lk_max_level').get_parameter_value().integer_value,
            lk_max_iters=self.get_parameter('lk_max_iters').get_parameter_value().integer_value,
            lk_eps=self.get_parameter('lk_eps').get_parameter_value().double_value,
            fb_err_threshold=self.get_parameter('fb_err_threshold').get_parameter_value().double_value,
            min_features_threshold=self.get_parameter('min_features_threshold').get_parameter_value().integer_value,
            outlier_mad_k=self.get_parameter('outlier_mad_k').get_parameter_value().double_value,
            max_displacement_px=self.get_parameter('max_displacement_px').get_parameter_value().double_value,
            self_mask_height_ratio=self.get_parameter('self_mask_height_ratio').get_parameter_value().double_value,
        )

        geom_config = GeometricEstimatorConfig(
            ransac_threshold=self.get_parameter('geom_ransac_threshold').get_parameter_value().double_value,
            ransac_prob=self.get_parameter('geom_ransac_prob').get_parameter_value().double_value,
            min_inliers=self.get_parameter('geom_min_inliers').get_parameter_value().integer_value,
            min_baseline_disp_px=self.get_parameter('geom_min_baseline_disp_px').get_parameter_value().double_value,
        )

        # 2. Components Initialization
        self.tracker = FeatureTracker(tracker_config)
        self.surround_tracker = FeatureTracker(tracker_config)
        self.surround_tracks_count = 0
        self.surround_inliers_count = 0
        self.surround_status = "AWAITING_PANORAMA"
        self.geom_estimator = GeometricMotionEstimator(geom_config)
        self.bridge = CvBridge()

        # Camera intrinsics storage
        self.camera_info: Optional[CameraInfo] = None
        self.K_matrix: Optional[np.ndarray] = None
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Reference odometry storage
        self.latest_odom_vx: Optional[float] = None
        self.latest_odom_wz: Optional[float] = None
        self.latest_odom_time: Optional[float] = None

        # Latest geometric estimate
        self.latest_geom_estimate = GeometricMotionEstimate(is_valid=False, status="INITIALIZING")

        # Processing & timing state
        self.frame_idx = 0
        self.prev_stamp_sec: Optional[float] = None
        self.prev_wall_time: Optional[float] = None
        self.input_fps = 0.0
        self.proc_fps = 0.0
        self.proc_latency_ms = 0.0
        self.last_diag_log_time = time.time()

        # 3. ROS Subscriptions & Publishers
        sub_sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            sub_sensor_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            sub_sensor_qos,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            odom_qos,
        )

        self.panorama_sub = self.create_subscription(
            Image,
            panorama_topic,
            self.panorama_callback,
            sub_sensor_qos,
        )

        self.debug_image_pub = self.create_publisher(
            Image,
            debug_image_topic,
            10,
        )

        self.panorama_debug_pub = self.create_publisher(
            Image,
            panorama_debug_topic,
            10,
        )

        self.telemetry_pub = self.create_publisher(
            DiagnosticArray,
            telemetry_topic,
            10,
        )

        self.get_logger().info(
            f"VisualOdometryNode initialized:\n"
            f"  Subscribing image:       {image_topic} (Best-Effort)\n"
            f"  Subscribing camera_info: {camera_info_topic} (Best-Effort)\n"
            f"  Subscribing odom:        {odom_topic} (Reliable)\n"
            f"  Subscribing panorama:    {panorama_topic} (360-deg Surround)\n"
            f"  Publishing debug image:  {debug_image_topic}\n"
            f"  Publishing panorama dbg: {panorama_debug_topic}\n"
            f"  Publishing telemetry:    {telemetry_topic}\n"
            f"  Max features:            {tracker_config.max_features}\n"
            f"  FB error threshold:      {tracker_config.fb_err_threshold} px\n"
            f"  Geom RANSAC thresh:      {geom_config.ransac_threshold} px\n"
            f"  Scale Mode:              STRICT MONOCULAR UNIT-SCALE (UNKNOWN)"
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics and build 3x3 camera calibration matrix K."""
        if self.camera_info is None:
            self.camera_info = msg
            if len(msg.k) >= 9:
                self.fx = float(msg.k[0])
                self.cx = float(msg.k[2])
                self.fy = float(msg.k[4])
                self.cy = float(msg.k[5])
                self.K_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
                self.get_logger().info(
                    f"Camera intrinsics received: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                    f"cx={self.cx:.2f}, cy={self.cy:.2f} [{msg.width}x{msg.height}]"
                )

    def odom_callback(self, msg: Odometry) -> None:
        """Cache reference wheel odometry for cross-validation."""
        self.latest_odom_vx = float(msg.twist.twist.linear.x)
        self.latest_odom_wz = float(msg.twist.twist.angular.z)
        self.latest_odom_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def image_callback(self, msg: Image) -> None:
        """Process incoming camera frame: optical flow, geometric pose estimation, HUD & telemetry."""
        wall_start = time.perf_counter()
        now_wall = time.time()
        self.frame_idx += 1

        # Calculate simulation time delta
        curr_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if self.prev_stamp_sec is not None:
            dt_sim = curr_stamp_sec - self.prev_stamp_sec
        else:
            dt_sim = 0.0
        self.prev_stamp_sec = curr_stamp_sec

        # Calculate input FPS based on wall interval
        if self.prev_wall_time is not None:
            dt_wall = now_wall - self.prev_wall_time
            if dt_wall > 0.0:
                current_fps = 1.0 / dt_wall
                self.input_fps = 0.9 * self.input_fps + 0.1 * current_fps if self.input_fps > 0.0 else current_fps
        self.prev_wall_time = now_wall

        # Convert ROS Image to OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as err:
            self.get_logger().error(f"CvBridge conversion error: {err}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 1. Track features and compute pixel-space motion
        tracks, stats = self.tracker.track(gray)

        # 2. Geometric Motion Estimation (Essential Matrix -> R, unit-t)
        inlier_tracks = [t for t in tracks if t.is_inlier]
        if len(inlier_tracks) > 0:
            pts_prev = np.array([[t.prev_x, t.prev_y] for t in inlier_tracks], dtype=np.float32)
            pts_curr = np.array([[t.curr_x, t.curr_y] for t in inlier_tracks], dtype=np.float32)
        else:
            pts_prev = np.empty((0, 2), dtype=np.float32)
            pts_curr = np.empty((0, 2), dtype=np.float32)

        if self.K_matrix is not None:
            geom_estimate = self.geom_estimator.estimate(
                pts_prev,
                pts_curr,
                self.K_matrix,
                median_displacement_px=stats.get('median_displacement_px', 0.0),
            )
        else:
            geom_estimate = GeometricMotionEstimate(is_valid=False, status="WAITING_CALIB")

        self.latest_geom_estimate = geom_estimate

        # Performance latency tracking
        wall_end = time.perf_counter()
        self.proc_latency_ms = (wall_end - wall_start) * 1000.0
        if self.proc_latency_ms > 0.0:
            inst_proc_fps = 1000.0 / self.proc_latency_ms
            self.proc_fps = 0.9 * self.proc_fps + 0.1 * inst_proc_fps if self.proc_fps > 0.0 else inst_proc_fps

        # Build metadata for visualization
        metadata = {
            'frame_idx': self.frame_idx,
            'stamp_sec': msg.header.stamp.sec,
            'stamp_nanosec': msg.header.stamp.nanosec,
            'frame_id': msg.header.frame_id,
            'input_fps': self.input_fps,
            'proc_latency_ms': self.proc_latency_ms,
            'proc_fps': self.proc_fps,
            'wheel_vx': self.latest_odom_vx,
            'wheel_wz': self.latest_odom_wz,
            'dt_sim': dt_sim,
            # Geometric Motion Metadata
            'geom_is_valid': geom_estimate.is_valid,
            'geom_status': geom_estimate.status,
            'geom_num_inliers': geom_estimate.num_inliers,
            'unit_tx': geom_estimate.unit_tx,
            'unit_ty': geom_estimate.unit_ty,
            'unit_tz': geom_estimate.unit_tz,
            'rot_yaw_deg': geom_estimate.yaw_deg,
            'rot_pitch_deg': geom_estimate.pitch_deg,
            'rot_roll_deg': geom_estimate.roll_deg,
            'rot_angle_deg': geom_estimate.rot_angle_deg,
            'translation_scale_status': geom_estimate.translation_scale_status,
        }

        # Render HUD Canvas and Publish
        debug_canvas = self.tracker.render_debug_canvas(cv_image, tracks, stats, metadata)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_canvas, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

        # Publish Diagnostic Telemetry Array
        self._publish_telemetry(msg.header, dt_sim, stats, geom_estimate)

        # Periodic Diagnostic Console Logging
        if (now_wall - self.last_diag_log_time) >= self.diag_period_sec:
            self.last_diag_log_time = now_wall
            odom_str = (
                f"ref_odom vx={self.latest_odom_vx:+.2f}m/s wz={self.latest_odom_wz:+.2f}r/s"
                if self.latest_odom_vx is not None
                else "ref_odom=none"
            )
            geom_str = (
                f"Geom: {geom_estimate.status} ({geom_estimate.num_inliers} inl) | "
                f"yaw={geom_estimate.yaw_deg:+.1f}deg | "
                f"t=[{geom_estimate.unit_tx:+.2f}, {geom_estimate.unit_ty:+.2f}, {geom_estimate.unit_tz:+.2f}] (scale: UNKNOWN)"
            )
            self.get_logger().info(
                f"[VO Frame #{self.frame_idx:04d}] "
                f"Tracks: {stats['num_inliers']:3d}/{stats['num_detected']:3d} | "
                f"flow dx={stats['median_dx_px']:+5.2f} dy={stats['median_dy_px']:+5.2f} px | "
                f"{self.input_fps:.1f} FPS ({self.proc_latency_ms:.1f}ms) | "
                f"{odom_str} | {geom_str}"
            )

    def _publish_telemetry(
        self,
        header,
        dt_sim: float,
        stats: dict,
        geom: GeometricMotionEstimate,
    ) -> None:
        """Construct and publish a DiagnosticArray message with tracking and geometric motion statistics."""
        diag_msg = DiagnosticArray()
        diag_msg.header = header

        diag_status = DiagnosticStatus()
        diag_status.name = 'naviguard_visual_odometry: Motion Tracking'
        diag_status.hardware_id = 'NAVIGUARD_CAMERA_VO'

        if stats['num_inliers'] >= 20 and (geom.is_valid or geom.status == "DEGENERATE_SMALL_BASELINE"):
            diag_status.level = DiagnosticStatus.OK
            diag_status.message = f"{stats['status']} | {geom.status}"
        elif stats['num_inliers'] > 0:
            diag_status.level = DiagnosticStatus.WARN
            diag_status.message = f"{stats['status']} | {geom.status}"
        else:
            diag_status.level = DiagnosticStatus.ERROR
            diag_status.message = "Tracking lost: 0 inliers"

        diag_status.values = [
            KeyValue(key='frame_index', value=str(self.frame_idx)),
            KeyValue(key='dt_sec', value=f"{dt_sim:.4f}"),
            KeyValue(key='num_detected', value=str(stats['num_detected'])),
            KeyValue(key='num_tracked', value=str(stats['num_tracked'])),
            KeyValue(key='num_inliers', value=str(stats['num_inliers'])),
            KeyValue(key='tracking_ratio', value=f"{stats['tracking_ratio']:.3f}"),
            KeyValue(key='median_dx_px', value=f"{stats['median_dx_px']:.3f}"),
            KeyValue(key='median_dy_px', value=f"{stats['median_dy_px']:.3f}"),
            KeyValue(key='mean_dx_px', value=f"{stats['mean_dx_px']:.3f}"),
            KeyValue(key='mean_dy_px', value=f"{stats['mean_dy_px']:.3f}"),
            KeyValue(key='median_displacement_px', value=f"{stats['median_displacement_px']:.3f}"),
            KeyValue(key='mean_displacement_px', value=f"{stats['mean_displacement_px']:.3f}"),
            KeyValue(key='disp_std_px', value=f"{stats['disp_std_px']:.3f}"),
            # Geometric Motion Estimates
            KeyValue(key='geom_is_valid', value=str(geom.is_valid).lower()),
            KeyValue(key='geom_status', value=geom.status),
            KeyValue(key='geom_num_inliers', value=str(geom.num_inliers)),
            KeyValue(key='geom_inlier_ratio', value=f"{geom.inlier_ratio:.3f}"),
            KeyValue(key='unit_tx', value=f"{geom.unit_tx:+.4f}"),
            KeyValue(key='unit_ty', value=f"{geom.unit_ty:+.4f}"),
            KeyValue(key='unit_tz', value=f"{geom.unit_tz:+.4f}"),
            KeyValue(key='translation_scale_status', value=geom.translation_scale_status),
            KeyValue(key='rot_angle_deg', value=f"{geom.rot_angle_deg:+.3f}"),
            KeyValue(key='rot_yaw_deg', value=f"{geom.yaw_deg:+.3f}"),
            KeyValue(key='rot_pitch_deg', value=f"{geom.pitch_deg:+.3f}"),
            KeyValue(key='rot_roll_deg', value=f"{geom.roll_deg:+.3f}"),
            KeyValue(key='rot_axis_x', value=f"{geom.rot_axis[0]:+.3f}"),
            KeyValue(key='rot_axis_y', value=f"{geom.rot_axis[1]:+.3f}"),
            KeyValue(key='rot_axis_z', value=f"{geom.rot_axis[2]:+.3f}"),
            # Performance & Odometry Reference
            KeyValue(key='input_fps', value=f"{self.input_fps:.1f}"),
            KeyValue(key='proc_latency_ms', value=f"{self.proc_latency_ms:.2f}"),
            KeyValue(key='proc_fps', value=f"{self.proc_fps:.1f}"),
            KeyValue(
                key='ref_odom_vx',
                value=f"{self.latest_odom_vx:.3f}" if self.latest_odom_vx is not None else "nan",
            ),
            KeyValue(
                key='ref_odom_wz',
                value=f"{self.latest_odom_wz:.3f}" if self.latest_odom_wz is not None else "nan",
            ),
            KeyValue(key='camera_fx', value=f"{self.fx:.2f}" if self.fx is not None else "nan"),
            KeyValue(key='camera_fy', value=f"{self.fy:.2f}" if self.fy is not None else "nan"),
            # 360-Degree Surround Panoramic Visual Odometry
            KeyValue(key='surround_tracks', value=str(self.surround_tracks_count)),
            KeyValue(key='surround_inliers', value=str(self.surround_inliers_count)),
            KeyValue(key='surround_status', value=self.surround_status),
        ]

        diag_msg.status.append(diag_status)
        self.telemetry_pub.publish(diag_msg)

    def panorama_callback(self, msg: Image) -> None:
        """Process 360-degree panoramic camera frame: surround tracking & wide-angle optical flow."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError:
            return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        tracks, stats = self.surround_tracker.track(gray)
        inliers = [t for t in tracks if t.is_inlier]
        self.surround_tracks_count = len(tracks)
        self.surround_inliers_count = len(inliers)
        self.surround_status = "TRACKING_360" if len(inliers) >= 15 else "SURROUND_SEARCH"

        meta = {
            'frame_idx': self.frame_idx,
            'stamp_sec': msg.header.stamp.sec,
            'stamp_nanosec': msg.header.stamp.nanosec,
            'frame_id': 'camera_360_link',
            'input_fps': self.input_fps,
            'proc_latency_ms': self.proc_latency_ms,
            'proc_fps': self.proc_fps,
            'wheel_vx': self.latest_odom_vx,
            'wheel_wz': self.latest_odom_wz,
            'geom_status': f"360_SURROUND_{self.surround_status}",
            'geom_num_inliers': len(inliers),
        }
        debug_canvas = self.surround_tracker.render_debug_canvas(cv_img, tracks, stats, meta)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_canvas, encoding='bgr8')
        debug_msg.header = msg.header
        self.panorama_debug_pub.publish(debug_msg)


def main(args=None) -> None:
    """ROS 2 entry point."""
    rclpy.init(args=args)
    node = VisualOdometryNode()
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
