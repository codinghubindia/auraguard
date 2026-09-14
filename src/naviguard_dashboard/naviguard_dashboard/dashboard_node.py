"""
NAVIGUARD Operator Dashboard ROS 2 Node.

Subscribes to all operational autonomy streams, bridges live telemetry to the
state cache, provides bounded latest-frame image buffering, and exposes HTTP REST endpoints
on port 8080 for operator control. Strictly NEVER publishes to /cmd_vel.
"""

import os
import json
import math
import time
import threading
from typing import Optional, List, Tuple
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray
from std_msgs.msg import String
from std_srvs.srv import Trigger
from diagnostic_msgs.msg import DiagnosticArray
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory
from http.server import ThreadingHTTPServer

from naviguard_dashboard.state_cache import StateCache
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
from naviguard_dashboard.goal_validator import GoalValidator
from naviguard_dashboard.web_server import NaviguardRequestHandler


def quaternion_to_yaw(q) -> float:
    """Extract Euler yaw rotation from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(math.atan2(siny_cosp, cosy_cosp))


class NaviguardDashboardNode(Node):
    """Bridges ROS 2 topics to the Operator Web Dashboard with bounded latest-frame buffers."""

    def __init__(self):
        super().__init__('naviguard_dashboard_node')

        self.declare_parameter('port', 8080)
        self.declare_parameter('host', '0.0.0.0')
        self.port = int(self.get_parameter('port').value)
        self.host = str(self.get_parameter('host').value)

        self.bridge = CvBridge()
        self.cache = StateCache()
        self.converter = MapCoordinateConverter()
        self.validator = GoalValidator(self.converter)

        # Rate tracking and display rate throttling
        self._counts = {
            'camera': 0, 'chase': 0, 'imu': 0, 'odom': 0, 'percept': 0, 'seg': 0,
            'vo': 0, 'slam': 0, 'state_est': 0, 'decision': 0,
            'recovery': 0, 'nav': 0, 'replan_diag': 0, 'yolo': 0, 'panorama': 0
        }
        self._subsystem_rates = {k: 0.0 for k in self._counts}
        self._last_rate_calc = time.time()

        # Stream encode rate limiting (Decouple autonomy from dashboard display rate)
        self._last_encode_time = {
            'raw': 0.0, 'chase': 0.0, 'perception': 0.0, 'segmentation': 0.0, 'vo': 0.0, 'yolo': 0.0, 'panorama': 0.0
        }
        self._min_encode_interval = {
            'raw': 1.0 / 15.0,         # 15 FPS display limit
            'chase': 1.0 / 15.0,       # 15 FPS display limit
            'perception': 1.0 / 10.0,  # 10 FPS display limit
            'segmentation': 1.0 / 10.0,# 10 FPS display limit
            'vo': 1.0 / 10.0,          # 10 FPS display limit
            'yolo': 1.0 / 10.0,        # 10 FPS display limit
            'panorama': 1.0 / 15.0,    # 15 FPS display limit
        }

        # Static assets path
        try:
            pkg_share = get_package_share_directory('naviguard_dashboard')
            static_dir = os.path.join(pkg_share, 'static')
            if not os.path.exists(static_dir):
                static_dir = os.path.join(os.path.dirname(__file__), 'static')
        except Exception:
            static_dir = os.path.join(os.path.dirname(__file__), 'static')

        # Setup Web Server
        NaviguardRequestHandler.state_cache = self.cache
        NaviguardRequestHandler.coord_converter = self.converter
        NaviguardRequestHandler.goal_validator = self.validator
        NaviguardRequestHandler.ros_bridge = self
        NaviguardRequestHandler.static_dir = static_dir

        # Pre-cache high-fidelity outdoor basemap
        try:
            from naviguard_dashboard.basemap_generator import generate_rellis_basemap
            bmap_img = generate_rellis_basemap(600, 600, 0.05, -15.0, -15.0)
            _, enc = cv2.imencode('.png', bmap_img)
            NaviguardRequestHandler._cached_basemap_bytes = enc.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Failed to pre-cache basemap: {e}")

        # ---------------------------------------------------------------------
        # QoS Profiles
        # ---------------------------------------------------------------------
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---------------------------------------------------------------------
        # Subscriptions
        # ---------------------------------------------------------------------
        # 1. Cameras
        self.create_subscription(Image, '/camera/image_raw', self._cb_raw_cam, qos_sensor)
        self.create_subscription(Image, '/perception/debug_image', self._cb_percept_img, qos_sensor)
        self.create_subscription(Image, '/perception/segmentation', self._cb_seg_img, qos_sensor)
        self.create_subscription(Image, '/visual_odometry/debug_image', self._cb_vo_img, qos_sensor)
        self.create_subscription(Image, '/camera/chase_image', self._cb_chase_cam, qos_sensor)
        self.create_subscription(Image, '/perception/yolo/debug_image', self._cb_yolo_cam, qos_sensor)
        self.create_subscription(Image, '/camera/panorama_image', self._cb_panorama_cam, qos_sensor)

        # 2. Sensors & State
        self.create_subscription(Imu, '/imu', self._cb_imu, qos_sensor)
        self.create_subscription(Odometry, '/odom', self._cb_odom, qos_sensor)
        self.create_subscription(Odometry, '/state_estimation/odom', self._cb_state_est_odom, qos_reliable)

        # 3. Visual Odometry Telemetry
        self.create_subscription(DiagnosticArray, '/visual_odometry/telemetry', self._cb_vo_telemetry, qos_reliable)

        # 4. SLAM Map & Pose
        self.create_subscription(OccupancyGrid, '/slam/map', self._cb_map, qos_map)
        self.create_subscription(PoseWithCovarianceStamped, '/slam/pose', self._cb_slam_pose, qos_reliable)
        self.create_subscription(Path, '/slam/trajectory', self._cb_slam_traj, qos_reliable)

        # 5. Confidence Decision
        self.create_subscription(String, '/naviguard/decision', self._cb_decision, qos_reliable)

        # 6. Recovery State & Checkpoint Visualization
        self.create_subscription(String, '/recovery/state', self._cb_recovery, qos_reliable)
        self.create_subscription(MarkerArray, '/recovery/visualization', self._cb_recovery_markers, qos_reliable)

        # 7. Navigation State, Path, Waypoints & Obstacles
        self.create_subscription(String, '/navigation/state', self._cb_nav_state, qos_reliable)
        self.create_subscription(Path, '/navigation/path', self._cb_nav_path, qos_reliable)
        self.create_subscription(PoseArray, '/navigation/waypoints', self._cb_nav_waypoints, qos_reliable)
        self.create_subscription(String, '/navigation/replan_diagnostics', self._cb_replan_diag, qos_reliable)
        self.create_subscription(String, '/navigation/vehicle_diagnostics', self._cb_vehicle_diag, qos_reliable)

        # 8. YOLO Perception Diagnostics
        self.create_subscription(String, '/perception/yolo/diagnostics', self._cb_yolo_diag, qos_reliable)

        # ---------------------------------------------------------------------
        # Publishers (CRITICAL: NEVER publish to /cmd_vel)
        # ---------------------------------------------------------------------
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/navigation/set_goal', 10)

        # ---------------------------------------------------------------------
        # Service Clients
        # ---------------------------------------------------------------------
        self.cancel_cli = self.create_client(Trigger, '/navigation/cancel_goal')
        self.fault_cli = self.create_client(Trigger, '/recovery/trigger_manual_recovery')
        self.reset_cli = self.create_client(Trigger, '/recovery/reset_budget')

        # Background Rate & Health Timer
        self.timer = self.create_timer(1.0, self._rate_and_health_loop)

        # Start HTTP Server thread (allow immediate address re-use)
        ThreadingHTTPServer.allow_reuse_address = True
        self.server = ThreadingHTTPServer((self.host, self.port), NaviguardRequestHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.get_logger().info(
            f"NAVIGUARD Operator Dashboard running at http://{self.host}:{self.port} (Static: {static_dir})"
        )

    # -------------------------------------------------------------------------
    # Image Callbacks (Rate-Throttled Encoding to protect Autonomy Stack)
    # -------------------------------------------------------------------------
    def _cb_raw_cam(self, msg: Image):
        self._counts['camera'] += 1
        now = time.time()
        if now - self._last_encode_time['raw'] < self._min_encode_interval['raw']:
            return
        self._last_encode_time['raw'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("raw", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates['camera'])
        except Exception:
            pass

    def _cb_percept_img(self, msg: Image):
        self._counts['percept'] += 1
        now = time.time()
        if now - self._last_encode_time['perception'] < self._min_encode_interval['perception']:
            return
        self._last_encode_time['perception'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("perception", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates['percept'])
        except Exception:
            pass

    def _cb_seg_img(self, msg: Image):
        self._counts['seg'] += 1
        now = time.time()
        if now - self._last_encode_time['segmentation'] < self._min_encode_interval['segmentation']:
            return
        self._last_encode_time['segmentation'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("segmentation", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates['seg'])
        except Exception:
            pass

    def _cb_vo_img(self, msg: Image):
        self._counts['vo'] += 1
        now = time.time()
        if now - self._last_encode_time['vo'] < self._min_encode_interval['vo']:
            return
        self._last_encode_time['vo'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("vo", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates['vo'])
        except Exception:
            pass

    def _cb_chase_cam(self, msg: Image):
        self._counts['chase'] += 1
        now = time.time()
        if now - self._last_encode_time['chase'] < self._min_encode_interval['chase']:
            return
        self._last_encode_time['chase'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("chase", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates['chase'])
        except Exception:
            pass

    def _cb_yolo_cam(self, msg: Image):
        self._counts['yolo'] += 1
        now = time.time()
        if now - self._last_encode_time['yolo'] < self._min_encode_interval['yolo']:
            return
        self._last_encode_time['yolo'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 640:
                cv_img = cv2.resize(cv_img, (640, int(640 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("yolo", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates.get('yolo', 10.0))
        except Exception:
            pass

    def _cb_panorama_cam(self, msg: Image):
        self._counts['panorama'] += 1
        now = time.time()
        if now - self._last_encode_time['panorama'] < self._min_encode_interval['panorama']:
            return
        self._last_encode_time['panorama'] = now

        t0 = time.perf_counter()
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if cv_img.shape[1] > 1024:
                cv_img = cv2.resize(cv_img, (1024, int(1024 * cv_img.shape[0] / cv_img.shape[1])))
            _, enc = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.cache.set_jpeg_frame("panorama", enc.tobytes(), timestamp=now, encode_latency_ms=latency_ms, source_fps=self._subsystem_rates.get('panorama', 15.0))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Sensor Callbacks
    # -------------------------------------------------------------------------
    def _cb_imu(self, msg: Imu):
        self._counts['imu'] += 1

    def _cb_odom(self, msg: Odometry):
        self._counts['odom'] += 1

    def _cb_state_est_odom(self, msg: Odometry):
        self._counts['state_est'] += 1

    # -------------------------------------------------------------------------
    # VO Telemetry Callback
    # -------------------------------------------------------------------------
    def _cb_vo_telemetry(self, msg: DiagnosticArray):
        for status in msg.status:
            vals = {kv.key: kv.value for kv in status.values}
            try:
                self.cache.vo_telemetry['num_detected'] = int(vals.get('num_detected', 0))
                self.cache.vo_telemetry['num_tracked'] = int(vals.get('num_tracked', 0))
                self.cache.vo_telemetry['num_inliers'] = int(vals.get('num_inliers', 0))
                self.cache.vo_telemetry['median_displacement_px'] = float(vals.get('median_displacement_px', 0.0))
                self.cache.vo_telemetry['geom_status'] = vals.get('geom_status', 'UNKNOWN')
                self.cache.vo_telemetry['scale_status'] = vals.get('translation_scale_status', 'UNKNOWN')
                self.cache.vo_telemetry['rot_yaw_deg'] = float(vals.get('rot_yaw_deg', 0.0))
                self.cache.vo_telemetry['proc_fps'] = float(vals.get('proc_fps', 0.0))
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # SLAM Map & Pose Callbacks
    # -------------------------------------------------------------------------
    def _cb_map(self, msg: OccupancyGrid):
        info = msg.info
        self.converter.update_map_meta(
            resolution=info.resolution,
            width=info.width,
            height=info.height,
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y
        )
        self.cache.map_meta['width'] = info.width
        self.cache.map_meta['height'] = info.height
        self.cache.map_meta['resolution'] = info.resolution
        self.cache.map_meta['origin_x'] = info.origin.position.x
        self.cache.map_meta['origin_y'] = info.origin.position.y
        self.cache.map_meta['has_data'] = True
        self.cache.grid_data = list(msg.data)

        # Render occupancy grid to RGBA PNG with alpha transparency
        try:
            arr = np.array(msg.data, dtype=np.int8).reshape((info.height, info.width))
            rgba = np.zeros((info.height, info.width, 4), dtype=np.uint8)
            # -1: Unknown -> completely transparent (Alpha = 0)
            rgba[arr == -1] = [0, 0, 0, 0]
            # 0: Explored free space -> subtle translucent tint (B=180, G=240, R=180, A=40)
            rgba[arr == 0] = [180, 240, 180, 40]
            # >50: Confirmed obstacles -> high contrast red/orange hazard (B=20, G=20, R=240, A=240)
            rgba[arr > 50] = [20, 20, 240, 240]
            rgba_flipped = cv2.flip(rgba, 0)
            _, enc = cv2.imencode('.png', rgba_flipped)
            self.cache.map_png_bytes = enc.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Map rendering error: {e}")

    def _cb_slam_pose(self, msg: PoseWithCovarianceStamped):
        self._counts['slam'] += 1
        p = msg.pose.pose.position
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.cache.update_robot_pose(p.x, p.y, p.z, yaw)

    def _cb_slam_traj(self, msg: Path):
        pts = [{'x': ps.pose.position.x, 'y': ps.pose.position.y} for ps in msg.poses[-150:]]
        self.cache.trajectory = pts

    # -------------------------------------------------------------------------
    # Decision, Recovery & Navigation Callbacks
    # -------------------------------------------------------------------------
    def _cb_decision(self, msg: String):
        self._counts['decision'] += 1
        try:
            data = json.loads(msg.data)
            prev_dec = self.cache.confidence_state
            new_dec = data.get('state', data.get('decision', 'CONTINUE'))
            self.cache.confidence_state = new_dec
            self.cache.confidence_primary_reason = data.get('primary_reason', 'NOMINAL_OPERATION')
            scores = data.get('scores', {})
            for k in self.cache.confidence_scores:
                if k in scores:
                    self.cache.confidence_scores[k] = float(scores[k])

            if new_dec != prev_dec:
                self.cache.add_event("DECISION", f"Confidence Transition: {prev_dec} -> {new_dec} ({self.cache.confidence_primary_reason})")
        except Exception:
            pass

    def _cb_recovery(self, msg: String):
        self._counts['recovery'] += 1
        try:
            data = json.loads(msg.data)
            prev_rec = self.cache.recovery_state
            new_rec = data.get('recovery_state', 'NORMAL')
            self.cache.recovery_state = new_rec
            budget = data.get('budget', {})
            self.cache.recovery_attempts = int(data.get('attempts', budget.get('attempt_count', 0)))
            self.cache.recovery_max_attempts = int(data.get('max_attempts', budget.get('max_attempts', 3)))
            self.cache.recovery_dwell_sec = float(data.get('dwell_sec', 0.0))
            self.cache.recovery_strategy = data.get('active_strategy', 'None')

            if new_rec != prev_rec:
                self.cache.add_event("RECOVERY", f"Recovery Transition: {prev_rec} -> {new_rec} (Strategy: {self.cache.recovery_strategy})")
        except Exception:
            pass

    def _cb_recovery_markers(self, msg: MarkerArray):
        """Extract trusted recovery checkpoints for Layer 10 visualization."""
        ckpts = []
        for m in msg.markers:
            if m.ns == "checkpoints":
                ckpts.append({
                    "x": round(m.pose.position.x, 2),
                    "y": round(m.pose.position.y, 2),
                    "id": m.id,
                })
        if ckpts:
            self.cache.set_checkpoints(ckpts)

    def _cb_nav_state(self, msg: String):
        self._counts['nav'] += 1
        try:
            data = json.loads(msg.data)
            prev_mstate = self.cache.mission_state
            new_mstate = data.get('mission_state', 'IDLE')
            self.cache.mission_state = new_mstate

            ns = self.cache.nav_stats
            ns['dist_to_goal_m'] = float(data.get('dist_to_goal_m', 0.0))
            ns['current_waypoint_idx'] = int(data.get('current_waypoint_idx', 0))
            ns['total_waypoints'] = int(data.get('total_waypoints', 0))
            ns['replan_count'] = int(data.get('replan_count', 0))
            ns['cmd_vx'] = float(data.get('cmd_vx', 0.0))
            ns['cmd_wz'] = float(data.get('cmd_wz', 0.0))
            ns['clearance_m'] = float(data.get('clearance_m', 1.0))
            ns['terrain_cost'] = float(data.get('terrain_cost', 0.0))
            ns['speed_scale'] = float(data.get('speed_scale', 1.0))
            ns['nav_reason'] = data.get('nav_reason', 'Nominal path following')

            goal_dict = data.get('goal')
            self.cache.set_active_goal(goal_dict)

            # Store persistent blocked regions for Layer 9
            if 'persistent_blocked_regions' in data:
                self.cache.set_persistent_obstacles(data['persistent_blocked_regions'])

            # Store structured failure record
            if 'failure_record' in data and data['failure_record']:
                self.cache.set_failure_record(data['failure_record'])
            elif new_mstate == "MISSION_FAILED" and not self.cache.failure_record:
                # Synthesize fallback structured record
                rec = {
                    "failure_code": "NO_SAFE_PATH",
                    "failure_category": "NAVIGATION",
                    "human_reason": data.get("failure_reason", "No collision-free traversable path exists under current constraints."),
                    "detail": "Planner failed to find a valid route to the goal.",
                    "timestamp": time.time(),
                    "mission_state": "MISSION_FAILED",
                    "navigation_state": "FAILED",
                    "recovery_state": self.cache.recovery_state,
                    "confidence": self.cache.confidence_scores.get("overall", 1.0),
                    "visual_confidence": self.cache.confidence_scores.get("visual", 1.0),
                    "localization_confidence": self.cache.confidence_scores.get("localization", 1.0),
                    "goal": self.cache.active_goal,
                    "robot_pose": self.cache.robot_pose,
                    "distance_to_goal": ns.get("dist_to_goal_m", 0.0),
                    "last_valid_pose": self.cache.last_valid_pose,
                    "recovery_attempts": self.cache.recovery_attempts,
                    "recovery_max_attempts": self.cache.recovery_max_attempts,
                    "replan_count": ns.get("replan_count", 0),
                    "blocked_regions": len(self.cache.persistent_obstacles),
                    "path_status": "BLOCKED",
                    "sensor_status": {k: v["status"] for k, v in self.cache.subsystems.items()},
                }
                self.cache.set_failure_record(rec)

            # Store structured success report
            if 'success_record' in data and data['success_record']:
                self.cache.set_success_record(data['success_record'])

            if new_mstate != prev_mstate:
                self.cache.add_event("MISSION", f"Mission State: {prev_mstate} -> {new_mstate}")
        except Exception:
            pass

    def _cb_nav_path(self, msg: Path):
        pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.cache.set_nav_path(pts)

    def _cb_nav_waypoints(self, msg: PoseArray):
        wps = [{"x": round(p.position.x, 2), "y": round(p.position.y, 2)} for p in msg.poses]
        self.cache.set_waypoints(wps)

    def _cb_replan_diag(self, msg: String):
        self._counts['replan_diag'] += 1
        try:
            data = json.loads(msg.data)
            self.cache.replan_diagnostics.update(data)
            reason = data.get("replan_reason", "UNKNOWN")
            old_l = data.get("old_path_length_m", 0.0)
            new_l = data.get("new_path_length_m", 0.0)
            diff = data.get("path_difference_m", 0.0)
            status = data.get("status", "REPLAN")
            self.cache.add_event("REPLAN", f"{status} [{reason}]: old={old_l:.2f}m, new={new_l:.2f}m (diff={diff:+.2f}m)")
        except Exception:
            pass

    def _cb_vehicle_diag(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.cache.set_vehicle_diagnostics(data)
        except Exception:
            pass

    def _cb_yolo_diag(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.cache.set_yolo_diagnostics(data)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Subsystem Rate & Health Calculation Loop (1 Hz)
    # -------------------------------------------------------------------------
    def _rate_and_health_loop(self):
        now = time.time()
        dt = max(0.1, now - self._last_rate_calc)
        self._last_rate_calc = now

        for k in self._counts:
            self._subsystem_rates[k] = self._counts[k] / dt

        rates = {
            'Camera': self._subsystem_rates['camera'],
            'IMU': self._subsystem_rates['imu'],
            'Odometry': self._subsystem_rates['odom'],
            'Perception': self._subsystem_rates['percept'],
            'Visual Odometry': self._subsystem_rates['vo'],
            'State Estimation': self._subsystem_rates['state_est'],
            'SLAM': self._subsystem_rates['slam'],
            'Confidence': self._subsystem_rates['decision'],
            'Recovery': self._subsystem_rates['recovery'],
            'Navigation': self._subsystem_rates['nav'],
        }

        # Gazebo is online if clock / bridge is producing IMU or Camera
        gz_rate = max(rates['IMU'], rates['Camera'])
        gz_online = gz_rate > 1.0
        self.cache.update_subsystem_rate("Gazebo", gz_rate, is_online=gz_online)

        thresholds = {
            'Camera': 1.0,
            'IMU': 10.0,
            'Odometry': 5.0,
            'Perception': 1.0,
            'Visual Odometry': 1.0,
            'State Estimation': 1.0,
            'SLAM': 0.5,
            'Confidence': 0.5,
            'Recovery': 0.5,
            'Navigation': 0.5,
        }

        for name, r in rates.items():
            thresh = thresholds.get(name, 0.5)
            is_on = (r >= thresh)
            self.cache.update_subsystem_rate(name, r, is_online=is_on)

        self.cache.update_subsystem_rate("Dashboard", 1.0, is_online=True)

        for k in self._counts:
            self._counts[k] = 0

    # -------------------------------------------------------------------------
    # Operator Actions (Published to Real ROS Interfaces)
    # -------------------------------------------------------------------------
    def publish_navigation_goal(self, x: float, y: float, yaw: float = 0.0) -> bool:
        """Publishes PoseStamped goal to /goal_pose and /navigation/set_goal."""
        try:
            self.cache.clear_failure_and_success()
            msg = GoalValidator.create_goal_msg(x, y, yaw, frame_id="map", stamp=self.get_clock().now().to_msg())
            self.goal_pub.publish(msg)
            self.nav_goal_pub.publish(msg)
            self.get_logger().info(f"Operator published goal to /goal_pose and /navigation/set_goal: X={x:.2f}m, Y={y:.2f}m")
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to publish goal: {e}")
            return False

    def cancel_active_mission(self) -> bool:
        """Invokes /navigation/cancel_goal service."""
        try:
            self.cache.clear_failure_and_success()
            if not self.cancel_cli.service_is_ready():
                return False
            req = Trigger.Request()
            self.cancel_cli.call_async(req)
            return True
        except Exception as e:
            self.get_logger().error(f"Cancel mission call error: {e}")
            return False

    def trigger_recovery_fault(self) -> bool:
        """Invokes /recovery/trigger_manual_recovery service."""
        try:
            if not self.fault_cli.service_is_ready():
                return False
            req = Trigger.Request()
            self.fault_cli.call_async(req)
            return True
        except Exception as e:
            self.get_logger().error(f"Trigger fault call error: {e}")
            return False

    def reset_recovery_budget(self) -> bool:
        """Invokes /recovery/reset_budget service."""
        try:
            if not self.reset_cli.service_is_ready():
                return False
            req = Trigger.Request()
            self.reset_cli.call_async(req)
            return True
        except Exception as e:
            self.get_logger().error(f"Reset budget call error: {e}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = NaviguardDashboardNode()
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
