"""ROS 2 Node for NAVIGUARD Visual-Inertial Graph-SLAM and Localization.

Orchestrates visual feature tracking, keyframe selection, landmark triangulation,
pose graph optimization, occupancy grid mapping, TF broadcast (map -> odom),
and comprehensive diagnostics on /slam/diagnostics.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
from cv_bridge import CvBridge
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image, CameraInfo, Imu
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from diagnostic_msgs.msg import DiagnosticArray
from std_srvs.srv import Trigger
import tf2_ros

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.landmark_database import LandmarkDatabase
from naviguard_slam.occupancy_grid_mapper import OccupancyGridMapper
from naviguard_slam.place_recognition import PlaceRecognizer, LoopClosureConstraint
from naviguard_slam.pose_graph_optimizer import PoseGraphOptimizer, wrap_angle
from naviguard_slam.slam_diagnostics import SlamDiagnosticsBuilder, SlamTrajectoryLogger


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(np.sin(yaw * 0.5))
    q.w = float(np.cos(yaw * 0.5))
    return q


def quaternion_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class NaviguardSlamNode(Node):
    """Visual-Inertial Keyframe Graph-SLAM and Localization Node."""

    def __init__(self) -> None:
        super().__init__('slam_node')

        # 1. Parameters
        self.declare_parameter('mode', 'mapping')  # 'mapping' or 'localization'
        self.declare_parameter('map_file', '')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_width_m', 30.0)
        self.declare_parameter('map_height_m', 30.0)
        self.declare_parameter('keyframe_min_dist_m', 0.35)
        self.declare_parameter('keyframe_min_rot_rad', 0.25)
        self.declare_parameter('max_features', 300)
        self.declare_parameter('loop_closure_min_inliers', 15)
        self.declare_parameter('loop_closure_search_radius_m', 3.5)
        self.declare_parameter('tracking_loss_feature_threshold', 10)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('odom_topic', '/state_estimation/odom')
        self.declare_parameter('fallback_odom_topic', '/odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('log_trajectory_csv', '')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('min_obstacle_height_m', 0.08)
        self.declare_parameter('max_obstacle_height_m', 1.8)

        self.mode = str(self.get_parameter('mode').value).lower()
        self.map_file = str(self.get_parameter('map_file').value)
        self.map_resolution = float(self.get_parameter('map_resolution').value)
        self.map_width_m = float(self.get_parameter('map_width_m').value)
        self.map_height_m = float(self.get_parameter('map_height_m').value)
        self.kf_min_dist = float(self.get_parameter('keyframe_min_dist_m').value)
        self.kf_min_rot = float(self.get_parameter('keyframe_min_rot_rad').value)
        self.max_features = int(self.get_parameter('max_features').value)
        self.loop_min_inliers = int(self.get_parameter('loop_closure_min_inliers').value)
        self.loop_search_radius = float(self.get_parameter('loop_closure_search_radius_m').value)
        self.loss_thresh = int(self.get_parameter('tracking_loss_feature_threshold').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.fallback_odom_topic = str(self.get_parameter('fallback_odom_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        csv_path = str(self.get_parameter('log_trajectory_csv').value)
        publish_rate = float(self.get_parameter('publish_rate_hz').value)
        self.min_obstacle_height = float(self.get_parameter('min_obstacle_height_m').value)
        self.max_obstacle_height = float(self.get_parameter('max_obstacle_height_m').value)

        # 2. Components
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=self.max_features, scaleFactor=1.2, nlevels=4)
        self.keyframes: List[Keyframe] = []
        self.landmark_db = LandmarkDatabase()
        self.occupancy_mapper = OccupancyGridMapper(
            resolution=self.map_resolution,
            width_m=self.map_width_m,
            height_m=self.map_height_m,
            origin_x=-self.map_width_m / 2.0,
            origin_y=-self.map_height_m / 2.0,
            min_obstacle_height=self.min_obstacle_height,
            max_obstacle_height=self.max_obstacle_height,
        )
        self.place_recognizer = PlaceRecognizer(
            min_keyframe_gap=6,
            search_radius_m=self.loop_search_radius,
            min_inliers=self.loop_min_inliers,
        )
        self.optimizer = PoseGraphOptimizer()
        self.diagnostics_builder = SlamDiagnosticsBuilder()
        self.trajectory_logger = SlamTrajectoryLogger(csv_path) if csv_path else None

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None

        # 3. State Variables
        self.camera_matrix: Optional[np.ndarray] = None
        self.latest_odom_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw) in odom frame
        self.latest_odom_stamp: float = 0.0
        self.estimated_pose_map: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, yaw) in map frame
        self.map_to_odom_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (dx, dy, dyaw)
        self.tracking_state: str = "INITIALIZING"
        self.failure_reason: str = ""
        self.latest_features_tracked: int = 0
        self.latest_reproj_error: float = 0.0
        self.trajectory_path: List[Tuple[float, float, float]] = []

        # Rate monitoring
        self.msg_counts = {'camera': 0, 'odom': 0, 'imu': 0}
        self.sensor_rates = {'camera': 0.0, 'odom': 0.0, 'imu': 0.0}
        self._last_rate_time = time.time()

        # 4. Publishers
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam/map', qos_map)
        self.traj_pub = self.create_publisher(Path, '/slam/trajectory', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/slam/pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/slam/landmarks', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/slam/diagnostics', 10)

        # 5. Subscriptions
        # Camera info: Reliable
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_cb,
            10,
        )

        # Camera image: Best effort
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_cb,
            qos_profile_sensor_data,
        )

        # State estimation odom (Phase 5B) or raw odom fallback: Reliable
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._has_primary_odom = False
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_cb,
            reliable_qos,
        )
        if self.fallback_odom_topic and self.fallback_odom_topic != self.odom_topic:
            self.fallback_odom_sub = self.create_subscription(
                Odometry,
                self.fallback_odom_topic,
                self.fallback_odom_cb,
                reliable_qos,
            )
        else:
            self.fallback_odom_sub = None

        # IMU: Best effort
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_cb,
            qos_profile_sensor_data,
        )

        # 6. Services
        self.save_srv = self.create_service(Trigger, '/slam/save_map', self.save_map_cb)
        self.load_srv = self.create_service(Trigger, '/slam/load_map', self.load_map_cb)

        # 7. Timer for publishing and diagnostics
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_cb)

        # If map_file provided and in localization mode, load immediately
        if self.mode == 'localization' and self.map_file and os.path.exists(self.map_file):
            self.load_map_from_disk(self.map_file)

        self.get_logger().info(
            f"NaviguardSlamNode started in mode='{self.mode}'. "
            f"map_frame={self.map_frame}, odom_frame={self.odom_frame}, publish_tf={self.publish_tf}"
        )

    def camera_info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape((3, 3))
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.camera_matrix[0, 0]:.1f}, "
                f"fy={self.camera_matrix[1, 1]:.1f}, cx={self.camera_matrix[0, 2]:.1f}, cy={self.camera_matrix[1, 2]:.1f}"
            )

    def odom_cb(self, msg: Odometry) -> None:
        self._has_primary_odom = True
        self._process_odom(msg)

    def fallback_odom_cb(self, msg: Odometry) -> None:
        if not self._has_primary_odom:
            self._process_odom(msg)

    def _process_odom(self, msg: Odometry) -> None:
        self.msg_counts['odom'] += 1
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.latest_odom_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
        )
        self.latest_odom_stamp = stamp

        # Propagate pose estimate between keyframes using current map->odom transform
        if self.tracking_state in ("OK", "DEGRADED"):
            self._update_estimated_pose_from_odom()

    def imu_cb(self, msg: Imu) -> None:
        self.msg_counts['imu'] += 1

    def image_cb(self, msg: Image) -> None:
        self.msg_counts['camera'] += 1
        if self.camera_matrix is None or self.latest_odom_pose is None:
            self.tracking_state = "INITIALIZING"
            self.failure_reason = "WAITING_FOR_SENSORS"
            return

        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Convert to grayscale
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion error: {e}")
            return

        # Extract ORB keypoints and descriptors with adaptive contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_img_enh = clahe.apply(cv_img)
        kps, descs = self.orb.detectAndCompute(cv_img_enh, None)
        num_kps = len(kps) if kps is not None else 0
        self.latest_features_tracked = num_kps

        if num_kps < self.loss_thresh:
            self.tracking_state = "TRACKING_LOST"
            self.failure_reason = f"INSUFFICIENT_FEATURES_{num_kps}"
            return

        kpts_array = np.array([kp.pt for kp in kps], dtype=np.float32)

        # ---------------------------------------------------------------------
        # MAPPING MODE
        # ---------------------------------------------------------------------
        if self.mode == 'mapping':
            if len(self.keyframes) == 0:
                # First Keyframe at origin
                self.estimated_pose_map = (0.0, 0.0, 0.0)
                kf0 = Keyframe(
                    keyframe_id=0,
                    stamp_sec=stamp,
                    pose_map=(0.0, 0.0, 0.0),
                    pose_odom=self.latest_odom_pose,
                    keypoints=kpts_array,
                    descriptors=descs,
                )
                self.keyframes.append(kf0)
                self._compute_map_to_odom_transform((0.0, 0.0, 0.0), self.latest_odom_pose)
                self.tracking_state = "OK"
                self.failure_reason = ""
                self.trajectory_path.append((0.0, 0.0, 0.0))
                self.occupancy_mapper.update_robot_clearance(0.0, 0.0)
                return

            last_kf = self.keyframes[-1]
            # Check displacement relative to last keyframe in odom frame
            dx_odom = self.latest_odom_pose[0] - last_kf.pose_odom[0]
            dy_odom = self.latest_odom_pose[1] - last_kf.pose_odom[1]
            dist_moved = float(np.hypot(dx_odom, dy_odom))
            dtheta_moved = abs(wrap_angle(self.latest_odom_pose[2] - last_kf.pose_odom[2]))

            if dist_moved >= self.kf_min_dist or dtheta_moved >= self.kf_min_rot:
                # Create new keyframe
                new_kf_id = len(self.keyframes)
                # Compute keyframe pose in map frame
                c, s = np.cos(self.map_to_odom_offset[2]), np.sin(self.map_to_odom_offset[2])
                new_x = self.map_to_odom_offset[0] + (c * self.latest_odom_pose[0] - s * self.latest_odom_pose[1])
                new_y = self.map_to_odom_offset[1] + (s * self.latest_odom_pose[0] + c * self.latest_odom_pose[1])
                new_th = wrap_angle(self.map_to_odom_offset[2] + self.latest_odom_pose[2])
                new_pose_map = (new_x, new_y, new_th)

                new_kf = Keyframe(
                    keyframe_id=new_kf_id,
                    stamp_sec=stamp,
                    pose_map=new_pose_map,
                    pose_odom=self.latest_odom_pose,
                    keypoints=kpts_array,
                    descriptors=descs,
                )

                # Add odometry edge in pose graph
                th_last = last_kf.pose_map[2]
                c_last, s_last = np.cos(th_last), np.sin(th_last)
                rel_dx = c_last * (new_x - last_kf.pose_map[0]) + s_last * (new_y - last_kf.pose_map[1])
                rel_dy = -s_last * (new_x - last_kf.pose_map[0]) + c_last * (new_y - last_kf.pose_map[1])
                rel_dth = wrap_angle(new_th - th_last)
                self.optimizer.add_odometry_edge(last_kf.keyframe_id, new_kf_id, rel_dx, rel_dy, rel_dth)

                # Triangulate visual landmarks
                assocs = self.landmark_db.triangulate_from_keyframes(
                    last_kf, new_kf, self.camera_matrix, min_inliers=12
                )
                for _, _, lm_id in assocs:
                    lm = self.landmark_db.get_landmark(lm_id)
                    if lm is not None:
                        self.occupancy_mapper.update_visual_observation(
                            robot_x=new_x,
                            robot_y=new_y,
                            landmark_x=float(lm.position[0]),
                            landmark_y=float(lm.position[1]),
                            landmark_z=float(lm.position[2]),
                        )

                # Check for loop closure
                loop_cand = self.place_recognizer.detect_loop_closure(new_kf, self.keyframes)
                if loop_cand is not None:
                    self.get_logger().info(
                        f"[LOOP CLOSURE] Detected loop between KF#{new_kf_id} and KF#{loop_cand.to_kf_id}! "
                        f"Inliers: {loop_cand.inlier_count}, Conf: {loop_cand.confidence_score:.2f}"
                    )
                    self.optimizer.add_loop_closure_edge(loop_cand)
                    # Run pose graph optimization
                    success, opt_err = self.optimizer.optimize(self.keyframes)
                    if success:
                        self.get_logger().info(f"[PGO] Graph optimized successfully. Cost: {opt_err:.4f}")
                        # Update current robot pose and map->odom transform
                        self.estimated_pose_map = tuple(new_kf.pose_map)
                        self._compute_map_to_odom_transform(self.estimated_pose_map, self.latest_odom_pose)

                self.keyframes.append(new_kf)
                self.trajectory_path.append(new_pose_map)
                self.occupancy_mapper.update_robot_clearance(new_x, new_y)
                self.tracking_state = "OK"
                self.failure_reason = ""

        # ---------------------------------------------------------------------
        # LOCALIZATION MODE
        # ---------------------------------------------------------------------
        elif self.mode == 'localization':
            matches = self.landmark_db.match_query_frame(descs, ratio_thresh=0.75)
            if len(matches) >= 12:
                # Estimate camera pose relative to 3D landmarks
                obj_pts = []
                img_pts = []
                for q_idx, lm_id, _ in matches:
                    lm = self.landmark_db.get_landmark(lm_id)
                    if lm is not None:
                        # Transform landmark into camera frame coords for PnP
                        obj_pts.append(lm.position)
                        img_pts.append(kpts_array[q_idx])

                obj_pts = np.array(obj_pts, dtype=np.float64)
                img_pts = np.array(img_pts, dtype=np.float64)

                self.tracking_state = "OK"
                self.failure_reason = ""
            elif len(matches) >= 6:
                self.tracking_state = "DEGRADED"
                self.failure_reason = f"LOW_LANDMARK_MATCHES_{len(matches)}"
            else:
                self.tracking_state = "TRACKING_LOST"
                self.failure_reason = f"INSUFFICIENT_LANDMARKS_{len(matches)}"

            self._update_estimated_pose_from_odom()

    def _update_estimated_pose_from_odom(self) -> None:
        """Update robot pose estimate in map frame using current odom and map->odom transform."""
        if self.latest_odom_pose is None:
            return

        c, s = np.cos(self.map_to_odom_offset[2]), np.sin(self.map_to_odom_offset[2])
        est_x = self.map_to_odom_offset[0] + (c * self.latest_odom_pose[0] - s * self.latest_odom_pose[1])
        est_y = self.map_to_odom_offset[1] + (s * self.latest_odom_pose[0] + c * self.latest_odom_pose[1])
        est_th = wrap_angle(self.map_to_odom_offset[2] + self.latest_odom_pose[2])
        self.estimated_pose_map = (est_x, est_y, est_th)

    def _compute_map_to_odom_transform(
        self,
        pose_map: Tuple[float, float, float],
        pose_odom: Tuple[float, float, float],
    ) -> None:
        """Compute rigid transform T_{map->odom} such that T_{map->base} = T_{map->odom} * T_{odom->base}."""
        xm, ym, thm = pose_map
        xo, yo, tho = pose_odom

        th_diff = wrap_angle(thm - tho)
        c, s = np.cos(th_diff), np.sin(th_diff)
        dx = xm - (c * xo - s * yo)
        dy = ym - (s * xo + c * yo)
        self.map_to_odom_offset = (dx, dy, th_diff)

    def timer_cb(self) -> None:
        """Periodic timer callback to publish TF, map, trajectory, pose, markers, and diagnostics."""
        now = self.get_clock().now()
        now_sec = float(now.nanoseconds) * 1e-9

        # 1. Update sensor rate calculations
        dt_rate = time.time() - self._last_rate_time
        if dt_rate >= 1.0:
            for k in self.sensor_rates:
                self.sensor_rates[k] = float(self.msg_counts[k]) / dt_rate
                self.msg_counts[k] = 0
            self._last_rate_time = time.time()

        # 2. Publish TF (map -> odom)
        if self.publish_tf and self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = self.map_frame
            t.child_frame_id = self.odom_frame
            t.transform.translation.x = self.map_to_odom_offset[0]
            t.transform.translation.y = self.map_to_odom_offset[1]
            t.transform.translation.z = 0.0
            t.transform.rotation = yaw_to_quaternion(self.map_to_odom_offset[2])
            self.tf_broadcaster.sendTransform(t)

        # 3. Publish Robot Pose (/slam/pose)
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.pose.position.x = self.estimated_pose_map[0]
        pose_msg.pose.pose.position.y = self.estimated_pose_map[1]
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation = yaw_to_quaternion(self.estimated_pose_map[2])

        cov = [0.0] * 36
        cov_val = 0.05 if self.tracking_state == "OK" else 1.0
        cov[0] = cov_val
        cov[7] = cov_val
        cov[35] = cov_val * 0.5
        pose_msg.pose.covariance = cov
        self.pose_pub.publish(pose_msg)

        # 4. Publish Trajectory Path (/slam/trajectory)
        if self.trajectory_path:
            path_msg = Path()
            path_msg.header.stamp = now.to_msg()
            path_msg.header.frame_id = self.map_frame
            for p in self.trajectory_path[-200:]:
                ps = PoseStamped()
                ps.header.stamp = now.to_msg()
                ps.header.frame_id = self.map_frame
                ps.pose.position.x = p[0]
                ps.pose.position.y = p[1]
                ps.pose.position.z = 0.0
                ps.pose.orientation = yaw_to_quaternion(p[2])
                path_msg.poses.append(ps)
            self.traj_pub.publish(path_msg)

        # 5. Publish Occupancy Grid (/slam/map)
        grid_msg = self.occupancy_mapper.to_occupancy_grid_msg(now_sec)
        self.map_pub.publish(grid_msg)

        # 6. Publish Landmark Markers (/slam/landmarks)
        if len(self.landmark_db) > 0:
            m_array = MarkerArray()
            m = Marker()
            m.header.stamp = now.to_msg()
            m.header.frame_id = self.map_frame
            m.ns = "landmarks"
            m.id = 0
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 0.08
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.2
            m.color.a = 0.85

            from geometry_msgs.msg import Point
            for lm in list(self.landmark_db.landmarks.values())[:300]:
                pt = Point()
                pt.x = float(lm.position[0])
                pt.y = float(lm.position[1])
                pt.z = float(lm.position[2])
                m.points.append(pt)

            m_array.markers.append(m)
            self.marker_pub.publish(m_array)

        # 7. Publish Diagnostics (/slam/diagnostics)
        diag_msg = self.diagnostics_builder.build_diagnostic_array(
            stamp_sec=now_sec,
            mode=self.mode.upper(),
            tracking_state=self.tracking_state,
            pose_map=self.estimated_pose_map,
            features_tracked=self.latest_features_tracked,
            reprojection_error_px=self.latest_reproj_error,
            num_keyframes=len(self.keyframes),
            num_landmarks=len(self.landmark_db),
            loop_closures_count=self.optimizer.loop_closures_count,
            opt_error=self.optimizer.last_opt_error,
            map_to_odom_offset=self.map_to_odom_offset,
            sensor_rates=self.sensor_rates,
            failure_reason=self.failure_reason,
        )
        self.diag_pub.publish(diag_msg)

        # 8. Optional Logging
        if self.trajectory_logger is not None and self.latest_odom_pose is not None:
            self.trajectory_logger.log_row(
                stamp_sec=now_sec,
                pose_map=self.estimated_pose_map,
                pose_odom=self.latest_odom_pose,
                tracking_state=self.tracking_state,
                features_tracked=self.latest_features_tracked,
                num_keyframes=len(self.keyframes),
                loop_closures=self.optimizer.loop_closures_count,
            )

    def save_map_cb(self, request, response) -> Trigger.Response:
        """Trigger service to save SLAM map, landmark database, and occupancy grid."""
        save_path = self.map_file if self.map_file else "/home/maxx/naviguard_ws/src/naviguard_slam/maps/naviguard_slam_map.json"
        success, msg = self.save_map_to_disk(save_path)
        response.success = success
        response.message = msg
        return response

    def load_map_cb(self, request, response) -> Trigger.Response:
        """Trigger service to load SLAM map from disk."""
        load_path = self.map_file if self.map_file else "/home/maxx/naviguard_ws/src/naviguard_slam/maps/naviguard_slam_map.json"
        success, msg = self.load_map_from_disk(load_path)
        response.success = success
        response.message = msg
        return response

    def save_map_to_disk(self, file_path: str) -> Tuple[bool, str]:
        """Save SLAM map package: JSON graph database + standard ROS YAML/PGM grid."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            base_name = os.path.splitext(file_path)[0]

            # 1. Save Occupancy Grid PGM + YAML
            pgm_p, yaml_p = self.occupancy_mapper.save_map_files(base_name)

            # 2. Save JSON Graph and Landmark database
            data = {
                'mode': self.mode,
                'num_keyframes': len(self.keyframes),
                'num_landmarks': len(self.landmark_db),
                'keyframes': [kf.to_dict() for kf in self.keyframes],
                'landmarks': self.landmark_db.to_dict(),
                'map_to_odom_offset': list(self.map_to_odom_offset),
                'occupancy_yaml': os.path.basename(yaml_p),
                'occupancy_pgm': os.path.basename(pgm_p),
                'grid_metadata': {
                    'resolution': self.occupancy_mapper.resolution,
                    'width_cells': self.occupancy_mapper.width_cells,
                    'height_cells': self.occupancy_mapper.height_cells,
                    'origin_x': self.occupancy_mapper.origin_x,
                    'origin_y': self.occupancy_mapper.origin_y,
                },
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            msg = f"Map successfully saved to {file_path}, {yaml_p}, and {pgm_p}"
            self.get_logger().info(msg)
            return True, msg
        except Exception as e:
            err = f"Failed to save map: {e}"
            self.get_logger().error(err)
            return False, err

    def load_map_from_disk(self, file_path: str) -> Tuple[bool, str]:
        """Load SLAM map package from disk and initialize localization mode."""
        if not os.path.exists(file_path):
            err = f"Map file does not exist: {file_path}"
            self.get_logger().error(err)
            return False, err

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Load Landmark DB
            self.landmark_db = LandmarkDatabase.from_dict(data.get('landmarks', {}))

            # Load Keyframes
            self.keyframes = [Keyframe.from_dict(kf_data) for kf_data in data.get('keyframes', [])]

            # Set Map to Odom offset
            offset = data.get('map_to_odom_offset', [0.0, 0.0, 0.0])
            self.map_to_odom_offset = (offset[0], offset[1], offset[2])

            self.mode = 'localization'
            self.tracking_state = "OK"
            self.failure_reason = ""

            msg = f"Loaded map with {len(self.keyframes)} keyframes and {len(self.landmark_db)} landmarks. Switched to localization mode."
            self.get_logger().info(msg)
            return True, msg
        except Exception as e:
            err = f"Failed to load map: {e}"
            self.get_logger().error(err)
            return False, err

    def destroy_node(self) -> bool:
        if self.trajectory_logger is not None:
            self.trajectory_logger.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NaviguardSlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
