"""ROS 2 Perception Node for NAVIGUARD Outdoor Autonomous UGV.

Subscribes to camera image and camera_info streams, executes modular
OpenCV preprocessing, calculates performance metrics, and publishes
diagnostic perception images.
"""

import json
import time
from collections import deque
from typing import Deque, Optional
import cv_bridge
from cv_bridge import CvBridgeError
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from naviguard_perception.image_processor import (
    ImageProcessor,
    ImageProcessorConfig,
)
from naviguard_perception.yolo_detector import YOLODetector, YOLOConfig
from naviguard_perception.perception_fusion import PerceptionFusion
from naviguard_perception.lidar_obstacle_detector import LidarObstacleDetector


class NaviguardPerceptionNode(Node):
    """ROS 2 Node managing camera stream subscriptions and image processing."""

    def __init__(self) -> None:
        super().__init__('naviguard_perception_node')

        # ------------------- Declare Parameters -------------------
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('debug_image_topic', '/perception/debug_image')
        self.declare_parameter('segmentation_topic', '/perception/segmentation')
        self.declare_parameter('display_mode', 'overlay')
        self.declare_parameter('enable_clahe', True)
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_grid_size', 8)
        self.declare_parameter('canny_low_threshold', 50)
        self.declare_parameter('canny_high_threshold', 150)
        self.declare_parameter('ground_roi_top_ratio', 0.60)
        self.declare_parameter('ground_roi_bottom_ratio', 0.98)
        self.declare_parameter('self_mask_height_ratio', 0.15)
        self.declare_parameter('diag_publish_period_sec', 3.0)
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('yolo_device', 'AUTO')
        self.declare_parameter('yolo_conf_thresh', 0.40)
        self.declare_parameter('yolo_iou_thresh', 0.45)
        self.declare_parameter('yolo_inference_rate', 12.0)
        self.declare_parameter('yolo_half_precision', False)

        # Retrieve parameter values
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        debug_image_topic = self.get_parameter('debug_image_topic').get_parameter_value().string_value
        segmentation_topic = self.get_parameter('segmentation_topic').get_parameter_value().string_value

        config = ImageProcessorConfig(
            display_mode=self.get_parameter('display_mode').get_parameter_value().string_value,
            enable_clahe=self.get_parameter('enable_clahe').get_parameter_value().bool_value,
            clahe_clip_limit=self.get_parameter('clahe_clip_limit').get_parameter_value().double_value,
            clahe_grid_size=self.get_parameter('clahe_grid_size').get_parameter_value().integer_value,
            canny_low_threshold=self.get_parameter('canny_low_threshold').get_parameter_value().integer_value,
            canny_high_threshold=self.get_parameter('canny_high_threshold').get_parameter_value().integer_value,
            ground_roi_top_ratio=self.get_parameter('ground_roi_top_ratio').get_parameter_value().double_value,
            ground_roi_bottom_ratio=self.get_parameter('ground_roi_bottom_ratio').get_parameter_value().double_value,
            self_mask_height_ratio=self.get_parameter('self_mask_height_ratio').get_parameter_value().double_value,
        )

        self.processor = ImageProcessor(config)
        self.bridge = cv_bridge.CvBridge()

        # Initialize YOLO Neural Detector & Perception Fusion Engine
        yolo_cfg = YOLOConfig(
            model_path=self.get_parameter('yolo_model_path').get_parameter_value().string_value,
            device=self.get_parameter('yolo_device').get_parameter_value().string_value,
            confidence_threshold=self.get_parameter('yolo_conf_thresh').get_parameter_value().double_value,
            iou_threshold=self.get_parameter('yolo_iou_thresh').get_parameter_value().double_value,
            inference_rate_fps=self.get_parameter('yolo_inference_rate').get_parameter_value().double_value,
            half_precision=self.get_parameter('yolo_half_precision').get_parameter_value().bool_value,
        )
        self.yolo_detector = YOLODetector(yolo_cfg)
        self.fusion = PerceptionFusion()
        self.lidar_detector = LidarObstacleDetector()
        self.latest_lidar_obstacles: List[Dict[str, Any]] = []

        # ------------------- Camera Calibration State -------------------
        self.camera_info_received = False
        self.cam_fx: Optional[float] = None
        self.cam_fy: Optional[float] = None
        self.cam_cx: Optional[float] = None
        self.cam_cy: Optional[float] = None
        self.cam_width: Optional[int] = None
        self.cam_height: Optional[int] = None
        self.cam_frame_id: Optional[str] = None

        # ------------------- Performance Metrics -------------------
        self.frame_idx = 0
        self.total_received = 0
        self.total_processed = 0
        self.total_dropped = 0

        self._recent_input_times: Deque[float] = deque(maxlen=40)
        self._recent_latency_ms: Deque[float] = deque(maxlen=40)
        self._rolling_input_fps = 0.0
        self._rolling_proc_fps = 0.0
        self._rolling_latency_ms = 0.0

        # ------------------- QoS Profiles -------------------
        # Camera topics bridged from Gazebo Harmonic use Best Effort QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ------------------- Subscriptions -------------------
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile=sensor_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile=sensor_qos,
        )

        # ------------------- Publishers -------------------
        self.debug_image_pub = self.create_publisher(
            Image,
            debug_image_topic,
            qos_profile=sensor_qos,
        )
        self.segmentation_pub = self.create_publisher(
            Image,
            segmentation_topic,
            qos_profile=sensor_qos,
        )
        self.unified_image_pub = self.create_publisher(
            Image,
            '/perception/unified_image',
            qos_profile=sensor_qos,
        )
        self.yolo_image_pub = self.create_publisher(
            Image,
            '/perception/yolo/debug_image',
            qos_profile=sensor_qos,
        )
        self.yolo_detections_pub = self.create_publisher(
            String,
            '/perception/yolo/detections',
            qos_profile=10,
        )
        self.yolo_diag_pub = self.create_publisher(
            String,
            '/perception/yolo/diagnostics',
            qos_profile=10,
        )
        self.fused_image_pub = self.create_publisher(
            Image,
            '/perception/fused_image',
            qos_profile=sensor_qos,
        )
        self.fused_obstacles_pub = self.create_publisher(
            String,
            '/perception/fused_obstacles',
            qos_profile=10,
        )
        self.confidence_pub = self.create_publisher(
            String,
            '/perception/confidence',
            qos_profile=10,
        )
        self.heatmap_pub = self.create_publisher(
            Image,
            '/perception/heatmap',
            qos_profile=sensor_qos,
        )
        self.lidar_obstacles_pub = self.create_publisher(
            String,
            '/perception/lidar_obstacles',
            qos_profile=10,
        )

        # LiDAR / Radar scan subscription
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            qos_profile=sensor_qos,
        )

        # Steering & velocity subscription for dynamic trajectory overlay
        self.latest_cmd_vx = 0.15
        self.latest_cmd_wz = 0.0
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            qos_profile=10,
        )

        # ------------------- Diagnostic Timer -------------------
        diag_period = self.get_parameter('diag_publish_period_sec').get_parameter_value().double_value
        self.diag_timer = self.create_timer(diag_period, self.diagnostics_callback)

        self.get_logger().info(
            f'NAVIGUARD Perception Node initialized.\n'
            f'  Subscribing: {image_topic} (sensor_data QoS)\n'
            f'  Subscribing: {camera_info_topic} (sensor_data QoS)\n'
            f'  Publishing:  {debug_image_topic}\n'
            f'  Publishing:  {segmentation_topic}\n'
            f'  Publishing:  /perception/yolo/debug_image\n'
            f'  Publishing:  /perception/fused_image\n'
            f'  YOLO Device: {self.yolo_detector.active_device}\n'
            f'  Display Mode: {config.display_mode}'
        )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Process incoming CameraInfo message and store intrinsic calibration."""
        if not self.camera_info_received:
            self.cam_width = msg.width
            self.cam_height = msg.height
            self.cam_frame_id = msg.header.frame_id
            # Matrix K is [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            if len(msg.k) >= 6:
                self.cam_fx = float(msg.k[0])
                self.cam_cx = float(msg.k[2])
                self.cam_fy = float(msg.k[4])
                self.cam_cy = float(msg.k[5])

            self.camera_info_received = True
            self.get_logger().info(
                f'Camera calibration received on {msg.header.frame_id}:\n'
                f'  Resolution: {self.cam_width}x{self.cam_height}\n'
                f'  Intrinsics: fx={self.cam_fx:.2f}, fy={self.cam_fy:.2f}, '
                f'cx={self.cam_cx:.2f}, cy={self.cam_cy:.2f}'
            )

    def _scan_callback(self, msg: LaserScan) -> None:
        """Process incoming LiDAR / Radar scan returns and publish detected obstacles."""
        try:
            res = self.lidar_detector.process_scan(msg)
            self.latest_lidar_obstacles = res.get("obstacles", [])

            lidar_msg = String()
            lidar_msg.data = json.dumps({
                "stamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                "closest_distance_m": res.get("closest_distance_m", 99.0),
                "critical_hazard": res.get("critical_hazard", False),
                "corridor_clearance": res.get("corridor_clearance", {}),
                "obstacles": self.latest_lidar_obstacles,
            })
            self.lidar_obstacles_pub.publish(lidar_msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to process LiDAR scan: {exc}', throttle_duration_sec=2.0)

    def _cmd_vel_callback(self, msg: Twist) -> None:
        """Track commanded velocities for dynamic trajectory projection."""
        self.latest_cmd_vx = float(msg.linear.x)
        self.latest_cmd_wz = float(msg.angular.z)

    def image_callback(self, msg: Image) -> None:
        """Handle incoming image message, process, and publish debug visualization."""
        now = time.perf_counter()
        self.total_received += 1
        self._recent_input_times.append(now)

        # Calculate instantaneous input FPS
        if len(self._recent_input_times) >= 2:
            time_span = self._recent_input_times[-1] - self._recent_input_times[0]
            if time_span > 1e-4:
                self._rolling_input_fps = (len(self._recent_input_times) - 1) / time_span

        # 1. Safe cv_bridge conversion
        try:
            # Handle encoding dynamically; convert whatever valid color image to BGR for OpenCV
            if msg.encoding in ('rgb8', 'rgb16'):
                # cv_bridge handles converting rgb8 to bgr8
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif msg.encoding in ('bgr8', 'bgr16'):
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            elif msg.encoding in ('mono8', '8UC1'):
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, Exception) as exc:
            self.total_dropped += 1
            self.get_logger().warn(
                f'Failed to convert incoming image frame #{self.total_received} '
                f'(encoding={msg.encoding}): {exc}',
                throttle_duration_sec=2.0,
            )
            return

        if cv_image is None or cv_image.size == 0:
            self.total_dropped += 1
            self.get_logger().warn('Received empty image frame; skipping.', throttle_duration_sec=2.0)
            return

        # 2. Package context metadata
        self.frame_idx += 1
        metadata = {
            'frame_idx': self.frame_idx,
            'frame_id': msg.header.frame_id,
            'stamp_sec': msg.header.stamp.sec,
            'stamp_nanosec': msg.header.stamp.nanosec,
            'input_fps': self._rolling_input_fps,
            'proc_latency_ms': self._rolling_latency_ms,
            'proc_fps': self._rolling_proc_fps,
            'fx': self.cam_fx,
            'fy': self.cam_fy,
            'cx': self.cam_cx,
            'cy': self.cam_cy,
            'cmd_vx': self.latest_cmd_vx,
            'cmd_wz': self.latest_cmd_wz,
        }

        # 3. Process frame through modular perception pipeline
        try:
            processed_bgr, diag_info = self.processor.process_frame(cv_image, metadata)
            latency_ms = diag_info.get('processing_latency_ms', 0.0)
            self._recent_latency_ms.append(latency_ms)

            if len(self._recent_latency_ms) > 0:
                self._rolling_latency_ms = float(np.mean(self._recent_latency_ms))
                if self._rolling_latency_ms > 0:
                    self._rolling_proc_fps = 1000.0 / self._rolling_latency_ms

        except Exception as exc:
            self.total_dropped += 1
            self.get_logger().error(
                f'Error processing frame #{self.frame_idx}: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        # 4. Real YOLO Visual Object Detection
        detections = []
        try:
            detections, yolo_vis, yolo_diag = self.yolo_detector.detect(
                cv_image,
                timestamp=time.time(),
                frame_id=msg.header.frame_id,
            )
            if yolo_vis is not None:
                yolo_msg = self.bridge.cv2_to_imgmsg(yolo_vis, encoding='bgr8')
                yolo_msg.header.stamp = msg.header.stamp
                yolo_msg.header.frame_id = msg.header.frame_id
                self.yolo_image_pub.publish(yolo_msg)

            det_msg = String()
            det_msg.data = json.dumps({
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "frame_id": msg.header.frame_id,
                "count": len(detections),
                "detections": detections,
            })
            self.yolo_detections_pub.publish(det_msg)

            diag_msg = String()
            diag_msg.data = json.dumps(yolo_diag)
            self.yolo_diag_pub.publish(diag_msg)
        except Exception as exc:
            self.get_logger().warn(f'YOLO detection error: {exc}', throttle_duration_sec=2.0)

        # 5. Generate and publish Unified Multi-Spectral Perception View (YOLO + Segmentation + Trajectory)
        try:
            unified_bgr, unified_diag = self.processor.generate_unified_perception_view(
                cv_image, metadata, yolo_detections=detections
            )
            unified_msg = self.bridge.cv2_to_imgmsg(unified_bgr, encoding='bgr8')
            unified_msg.header.stamp = msg.header.stamp
            unified_msg.header.frame_id = msg.header.frame_id
            self.unified_image_pub.publish(unified_msg)
            # Also publish to segmentation topic for seamless backward compatibility
            self.segmentation_pub.publish(unified_msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish unified perception image: {exc}', throttle_duration_sec=2.0)

        # 5b. Generate and publish distance & proximity heatmap view
        try:
            heatmap_bgr, heatmap_diag = self.processor.generate_heatmap_view(cv_image, metadata)
            heatmap_msg = self.bridge.cv2_to_imgmsg(heatmap_bgr, encoding='bgr8')
            heatmap_msg.header.stamp = msg.header.stamp
            heatmap_msg.header.frame_id = msg.header.frame_id
            self.heatmap_pub.publish(heatmap_msg)
        except Exception as exc:
            self.get_logger().warn(f'Failed to publish heatmap image: {exc}', throttle_duration_sec=2.0)

        # 5c. Convert processed image back to ROS message preserving headers
        try:
            out_msg = self.bridge.cv2_to_imgmsg(processed_bgr, encoding='bgr8')
            out_msg.header.stamp = msg.header.stamp
            out_msg.header.frame_id = msg.header.frame_id
            self.debug_image_pub.publish(out_msg)
            self.total_processed += 1
        except Exception as exc:
            self.total_dropped += 1
            self.get_logger().warn(f'Failed to publish debug image: {exc}', throttle_duration_sec=2.0)

        # 7. Perception Fusion (Classical Traversability + YOLO Semantics + LiDAR/Radar)
        try:
            fused_vis, confirmed_obstacles, confidences = self.fusion.update(
                cv_image,
                unified_bgr if 'unified_bgr' in locals() else processed_bgr,
                detections,
                timestamp=time.time(),
                lidar_obstacles=self.latest_lidar_obstacles,
            )
            if fused_vis is not None:
                fused_msg = self.bridge.cv2_to_imgmsg(fused_vis, encoding='bgr8')
                fused_msg.header.stamp = msg.header.stamp
                fused_msg.header.frame_id = msg.header.frame_id
                self.fused_image_pub.publish(fused_msg)

            obst_msg = String()
            obst_msg.data = json.dumps({
                "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "obstacles": confirmed_obstacles,
            })
            self.fused_obstacles_pub.publish(obst_msg)

            conf_msg = String()
            conf_msg.data = json.dumps(confidences)
            self.confidence_pub.publish(conf_msg)
        except Exception as exc:
            self.get_logger().warn(f'Perception fusion error: {exc}', throttle_duration_sec=2.0)

    def diagnostics_callback(self) -> None:
        """Log concise performance diagnostics at periodic intervals."""
        if self.total_received == 0:
            self.get_logger().info('NAVIGUARD Perception: Waiting for camera frames on /camera/image_raw...')
            return

        self.get_logger().info(
            f'NAVIGUARD Perception Performance: '
            f'Total={self.total_received} | Processed={self.total_processed} | Dropped={self.total_dropped} | '
            f'Input: {self._rolling_input_fps:.1f} FPS | Latency: {self._rolling_latency_ms:.1f} ms | '
            f'Capacity: {self._rolling_proc_fps:.1f} FPS'
        )


def main(args=None) -> None:
    """Entry point for naviguard_perception_node."""
    rclpy.init(args=args)
    node = NaviguardPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
