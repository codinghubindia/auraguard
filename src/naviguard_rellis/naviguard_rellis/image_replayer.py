"""
RELLIS-3D Dataset Image Replayer Node for NAVIGUARD.

Replays synchronized camera images and camera_info from the RELLIS-3D dataset,
and publishes ground-truth poses STRICTLY to the /evaluation/* namespace.
Includes deterministic synthetic fallback when external dataset is absent.
"""

import os
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from naviguard_rellis.ground_truth_guard import GroundTruthGuard
from naviguard_rellis.sequence_loader import SequenceLoader
from naviguard_rellis.calibration_loader import create_camera_info_msg


def generate_synthetic_outdoor_frame(frame_idx: int, width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates a synthetic off-road outdoor image simulating RELLIS-3D trails:
    dirt path in center, green grass/vegetation on sides, blue sky with horizon,
    and visual features for feature tracking.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky (top 40%)
    horizon_y = int(height * 0.45)
    for y in range(horizon_y):
        ratio = y / max(1, horizon_y)
        # Sky gradient from deep blue to hazy horizon
        b = int(220 - 40 * ratio)
        g = int(180 - 30 * ratio)
        r = int(120 - 20 * ratio)
        img[y, :] = (b, g, r)

    # Ground (bottom 55%)
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / max(1, height - horizon_y)
        # Grass backdrop
        b = int(25 + 15 * ratio)
        g = int(90 + 35 * ratio)
        r = int(35 + 20 * ratio)
        img[y, :] = (b, g, r)

    # Dirt trail in center with perspective
    # Trail widens toward bottom
    center_shift = int(20.0 * math.sin(frame_idx * 0.05))
    top_center = width // 2 + center_shift
    bottom_center = width // 2 + int(40.0 * math.sin(frame_idx * 0.05))

    top_w = 40
    bottom_w = int(width * 0.65)

    trail_pts = np.array([
        [top_center - top_w // 2, horizon_y],
        [top_center + top_w // 2, horizon_y],
        [bottom_center + bottom_w // 2, height],
        [bottom_center - bottom_w // 2, height]
    ], dtype=np.int32)
    # Dirt color (BGR: 30, 70, 110)
    cv2.fillPoly(img, [trail_pts], (40, 85, 135))

    # Add tree/vegetation landmarks along trail borders
    np.random.seed(frame_idx % 1000 + 42)
    for _ in range(25):
        tx = np.random.randint(20, width - 20)
        ty = np.random.randint(horizon_y - 30, height - 30)
        # Only draw if outside main trail center
        trail_x_at_y = top_center + (bottom_center - top_center) * ((ty - horizon_y) / max(1, height - horizon_y))
        w_at_y = top_w + (bottom_w - top_w) * ((ty - horizon_y) / max(1, height - horizon_y))
        if abs(tx - trail_x_at_y) > (w_at_y // 2 + 10):
            radius = np.random.randint(4, 12)
            color = (np.random.randint(15, 40), np.random.randint(80, 160), np.random.randint(20, 60))
            cv2.circle(img, (tx, ty), radius, color, -1)

    # Add simulated motion texture / grain
    noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Frame counter watermark
    cv2.putText(img_noisy, f"RELLIS SIM Frame: {frame_idx:04d}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img_noisy


class RellisImageReplayerNode(Node):
    """Replays RELLIS-3D camera stream and ground truth for offline benchmarking."""

    def __init__(self):
        super().__init__('rellis_replay_node')

        # Declare parameters
        from rcl_interfaces.msg import ParameterDescriptor
        dyn_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('data_root', '~/datasets/rellis3d')
        self.declare_parameter('sequence', '00000', dyn_param)
        self.declare_parameter('replay_rate_hz', 10.0)
        self.declare_parameter('loop', False)
        self.declare_parameter('camera_frame_id', 'camera_optical_link')
        self.declare_parameter('use_synthetic_fallback', True)

        self.data_root = os.path.expanduser(self.get_parameter('data_root').value)
        raw_seq = self.get_parameter('sequence').value
        self.sequence = f"{int(raw_seq):05d}" if str(raw_seq).isdigit() else str(raw_seq)
        self.replay_rate = float(self.get_parameter('replay_rate_hz').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.camera_frame_id = str(self.get_parameter('camera_frame_id').value)
        self.use_synthetic_fallback = bool(self.get_parameter('use_synthetic_fallback').value)

        self.bridge = CvBridge()
        self.current_idx = 0
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = 'odom'

        # Verify ground truth topics with safety guard
        gt_pose_topic = '/evaluation/ground_truth_pose'
        gt_path_topic = '/evaluation/ground_truth_path'
        GroundTruthGuard.assert_topic_allowed(gt_pose_topic)
        GroundTruthGuard.assert_topic_allowed(gt_path_topic)

        # Standard camera publishers for NAVIGUARD pipeline
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.cam_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)

        # Isolated evaluation publishers
        self.gt_pose_pub = self.create_publisher(PoseStamped, gt_pose_topic, 10)
        self.gt_path_pub = self.create_publisher(Path, gt_path_topic, 10)
        self.active_pub = self.create_publisher(Bool, '/evaluation/status', 10)

        # Attempt to load sequence
        self.loader = SequenceLoader(self.data_root, self.sequence, fps=self.replay_rate)
        if self.loader.is_loaded:
            self.get_logger().info(
                f"Loaded RELLIS-3D sequence {self.sequence} with {len(self.loader)} frames from {self.data_root}"
            )
            self.total_frames = len(self.loader)
            self.mode = "dataset"
        elif self.use_synthetic_fallback:
            self.get_logger().warn(
                f"RELLIS-3D dataset not found at '{os.path.join(self.data_root, self.sequence)}'. "
                f"Using realistic synthetic off-road fallback frames (100 frames)."
            )
            self.total_frames = 100
            self.mode = "synthetic"
        else:
            self.get_logger().error(f"Failed to load dataset and synthetic fallback is disabled!")
            self.total_frames = 0
            self.mode = "empty"

        # Timer for replay loop
        timer_period = 1.0 / max(0.1, self.replay_rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"RELLIS replay node initialized at {self.replay_rate:.1f} Hz (Mode: {self.mode})")

    def timer_callback(self):
        if self.mode == "empty" or self.total_frames == 0:
            return

        if self.current_idx >= self.total_frames:
            if self.loop:
                self.current_idx = 0
                self.gt_path_msg.poses.clear()
                self.get_logger().info("Replay reached end. Looping to start.")
            else:
                status_msg = Bool()
                status_msg.data = False
                self.active_pub.publish(status_msg)
                return

        now = self.get_clock().now().to_msg()

        # Generate or load image
        if self.mode == "dataset":
            frame_data = self.loader.get_frame(self.current_idx)
            if frame_data and os.path.isfile(frame_data.image_path):
                img_bgr = cv2.imread(frame_data.image_path)
            else:
                img_bgr = generate_synthetic_outdoor_frame(self.current_idx)

            calib_params = frame_data.camera_info_params if frame_data else {}
            gt_pose_dict = frame_data.gt_pose if frame_data else None
        else:
            img_bgr = generate_synthetic_outdoor_frame(self.current_idx)
            calib_params = {
                'width': 640,
                'height': 480,
                'k': np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
                'd': [0.0, 0.0, 0.0, 0.0, 0.0]
            }
            # Synthetic ground truth motion (forward along X, slight curvature along Y)
            dist_x = self.current_idx * 0.1
            dist_y = 0.5 * math.sin(self.current_idx * 0.05)
            gt_pose_dict = {
                'position': (dist_x, dist_y, 0.0),
                'orientation': (0.0, 0.0, 0.0, 1.0)
            }

        # Publish Image
        img_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.camera_frame_id
        self.image_pub.publish(img_msg)

        # Publish CameraInfo
        info_msg = create_camera_info_msg(calib_params, frame_id=self.camera_frame_id, timestamp=now)
        self.cam_info_pub.publish(info_msg)

        # Publish Ground Truth (strictly on /evaluation/*)
        if gt_pose_dict:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(gt_pose_dict['position'][0])
            ps.pose.position.y = float(gt_pose_dict['position'][1])
            ps.pose.position.z = float(gt_pose_dict['position'][2])
            ps.pose.orientation.x = float(gt_pose_dict['orientation'][0])
            ps.pose.orientation.y = float(gt_pose_dict['orientation'][1])
            ps.pose.orientation.z = float(gt_pose_dict['orientation'][2])
            ps.pose.orientation.w = float(gt_pose_dict['orientation'][3])

            self.gt_pose_pub.publish(ps)

            self.gt_path_msg.header.stamp = now
            self.gt_path_msg.poses.append(ps)
            if len(self.gt_path_msg.poses) > 1000:
                self.gt_path_msg.poses.pop(0)
            self.gt_path_pub.publish(self.gt_path_msg)

        status_msg = Bool()
        status_msg.data = True
        self.active_pub.publish(status_msg)

        self.current_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = RellisImageReplayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
