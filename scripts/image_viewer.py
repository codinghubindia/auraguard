#!/usr/bin/env python3
"""Versatile image viewer for NAVIGUARD camera and debug topics.

Displays live stream if DISPLAY environment is present, or saves snapshots
if in headless environment.
"""

import argparse
import os
import sys
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


class ImageViewerNode(Node):
    def __init__(self, topic: str, output_path: str = None, snapshot: bool = False):
        super().__init__('naviguard_image_viewer')
        self.topic = topic
        self.output_path = output_path
        self.snapshot = snapshot
        self.bridge = CvBridge()
        self.frame_count = 0
        self.latest_frame = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.sub = self.create_subscription(Image, self.topic, self._image_callback, qos)

    def _image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_frame = cv_img
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")


def main():
    parser = argparse.ArgumentParser(description="NAVIGUARD Topic Image Viewer")
    parser.add_argument("topic", help="ROS 2 Image topic name")
    parser.add_argument("--save", "-s", default=None, help="Save path for image snapshot")
    parser.add_argument("--snapshot", action="store_true", help="Capture single frame and exit")
    args = parser.parse_args()

    rclpy.init()
    node = ImageViewerNode(args.topic, args.save, args.snapshot)

    has_display = bool(os.environ.get("DISPLAY"))
    save_path = args.save
    if not save_path and not has_display:
        clean_topic = args.topic.replace("/", "_").strip("_")
        save_path = f"/tmp/{clean_topic}_snapshot.png"

    print(f"Subscribing to: {args.topic}")
    if has_display and not args.snapshot:
        print("GUI Display detected. Opening OpenCV visualization window... (Press 'q' or ESC to exit)")
    else:
        print(f"Display not available or snapshot mode. Will save frame to: {save_path}")

    start_time = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if node.latest_frame is not None:
                if has_display and not args.snapshot:
                    cv2.imshow(f"NAVIGUARD: {args.topic}", node.latest_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
                else:
                    cv2.imwrite(save_path, node.latest_frame)
                    print(f"Successfully captured frame {node.frame_count} to: {save_path}")
                    if args.snapshot or not has_display:
                        break

            if time.time() - start_time > 10.0 and node.frame_count == 0:
                print(f"Warning: No frames received on {args.topic} within 10 seconds. Check if publisher is running.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        if has_display:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
