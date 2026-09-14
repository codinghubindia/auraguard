"""
RELLIS-3D Evaluator Node for NAVIGUARD.

Subscribes to estimated SLAM poses and ground-truth evaluation poses,
computes online trajectory metrics (ATE RMSE, RPE, tracking continuity),
and logs benchmarking results without leaking ground truth to navigation.
"""

import os
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray

from naviguard_rellis.ground_truth_guard import GroundTruthGuard
from naviguard_rellis.evaluation_recorder import EvaluationRecorder


class RellisEvaluatorNode(Node):
    """Computes online and offline benchmarking metrics for SLAM against ground truth."""

    def __init__(self):
        super().__init__('rellis_evaluator_node')

        from rcl_interfaces.msg import ParameterDescriptor
        dyn_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('sequence', '00000', dyn_param)
        self.declare_parameter('metrics_output_file', 'rellis_evaluation_results.json')
        self.declare_parameter('publish_rate_hz', 1.0)

        raw_seq = self.get_parameter('sequence').value
        self.sequence = f"{int(raw_seq):05d}" if str(raw_seq).isdigit() else str(raw_seq)
        self.output_file = os.path.expanduser(str(self.get_parameter('metrics_output_file').value))
        self.pub_rate = float(self.get_parameter('publish_rate_hz').value)

        self.recorder = EvaluationRecorder(sequence_name=self.sequence)

        # Paths for RViz visualization
        self.gt_path = Path()
        self.gt_path.header.frame_id = 'odom'
        self.est_path = Path()
        self.est_path.header.frame_id = 'odom'

        # Publishers (strictly /evaluation/*)
        GroundTruthGuard.assert_topic_allowed('/evaluation/ground_truth_trajectory')
        GroundTruthGuard.assert_topic_allowed('/evaluation/estimated_trajectory')
        GroundTruthGuard.assert_topic_allowed('/evaluation/metrics')

        self.gt_path_pub = self.create_publisher(Path, '/evaluation/ground_truth_trajectory', 10)
        self.est_path_pub = self.create_publisher(Path, '/evaluation/estimated_trajectory', 10)
        self.metrics_pub = self.create_publisher(String, '/evaluation/metrics', 10)

        # Subscriptions
        # 1. Ground truth pose from replayer
        self.create_subscription(
            PoseStamped,
            '/evaluation/ground_truth_pose',
            self.gt_pose_callback,
            10
        )

        # 2. Estimated pose from SLAM
        self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.est_pose_callback,
            10
        )

        # 3. Confidence metrics
        self.create_subscription(
            DiagnosticArray,
            '/confidence/metrics',
            self.confidence_callback,
            10
        )

        timer_period = 1.0 / max(0.1, self.pub_rate)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"RELLIS Evaluator Node initialized for Sequence '{self.sequence}'. "
            f"Output file: {self.output_file}"
        )

    def gt_pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.recorder.add_ground_truth(t, x, y, z)

        self.gt_path.header.stamp = msg.header.stamp
        self.gt_path.poses.append(msg)
        if len(self.gt_path.poses) > 1500:
            self.gt_path.poses.pop(0)

    def est_pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.recorder.add_estimate(t, x, y, z)
        self.recorder.tracking_active_count += 1
        self.recorder.total_frames_evaluated += 1

        self.est_path.header.stamp = msg.header.stamp
        self.est_path.poses.append(msg)
        if len(self.est_path.poses) > 1500:
            self.est_path.poses.pop(0)

    def confidence_callback(self, msg: DiagnosticArray):
        score = 1.0
        decision = "CONTINUE"
        for status in msg.status:
            for kv in status.values:
                if kv.key == "overall_confidence":
                    try:
                        score = float(kv.value)
                    except ValueError:
                        pass
                elif kv.key == "decision":
                    decision = kv.value
        now_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.recorder.record_confidence(now_t, score, decision)

    def timer_callback(self):
        # Publish trajectories
        if self.gt_path.poses:
            self.gt_path_pub.publish(self.gt_path)
        if self.est_path.poses:
            self.est_path_pub.publish(self.est_path)

        metrics = self.recorder.get_metrics()
        metrics_str = json.dumps(metrics)

        msg = String()
        msg.data = metrics_str
        self.metrics_pub.publish(msg)

        if metrics['est_samples'] > 0 and metrics['est_samples'] % 20 == 0:
            self.get_logger().info(
                f"[RELLIS EVAL] Samples: {metrics['est_samples']} | "
                f"ATE RMSE: {metrics['ate_rmse_m']:.3f}m | "
                f"RPE: {metrics['rpe_rmse_m']:.3f}m | "
                f"Continuity: {metrics['tracking_continuity_pct']:.1f}%"
            )

    def save_results(self):
        try:
            self.recorder.save_json(self.output_file)
            self.get_logger().info(f"Saved RELLIS evaluation metrics to {self.output_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save metrics to {self.output_file}: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RellisEvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_results()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
