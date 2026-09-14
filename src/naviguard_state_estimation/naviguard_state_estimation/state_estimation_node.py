"""ROS 2 Node for NAVIGUARD UGV Multi-Sensor Metric State Estimation.

Fuses wheel odometry, IMU, and visual motion telemetry using an Extended Kalman Filter,
computes cross-sensor consistency residuals, performs local scale gating,
and publishes state estimation odometry and detailed diagnostics.
"""

from typing import Any, Dict, Optional
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
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped, Quaternion
import tf2_ros

from naviguard_state_estimation.measurement_buffer import MeasurementBuffer
from naviguard_state_estimation.imu_processor import ImuProcessor, ImuProcessorConfig
from naviguard_state_estimation.wheel_odom_processor import WheelOdomProcessor, WheelOdomProcessorConfig
from naviguard_state_estimation.visual_measurement_adapter import VisualMeasurementAdapter, VisualAdapterConfig
from naviguard_state_estimation.estimator import NaviguardStateEstimator, EstimatorConfig
from naviguard_state_estimation.consistency_checker import CrossSensorConsistencyChecker, ConsistencyConfig
from naviguard_state_estimation.diagnostics import StateEstimationDiagnostics, TrajectoryLogger


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert a planar yaw angle in radians to a geometry_msgs/Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(np.sin(yaw * 0.5))
    q.w = float(np.cos(yaw * 0.5))
    return q


class StateEstimationNode(Node):
    """Fuses visual relative motion, IMU, and wheel odometry into a metric state estimate."""

    def __init__(self) -> None:
        super().__init__('state_estimation_node')

        # 1. Declare Parameters
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('enable_visual', True)
        self.declare_parameter('enable_imu', True)
        self.declare_parameter('enable_wheel_odom', True)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('log_trajectory_csv', '')

        # Filter process noise parameters
        self.declare_parameter('q_pos', 0.01)
        self.declare_parameter('q_theta', 0.005)
        self.declare_parameter('q_vx', 0.08)
        self.declare_parameter('q_wz', 0.05)
        self.declare_parameter('mahalanobis_gate_1d', 9.0)
        self.declare_parameter('mahalanobis_gate_2d', 13.8)

        # Visual scale gating parameters
        self.declare_parameter('min_inliers', 15)
        self.declare_parameter('min_wheel_disp_m', 0.02)
        self.declare_parameter('min_wheel_speed_mps', 0.05)
        self.declare_parameter('max_direction_error_deg', 45.0)

        # 2. Retrieve Parameter Values
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.enable_visual = bool(self.get_parameter('enable_visual').value)
        self.enable_imu = bool(self.get_parameter('enable_imu').value)
        self.enable_wheel_odom = bool(self.get_parameter('enable_wheel_odom').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        csv_path = str(self.get_parameter('log_trajectory_csv').value)

        # 3. Initialize Processors and Buffers
        est_config = EstimatorConfig(
            q_pos=float(self.get_parameter('q_pos').value),
            q_theta=float(self.get_parameter('q_theta').value),
            q_vx=float(self.get_parameter('q_vx').value),
            q_wz=float(self.get_parameter('q_wz').value),
            mahalanobis_gate_1d=float(self.get_parameter('mahalanobis_gate_1d').value),
            mahalanobis_gate_2d=float(self.get_parameter('mahalanobis_gate_2d').value),
        )
        self.estimator = NaviguardStateEstimator(est_config)

        self.imu_buffer = MeasurementBuffer(max_size=300, max_age_sec=5.0)
        self.wheel_buffer = MeasurementBuffer(max_size=300, max_age_sec=5.0)
        self.visual_buffer = MeasurementBuffer(max_size=100, max_age_sec=5.0)

        self.imu_processor = ImuProcessor()
        self.wheel_processor = WheelOdomProcessor(
            WheelOdomProcessorConfig(
                expected_header_frame=self.odom_frame,
                expected_child_frame=self.base_frame,
            )
        )
        self.visual_adapter = VisualMeasurementAdapter(
            VisualAdapterConfig(
                min_inliers=int(self.get_parameter('min_inliers').value),
                min_wheel_disp_m=float(self.get_parameter('min_wheel_disp_m').value),
                min_wheel_speed_mps=float(self.get_parameter('min_wheel_speed_mps').value),
                max_direction_error_deg=float(self.get_parameter('max_direction_error_deg').value),
            )
        )
        self.consistency_checker = CrossSensorConsistencyChecker()
        self.diagnostics_builder = StateEstimationDiagnostics(hardware_id="naviguard_ugv")
        self.trajectory_logger = TrajectoryLogger(csv_path) if csv_path else None

        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        else:
            self.tf_broadcaster = None

        # 4. Setup Publishers
        self.odom_pub = self.create_publisher(Odometry, '/state_estimation/odom', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/state_estimation/diagnostics', 10)

        # 5. Setup Subscriptions
        # IMU: Best Effort
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            qos_profile_sensor_data,
        )

        # Wheel Odom: Reliable
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.wheel_odom_callback,
            reliable_qos,
        )

        # Visual VO Diagnostics: Reliable
        self.visual_sub = self.create_subscription(
            DiagnosticArray,
            '/visual_odometry/telemetry',
            self.visual_callback,
            reliable_qos,
        )

        # 6. Timer for Periodic State and Diagnostics Publishing
        timer_period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"StateEstimationNode started. Publishing at {self.publish_rate_hz} Hz. "
            f"publish_tf={self.publish_tf}, odom_frame={self.odom_frame}, base_frame={self.base_frame}"
        )

    def imu_callback(self, msg: Imu) -> None:
        """Handle incoming IMU measurement."""
        if not self.enable_imu:
            return

        meas = self.imu_processor.process(msg)
        self.imu_buffer.add(meas)

        if not meas.is_valid:
            self.estimator.rejections_count['imu'] += 1
            self.estimator.last_rejection_reason['imu'] = meas.rejection_reason
            return

        # Consistency check against latest wheel
        latest_wheel = self.wheel_buffer.get_latest()
        self.consistency_checker.check_wheel_vs_imu(latest_wheel, meas)

        # Update EKF with IMU yaw rate
        wz = float(meas.data['wz'])
        wz_cov = float(meas.data['wz_cov'])
        self.estimator.update_imu(wz=wz, wz_cov=wz_cov, stamp_sec=meas.stamp_sec)

    def wheel_odom_callback(self, msg: Odometry) -> None:
        """Handle incoming wheel odometry measurement."""
        if not self.enable_wheel_odom:
            return

        meas = self.wheel_processor.process(msg)
        self.wheel_buffer.add(meas)

        if not meas.is_valid:
            self.estimator.rejections_count['wheel'] += 1
            self.estimator.last_rejection_reason['wheel'] = meas.rejection_reason
            return

        # Initialize EKF pose from wheel odometry on initial startup
        if not self.estimator.is_initialized:
            self.estimator.initialize_pose(
                x=float(meas.data['pos_x']),
                y=float(meas.data['pos_y']),
                theta=float(meas.data['yaw']),
                stamp_sec=meas.stamp_sec,
            )
            self.get_logger().info(
                f"Initialized state estimator pose to x={meas.data['pos_x']:.3f}, "
                f"y={meas.data['pos_y']:.3f}, theta={np.degrees(meas.data['yaw']):.1f} deg"
            )

        # Consistency check against latest IMU
        latest_imu = self.imu_buffer.get_latest()
        self.consistency_checker.check_wheel_vs_imu(meas, latest_imu)

        # Update EKF with wheel metric velocities
        vx = float(meas.data['vx'])
        wz = float(meas.data['wz'])
        vx_cov = float(meas.data['vx_cov'])
        wz_cov = float(meas.data['wz_cov'])
        self.estimator.update_wheel_odometry(
            vx=vx,
            wz=wz,
            vx_cov=vx_cov,
            wz_cov=wz_cov,
            stamp_sec=meas.stamp_sec,
        )

    def visual_callback(self, msg: DiagnosticArray) -> None:
        """Handle incoming visual odometry telemetry."""
        if not self.enable_visual:
            return

        meas = self.visual_adapter.process(msg, wheel_buffer=self.wheel_buffer)
        self.visual_buffer.add(meas)

        if not meas.is_valid:
            self.estimator.rejections_count['visual'] += 1
            self.estimator.last_rejection_reason['visual'] = meas.rejection_reason
            return

        # Consistency check against wheel and IMU
        latest_wheel = self.wheel_buffer.get_latest()
        latest_imu = self.imu_buffer.get_latest()
        self.consistency_checker.check_visual_vs_others(meas, latest_wheel, latest_imu)

        wz_vis = float(meas.data['wz'])
        wz_cov = float(meas.data['wz_cov'])
        scale_valid = bool(meas.data['scale_valid'])

        if scale_valid:
            metric_vx = float(meas.data['metric_vx'])
            vx_cov = float(meas.data['metric_vx_cov'])
            self.estimator.update_visual(
                wz_vis=wz_vis,
                wz_cov=wz_cov,
                stamp_sec=meas.stamp_sec,
                metric_vx=metric_vx,
                vx_cov=vx_cov,
            )
        else:
            # Monocular VO has no intrinsic metric scale; update only yaw rate
            self.estimator.update_visual(
                wz_vis=wz_vis,
                wz_cov=wz_cov,
                stamp_sec=meas.stamp_sec,
            )

    def timer_callback(self) -> None:
        """Periodic timer callback to propagate state, publish odometry, and publish diagnostics."""
        now_time = self.get_clock().now()
        now_sec = float(now_time.nanoseconds) * 1e-9

        # Predict state forward to current ROS time
        if self.estimator.is_initialized:
            self.estimator.predict_to_time(now_sec)

        state = self.estimator.state

        # 1. Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = now_time.to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame

        # Position and orientation
        odom_msg.pose.pose.position.x = state.x
        odom_msg.pose.pose.position.y = state.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = yaw_to_quaternion(state.theta)

        # 6x6 Pose Covariance mapping from 5x5 EKF covariance
        # indices: [x=0, y=7, z=14, roll=21, pitch=28, yaw=35]
        pose_cov = [0.0] * 36
        pose_cov[0] = state.cov[0, 0]    # var(x)
        pose_cov[1] = state.cov[0, 1]    # cov(x, y)
        pose_cov[6] = state.cov[1, 0]    # cov(y, x)
        pose_cov[7] = state.cov[1, 1]    # var(y)
        pose_cov[14] = 1e-4             # var(z)
        pose_cov[21] = 1e-4             # var(roll)
        pose_cov[28] = 1e-4             # var(pitch)
        pose_cov[35] = state.cov[2, 2]   # var(yaw)
        odom_msg.pose.covariance = pose_cov

        # Velocities
        odom_msg.twist.twist.linear.x = state.vx
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = state.wz

        # 6x6 Twist Covariance mapping
        # indices: [vx=0, vy=7, vz=14, wx=21, wy=28, wz=35]
        twist_cov = [0.0] * 36
        twist_cov[0] = state.cov[3, 3]   # var(vx)
        twist_cov[7] = 1e-4             # var(vy) non-holonomic constraint
        twist_cov[14] = 1e-4            # var(vz)
        twist_cov[21] = 1e-4            # var(wx)
        twist_cov[28] = 1e-4            # var(wy)
        twist_cov[35] = state.cov[4, 4]  # var(wz)
        odom_msg.twist.covariance = twist_cov

        self.odom_pub.publish(odom_msg)

        # 2. Publish Diagnostics
        scale_data = {}
        latest_vis = self.visual_buffer.get_latest()
        if latest_vis and latest_vis.is_valid:
            scale_data = latest_vis.data

        diag_msg = self.diagnostics_builder.build_diagnostic_array(
            stamp_sec=now_sec,
            state=state,
            updates_count=self.estimator.updates_count,
            rejections_count=self.estimator.rejections_count,
            last_rejection_reason=self.estimator.last_rejection_reason,
            scale_data=scale_data,
            consistency=self.consistency_checker.metrics,
        )
        self.diag_pub.publish(diag_msg)

        # 3. Optional Trajectory CSV Logging
        if self.trajectory_logger is not None:
            self.trajectory_logger.log_row(
                stamp_sec=now_sec,
                state=state,
                scale_data=scale_data,
                consistency=self.consistency_checker.metrics,
            )

        # 4. Optional TF Broadcast (default disabled to prevent duplicate odom->base_link TF)
        if self.publish_tf and self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = now_time.to_msg()
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = state.x
            t.transform.translation.y = state.y
            t.transform.translation.z = 0.0
            t.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)

    def destroy_node(self) -> bool:
        """Clean up resources on node shutdown."""
        if self.trajectory_logger is not None:
            self.trajectory_logger.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
