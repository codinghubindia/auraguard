"""Wheel odometry processing and validity gating for NAVIGUARD UGV.

Extracts linear and angular velocities from /odom, validates coordinate frames,
enforces kinematic plausibility checks, and extracts measurement covariance.
"""

from typing import Optional, Tuple
from nav_msgs.msg import Odometry
import numpy as np

from naviguard_state_estimation.measurement_buffer import StampedMeasurement


class WheelOdomProcessorConfig:
    """Configuration thresholds for wheel odometry validation."""

    def __init__(
        self,
        expected_header_frame: str = "odom",
        expected_child_frame: str = "base_link",
        max_linear_velocity: float = 3.0,
        max_angular_velocity: float = 5.0,
        max_lateral_velocity: float = 0.5,
        default_vx_covariance: float = 0.02,
        default_wz_covariance: float = 0.04,
    ) -> None:
        self.expected_header_frame = expected_header_frame
        self.expected_child_frame = expected_child_frame
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_lateral_velocity = max_lateral_velocity
        self.default_vx_covariance = default_vx_covariance
        self.default_wz_covariance = default_wz_covariance


class WheelOdomProcessor:
    """Processes incoming Odometry messages and gates unphysical or corrupted data."""

    def __init__(self, config: Optional[WheelOdomProcessorConfig] = None) -> None:
        self.config = config or WheelOdomProcessorConfig()
        self.last_stamp_sec: Optional[float] = None

    def process(self, msg: Odometry) -> StampedMeasurement:
        """Validate and extract metric translational and angular velocity."""
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Timestamp validity
        if stamp <= 0.0 or np.isnan(stamp) or np.isinf(stamp):
            return StampedMeasurement(
                stamp_sec=0.0,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="INVALID_TIMESTAMP",
            )

        # Monotonicity check
        if self.last_stamp_sec is not None and stamp < self.last_stamp_sec:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="NON_MONOTONIC_TIMESTAMP",
            )
        self.last_stamp_sec = stamp

        # Frame consistency
        if (
            msg.header.frame_id != self.config.expected_header_frame
            or msg.child_frame_id != self.config.expected_child_frame
        ):
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason=f"FRAME_MISMATCH_EXPECTED_{self.config.expected_header_frame}_TO_{self.config.expected_child_frame}",
            )

        # Extract velocities
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        if any(np.isnan(v) or np.isinf(v) for v in [vx, vy, wz]):
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="NAN_OR_INF_VELOCITY",
            )

        # Gating: Velocity plausibility
        if abs(vx) > self.config.max_linear_velocity:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={'vx': vx, 'wz': wz},
                is_valid=False,
                rejection_reason=f"LINEAR_VELOCITY_EXCEEDS_LIMIT_{abs(vx):.2f}",
            )

        if abs(vy) > self.config.max_lateral_velocity:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={'vx': vx, 'wz': wz},
                is_valid=False,
                rejection_reason=f"LATERAL_VELOCITY_EXCEEDS_NON_HOLONOMIC_LIMIT_{abs(vy):.2f}",
            )

        if abs(wz) > self.config.max_angular_velocity:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="wheel_odom",
                frame_id=msg.header.frame_id,
                data={'vx': vx, 'wz': wz},
                is_valid=False,
                rejection_reason=f"ANGULAR_VELOCITY_EXCEEDS_LIMIT_{abs(wz):.2f}",
            )

        # Extract pose in odom frame
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        pz = msg.pose.pose.position.z
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Compute 2D yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        # Covariance extraction
        vx_cov = float(msg.twist.covariance[0])
        if vx_cov <= 0.0 or np.isnan(vx_cov):
            vx_cov = self.config.default_vx_covariance

        wz_cov = float(msg.twist.covariance[35])
        if wz_cov <= 0.0 or np.isnan(wz_cov):
            wz_cov = self.config.default_wz_covariance

        data = {
            'vx': float(vx),
            'vy': float(vy),
            'wz': float(wz),
            'pos_x': float(px),
            'pos_y': float(py),
            'yaw': yaw,
            'vx_cov': vx_cov,
            'wz_cov': wz_cov,
        }

        return StampedMeasurement(
            stamp_sec=stamp,
            source="wheel_odom",
            frame_id=msg.header.frame_id,
            data=data,
            is_valid=True,
            rejection_reason="",
        )
