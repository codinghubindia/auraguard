"""IMU message processing and measurement validation for NAVIGUARD UGV.

Extracts angular velocities and accelerations, enforces gating checks,
handles 6-axis unoriented IMUs without fabricating orientation, and maps into base_link coordinates.
"""

from typing import Optional, Tuple
import numpy as np
from sensor_msgs.msg import Imu

from naviguard_state_estimation.measurement_buffer import StampedMeasurement


class ImuProcessorConfig:
    """Configuration parameters for IMU gating and covariance."""

    def __init__(
        self,
        max_angular_velocity: float = 8.0,
        max_linear_acceleration: float = 40.0,
        default_wz_covariance: float = 0.005,
        default_ax_covariance: float = 0.04,
    ) -> None:
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_acceleration = max_linear_acceleration
        self.default_wz_covariance = default_wz_covariance
        self.default_ax_covariance = default_ax_covariance


class ImuProcessor:
    """Processes incoming IMU messages, performs gating, and extracts base_link rates."""

    def __init__(self, config: Optional[ImuProcessorConfig] = None) -> None:
        self.config = config or ImuProcessorConfig()
        self.last_stamp_sec: Optional[float] = None

    def process(self, msg: Imu) -> StampedMeasurement:
        """Validate and extract IMU measurement into base_link frame.

        Note: URDF specifies imu_link is rigidly aligned with base_link (rpy = 0, 0, 0).
        Thus [wx, wy, wz]_imu directly maps to [wx, wy, wz]_base.
        """
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Check timestamp validity
        if stamp <= 0.0 or np.isnan(stamp) or np.isinf(stamp):
            return StampedMeasurement(
                stamp_sec=0.0,
                source="imu",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="INVALID_TIMESTAMP",
            )

        # Monotonicity check
        if self.last_stamp_sec is not None and stamp < self.last_stamp_sec:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="imu",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="NON_MONOTONIC_TIMESTAMP",
            )
        self.last_stamp_sec = stamp

        # Extract angular velocity
        wx, wy, wz = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        if any(np.isnan(v) or np.isinf(v) for v in [wx, wy, wz]):
            return StampedMeasurement(
                stamp_sec=stamp,
                source="imu",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="NAN_OR_INF_ANGULAR_VELOCITY",
            )

        # Extract linear acceleration
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        if any(np.isnan(v) or np.isinf(v) for v in [ax, ay, az]):
            return StampedMeasurement(
                stamp_sec=stamp,
                source="imu",
                frame_id=msg.header.frame_id,
                data={},
                is_valid=False,
                rejection_reason="NAN_OR_INF_LINEAR_ACCELERATION",
            )

        # Gating: check maximum magnitude thresholds
        w_norm = float(np.hypot(np.hypot(wx, wy), wz))
        if w_norm > self.config.max_angular_velocity:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="imu",
                frame_id=msg.header.frame_id,
                data={'wz': wz, 'ax': ax},
                is_valid=False,
                rejection_reason=f"ANGULAR_VELOCITY_EXCEEDS_LIMIT_{w_norm:.2f}",
            )

        a_norm = float(np.hypot(np.hypot(ax, ay), az))
        if a_norm > self.config.max_linear_acceleration:
            return StampedMeasurement(
                stamp_sec=stamp,
                source="imu",
                frame_id=msg.header.frame_id,
                data={'wz': wz, 'ax': ax},
                is_valid=False,
                rejection_reason=f"LINEAR_ACCELERATION_EXCEEDS_LIMIT_{a_norm:.2f}",
            )

        # Covariance handling
        wz_cov = float(msg.angular_velocity_covariance[8])
        if wz_cov <= 0.0 or np.isnan(wz_cov):
            wz_cov = self.config.default_wz_covariance

        ax_cov = float(msg.linear_acceleration_covariance[0])
        if ax_cov <= 0.0 or np.isnan(ax_cov):
            ax_cov = self.config.default_ax_covariance

        # Explicit orientation check: raw 6-axis IMU without orientation
        has_orientation = (
            msg.orientation_covariance[0] != -1.0
            and abs(np.linalg.norm([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]) - 1.0) < 1e-2
        )

        data = {
            'wx': float(wx),
            'wy': float(wy),
            'wz': float(wz),  # in base_link coordinates
            'ax': float(ax),
            'ay': float(ay),
            'az': float(az),
            'wz_cov': wz_cov,
            'ax_cov': ax_cov,
            'has_orientation': has_orientation,
        }

        return StampedMeasurement(
            stamp_sec=stamp,
            source="imu",
            frame_id=msg.header.frame_id,
            data=data,
            is_valid=True,
            rejection_reason="",
        )
