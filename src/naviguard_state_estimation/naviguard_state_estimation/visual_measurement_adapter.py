"""Visual measurement adapter and local scale observer for NAVIGUARD UGV.

Adapts Phase 4 geometric visual motion telemetry, transforms optical coordinates
into base_link coordinates, and evaluates local metric scale using wheel odometry constraints.

CRITICAL MONOCULAR SCALE RULE:
Visual translation from recoverPose is scale-ambiguous (unit norm ||t|| = 1).
Local metric scale is only observed through physical wheel-motion constraints when
the motion geometry is non-degenerate and directional consistency is satisfied.
"""

from typing import Any, Dict, Optional, Tuple
from diagnostic_msgs.msg import DiagnosticArray
import numpy as np

from naviguard_state_estimation.measurement_buffer import MeasurementBuffer, StampedMeasurement


class VisualAdapterConfig:
    """Gating thresholds and covariance settings for visual measurements."""

    def __init__(
        self,
        min_inliers: int = 15,
        min_wheel_disp_m: float = 0.02,
        min_wheel_speed_mps: float = 0.05,
        max_direction_error_deg: float = 45.0,
        default_yaw_rate_covariance: float = 0.05,
        default_scaled_vx_covariance: float = 0.10,
    ) -> None:
        self.min_inliers = min_inliers
        self.min_wheel_disp_m = min_wheel_disp_m
        self.min_wheel_speed_mps = min_wheel_speed_mps
        self.max_direction_error_deg = max_direction_error_deg
        self.default_yaw_rate_covariance = default_yaw_rate_covariance
        self.default_scaled_vx_covariance = default_scaled_vx_covariance


class VisualScaleObservation:
    """Result of comparing visual translation direction with wheel metric displacement."""

    def __init__(
        self,
        is_valid: bool = False,
        status: str = "UNOBSERVED",
        observed_scale_m: float = 0.0,
        wheel_disp_m: float = 0.0,
        direction_error_deg: float = 0.0,
        metric_vx_mps: float = 0.0,
    ) -> None:
        self.is_valid = is_valid
        self.status = status
        self.observed_scale_m = observed_scale_m
        self.wheel_disp_m = wheel_disp_m
        self.direction_error_deg = direction_error_deg
        self.metric_vx_mps = metric_vx_mps


class VisualMeasurementAdapter:
    """Adapts geometric visual odometry diagnostics into base_link measurements."""

    def __init__(self, config: Optional[VisualAdapterConfig] = None) -> None:
        self.config = config or VisualAdapterConfig()
        self.last_stamp_sec: Optional[float] = None
        self.last_scale_obs = VisualScaleObservation()

    def process(
        self,
        msg: DiagnosticArray,
        wheel_buffer: Optional[MeasurementBuffer] = None,
    ) -> StampedMeasurement:
        """Parse DiagnosticArray from /visual_odometry/telemetry, transform frames, and observe scale."""
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

        # Parse key-values from status
        vals: Dict[str, str] = {}
        for st in msg.status:
            for kv in st.values:
                vals[kv.key] = kv.value

        geom_is_valid = vals.get('geom_is_valid', 'false').lower() == 'true'
        geom_status = vals.get('geom_status', 'UNKNOWN')
        num_inliers = int(vals.get('geom_num_inliers', '0'))
        dt_sec = float(vals.get('dt_sec', '0.033'))

        # Optical frame values:
        # X: right, Y: down, Z: forward
        u_tx = float(vals.get('unit_tx', '0.0'))
        u_ty = float(vals.get('unit_ty', '0.0'))
        u_tz = float(vals.get('unit_tz', '0.0'))
        rot_yaw_deg = float(vals.get('rot_yaw_deg', '0.0'))
        rot_angle_deg = float(vals.get('rot_angle_deg', '0.0'))

        # 1. Coordinate Transformation: camera_optical_link -> base_link
        # In base_link: +X is forward, +Y is left, +Z is up
        # Optical +Z is forward -> base +X
        # Optical +X is right -> base -Y
        # Optical +Y is down -> base -Z
        t_base_x = u_tz
        t_base_y = -u_tx
        t_base_z = -u_ty

        # Yaw in optical frame was around optical +Y (pointing down).
        # In base_link (+Z pointing up), a positive turn around +Z corresponds to -Y_opt.
        # Thus: yaw_base = -rot_yaw_opt
        yaw_rad_base = float(np.radians(-rot_yaw_deg))
        wz_vis = yaw_rad_base / max(dt_sec, 1e-4)

        # 2. Local Scale Observation using wheel motion constraints
        t_prev = self.last_stamp_sec if self.last_stamp_sec is not None else (stamp - dt_sec)
        self.last_stamp_sec = stamp

        scale_obs = self._observe_scale(
            t_prev=t_prev,
            t_curr=stamp,
            dt_sec=dt_sec,
            geom_is_valid=geom_is_valid,
            geom_status=geom_status,
            num_inliers=num_inliers,
            t_base_x=t_base_x,
            t_base_y=t_base_y,
            wheel_buffer=wheel_buffer,
        )
        self.last_scale_obs = scale_obs

        # Overall validity check
        is_measurement_valid = True
        rejection_reason = ""

        if not geom_is_valid:
            is_measurement_valid = False
            rejection_reason = f"GEOM_{geom_status}"
        elif num_inliers < self.config.min_inliers:
            is_measurement_valid = False
            rejection_reason = f"LOW_INLIERS_{num_inliers}"
        elif abs(wz_vis) > 8.0:
            is_measurement_valid = False
            rejection_reason = f"EXCESSIVE_VISUAL_YAW_RATE_{abs(wz_vis):.2f}"

        data = {
            'wz': wz_vis,
            'wz_cov': self.config.default_yaw_rate_covariance,
            'unit_tx_base': t_base_x,
            'unit_ty_base': t_base_y,
            'unit_tz_base': t_base_z,
            'geom_status': geom_status,
            'num_inliers': num_inliers,
            'rot_angle_deg': rot_angle_deg,
            # Scale Observation Results
            'scale_status': scale_obs.status,
            'scale_valid': scale_obs.is_valid,
            'observed_scale_m': scale_obs.observed_scale_m,
            'metric_vx': scale_obs.metric_vx_mps,
            'metric_vx_cov': self.config.default_scaled_vx_covariance,
            'direction_error_deg': scale_obs.direction_error_deg,
            'wheel_disp_m': scale_obs.wheel_disp_m,
        }

        return StampedMeasurement(
            stamp_sec=stamp,
            source="visual",
            frame_id=msg.header.frame_id,
            data=data,
            is_valid=is_measurement_valid,
            rejection_reason=rejection_reason,
        )

    def _observe_scale(
        self,
        t_prev: float,
        t_curr: float,
        dt_sec: float,
        geom_is_valid: bool,
        geom_status: str,
        num_inliers: int,
        t_base_x: float,
        t_base_y: float,
        wheel_buffer: Optional[MeasurementBuffer],
    ) -> VisualScaleObservation:
        """Relate unit translation direction to metric motion using wheel displacement constraints."""
        if not geom_is_valid:
            return VisualScaleObservation(
                is_valid=False,
                status=f"REJECTED_DEGENERATE_GEOMETRY_{geom_status}",
            )

        if num_inliers < self.config.min_inliers:
            return VisualScaleObservation(
                is_valid=False,
                status=f"REJECTED_LOW_INLIERS_{num_inliers}",
            )

        if wheel_buffer is None or len(wheel_buffer) < 2:
            return VisualScaleObservation(
                is_valid=False,
                status="REJECTED_NO_WHEEL_BUFFER",
            )

        # Integrate wheel displacement over the visual interval
        wheel_disp, _, mean_vx = wheel_buffer.integrate_wheel_motion(t_prev, t_curr)

        # 1. Gating: check sufficient wheel motion
        if abs(wheel_disp) < self.config.min_wheel_disp_m or abs(mean_vx) < self.config.min_wheel_speed_mps:
            return VisualScaleObservation(
                is_valid=False,
                status="REJECTED_ZERO_WHEEL_MOTION",
                wheel_disp_m=wheel_disp,
            )

        # 2. Directional Consistency check
        # For forward motion (+X), visual t_base_x must be positive
        norm_xy = np.hypot(t_base_x, t_base_y)
        if norm_xy < 1e-4:
            return VisualScaleObservation(
                is_valid=False,
                status="REJECTED_NEAR_ZERO_VISUAL_XY",
            )

        # Angle of visual translation vector relative to robot forward (+X)
        angle_rad = abs(np.arctan2(t_base_y, t_base_x))
        angle_deg = float(np.degrees(angle_rad))

        if angle_deg > self.config.max_direction_error_deg:
            return VisualScaleObservation(
                is_valid=False,
                status=f"REJECTED_INCONSISTENT_DIRECTION_{angle_deg:.1f}deg",
                wheel_disp_m=wheel_disp,
                direction_error_deg=angle_deg,
            )

        # 3. Compute observed local scale and metric velocity
        observed_scale = float(abs(wheel_disp) / norm_xy)
        metric_vx = float((observed_scale * t_base_x) / max(dt_sec, 1e-4))

        return VisualScaleObservation(
            is_valid=True,
            status="LOCAL_SCALE_OBSERVED",
            observed_scale_m=observed_scale,
            wheel_disp_m=wheel_disp,
            direction_error_deg=angle_deg,
            metric_vx_mps=metric_vx,
        )
