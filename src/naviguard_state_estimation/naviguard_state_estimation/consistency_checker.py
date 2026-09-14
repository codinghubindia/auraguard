"""Cross-sensor consistency checker for NAVIGUARD UGV.

Computes and tracks physical residuals between redundant sensor observations:
- Visual vs Wheel: yaw rate residual and motion direction alignment
- Visual vs IMU: yaw rate residual
- Wheel vs IMU: yaw rate residual and linear acceleration vs velocity derivative
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from naviguard_state_estimation.measurement_buffer import StampedMeasurement


class ConsistencyConfig:
    """Configurable thresholds for cross-sensor consistency evaluation."""

    def __init__(
        self,
        max_yaw_rate_residual_rad_s: float = 0.5,
        max_direction_residual_deg: float = 45.0,
        max_accel_residual_mps2: float = 3.0,
        min_speed_for_direction_check_mps: float = 0.05,
    ) -> None:
        self.max_yaw_rate_residual_rad_s = max_yaw_rate_residual_rad_s
        self.max_direction_residual_deg = max_direction_residual_deg
        self.max_accel_residual_mps2 = max_accel_residual_mps2
        self.min_speed_for_direction_check_mps = min_speed_for_direction_check_mps


class ConsistencyMetrics:
    """Stores cross-sensor residuals and consistency statuses."""

    def __init__(self) -> None:
        # Yaw rate residuals (rad/s)
        self.res_wz_vis_wheel: Optional[float] = None
        self.res_wz_vis_imu: Optional[float] = None
        self.res_wz_wheel_imu: Optional[float] = None

        # Acceleration and direction residuals
        self.res_accel_wheel_imu: Optional[float] = None
        self.res_direction_deg: Optional[float] = None

        # Consistency flags (True = consistent, False = inconsistent / outlier)
        self.is_vis_wheel_consistent: bool = True
        self.is_vis_imu_consistent: bool = True
        self.is_wheel_imu_consistent: bool = True

        # Human-readable status notes
        self.warning_notes: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for diagnostics reporting."""
        return {
            'res_wz_vis_wheel': self.res_wz_vis_wheel if self.res_wz_vis_wheel is not None else 0.0,
            'res_wz_vis_imu': self.res_wz_vis_imu if self.res_wz_vis_imu is not None else 0.0,
            'res_wz_wheel_imu': self.res_wz_wheel_imu if self.res_wz_wheel_imu is not None else 0.0,
            'res_accel_wheel_imu': self.res_accel_wheel_imu if self.res_accel_wheel_imu is not None else 0.0,
            'res_direction_deg': self.res_direction_deg if self.res_direction_deg is not None else 0.0,
            'is_vis_wheel_consistent': self.is_vis_wheel_consistent,
            'is_vis_imu_consistent': self.is_vis_imu_consistent,
            'is_wheel_imu_consistent': self.is_wheel_imu_consistent,
            'warning_notes': "; ".join([f"{k}:{v}" for k, v in self.warning_notes.items()]),
        }


class CrossSensorConsistencyChecker:
    """Evaluates physical consistency across visual, wheel, and IMU modalities."""

    def __init__(self, config: Optional[ConsistencyConfig] = None) -> None:
        self.config = config or ConsistencyConfig()
        self.metrics = ConsistencyMetrics()
        self._last_wheel_stamp: Optional[float] = None
        self._last_wheel_vx: Optional[float] = None

    def check_wheel_vs_imu(
        self,
        wheel_meas: Optional[StampedMeasurement],
        imu_meas: Optional[StampedMeasurement],
    ) -> None:
        """Compare wheel odometry yaw rate and acceleration against IMU."""
        if wheel_meas is None or imu_meas is None or not wheel_meas.is_valid or not imu_meas.is_valid:
            return

        wz_wheel = float(wheel_meas.data.get('wz', 0.0))
        wz_imu = float(imu_meas.data.get('wz', 0.0))
        res_wz = abs(wz_wheel - wz_imu)
        self.metrics.res_wz_wheel_imu = res_wz

        if res_wz > self.config.max_yaw_rate_residual_rad_s:
            self.metrics.is_wheel_imu_consistent = False
            self.metrics.warning_notes['wheel_imu_wz'] = f"WHEEL_IMU_WZ_DIVERGENCE_{res_wz:.2f}"
        else:
            self.metrics.is_wheel_imu_consistent = True
            self.metrics.warning_notes.pop('wheel_imu_wz', None)

        # Acceleration check: compute dvx/dt from wheel
        vx_wheel = float(wheel_meas.data.get('vx', 0.0))
        if self._last_wheel_stamp is not None and self._last_wheel_vx is not None:
            dt = wheel_meas.stamp_sec - self._last_wheel_stamp
            if 1e-4 < dt < 0.2:
                a_wheel = (vx_wheel - self._last_wheel_vx) / dt
                ax_imu = float(imu_meas.data.get('ax', 0.0))
                res_accel = abs(a_wheel - ax_imu)
                self.metrics.res_accel_wheel_imu = res_accel
                if res_accel > self.config.max_accel_residual_mps2:
                    self.metrics.warning_notes['wheel_imu_accel'] = f"ACCEL_DIVERGENCE_{res_accel:.2f}"
                else:
                    self.metrics.warning_notes.pop('wheel_imu_accel', None)

        self._last_wheel_stamp = wheel_meas.stamp_sec
        self._last_wheel_vx = vx_wheel

    def check_visual_vs_others(
        self,
        vis_meas: Optional[StampedMeasurement],
        wheel_meas: Optional[StampedMeasurement],
        imu_meas: Optional[StampedMeasurement],
    ) -> None:
        """Compare visual yaw rate and translation direction against wheel and IMU."""
        if vis_meas is None or not vis_meas.is_valid:
            return

        wz_vis = float(vis_meas.data.get('wz', 0.0))

        # 1. Visual vs Wheel yaw rate & direction
        if wheel_meas is not None and wheel_meas.is_valid:
            wz_wheel = float(wheel_meas.data.get('wz', 0.0))
            res_wz_wheel = abs(wz_vis - wz_wheel)
            self.metrics.res_wz_vis_wheel = res_wz_wheel

            dir_error = float(vis_meas.data.get('direction_error_deg', 0.0))
            self.metrics.res_direction_deg = dir_error

            vx_wheel = float(wheel_meas.data.get('vx', 0.0))
            is_consistent = True

            if res_wz_wheel > self.config.max_yaw_rate_residual_rad_s:
                is_consistent = False
                self.metrics.warning_notes['vis_wheel_wz'] = f"VIS_WHEEL_WZ_DIVERGENCE_{res_wz_wheel:.2f}"
            else:
                self.metrics.warning_notes.pop('vis_wheel_wz', None)

            if abs(vx_wheel) > self.config.min_speed_for_direction_check_mps:
                if dir_error > self.config.max_direction_residual_deg:
                    is_consistent = False
                    self.metrics.warning_notes['vis_wheel_dir'] = f"VIS_DIRECTION_DIVERGENCE_{dir_error:.1f}deg"
                else:
                    self.metrics.warning_notes.pop('vis_wheel_dir', None)

            self.metrics.is_vis_wheel_consistent = is_consistent

        # 2. Visual vs IMU yaw rate
        if imu_meas is not None and imu_meas.is_valid:
            wz_imu = float(imu_meas.data.get('wz', 0.0))
            res_wz_imu = abs(wz_vis - wz_imu)
            self.metrics.res_wz_vis_imu = res_wz_imu

            if res_wz_imu > self.config.max_yaw_rate_residual_rad_s:
                self.metrics.is_vis_imu_consistent = False
                self.metrics.warning_notes['vis_imu_wz'] = f"VIS_IMU_WZ_DIVERGENCE_{res_wz_imu:.2f}"
            else:
                self.metrics.is_vis_imu_consistent = True
                self.metrics.warning_notes.pop('vis_imu_wz', None)
