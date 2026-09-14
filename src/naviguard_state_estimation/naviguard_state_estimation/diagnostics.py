"""Diagnostic publishing and trajectory logging for NAVIGUARD UGV state estimation.

Produces structured DiagnosticArray messages reporting state vector, covariance norms,
sensor update/rejection counts, local scale observations, and cross-sensor consistency metrics.
Provides an optional trajectory CSV logger for verification and benchmarking.
"""

import csv
import os
from typing import Any, Dict, Optional
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np

from naviguard_state_estimation.consistency_checker import ConsistencyMetrics
from naviguard_state_estimation.state_model import RobotState


class StateEstimationDiagnostics:
    """Builds DiagnosticArray messages for state estimation monitoring."""

    def __init__(self, hardware_id: str = "naviguard_ugv") -> None:
        self.hardware_id = hardware_id

    def build_diagnostic_array(
        self,
        stamp_sec: float,
        state: RobotState,
        updates_count: Dict[str, int],
        rejections_count: Dict[str, int],
        last_rejection_reason: Dict[str, str],
        scale_data: Dict[str, Any],
        consistency: ConsistencyMetrics,
    ) -> DiagnosticArray:
        """Create a multi-status DiagnosticArray detailing all estimator subsystems."""
        msg = DiagnosticArray()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "odom"

        # 1. Core State Status
        s_state = DiagnosticStatus()
        s_state.name = "naviguard_state_estimation: state"
        s_state.hardware_id = self.hardware_id
        pos_cov_norm = float(np.linalg.norm(state.cov[0:2, 0:2]))
        s_state.level = DiagnosticStatus.OK if pos_cov_norm < 5.0 else DiagnosticStatus.WARN
        s_state.message = "Nominal state tracking" if s_state.level == DiagnosticStatus.OK else "High position covariance"

        s_state.values = [
            KeyValue(key="x_m", value=f"{state.x:.4f}"),
            KeyValue(key="y_m", value=f"{state.y:.4f}"),
            KeyValue(key="theta_rad", value=f"{state.theta:.4f}"),
            KeyValue(key="theta_deg", value=f"{np.degrees(state.theta):.2f}"),
            KeyValue(key="vx_mps", value=f"{state.vx:.4f}"),
            KeyValue(key="wz_radps", value=f"{state.wz:.4f}"),
            KeyValue(key="pos_cov_norm", value=f"{pos_cov_norm:.4f}"),
            KeyValue(key="theta_cov", value=f"{state.cov[2, 2]:.6f}"),
            KeyValue(key="vx_cov", value=f"{state.cov[3, 3]:.6f}"),
            KeyValue(key="wz_cov", value=f"{state.cov[4, 4]:.6f}"),
        ]
        msg.status.append(s_state)

        # 2. Sensor Updates and Gating Status
        s_sens = DiagnosticStatus()
        s_sens.name = "naviguard_state_estimation: sensors"
        s_sens.hardware_id = self.hardware_id
        total_rej = sum(rejections_count.values())
        s_sens.level = DiagnosticStatus.OK if total_rej < 10 else DiagnosticStatus.WARN
        s_sens.message = "Sensors active"

        s_sens.values = [
            KeyValue(key="updates_wheel", value=str(updates_count.get('wheel', 0))),
            KeyValue(key="updates_imu", value=str(updates_count.get('imu', 0))),
            KeyValue(key="updates_visual", value=str(updates_count.get('visual', 0))),
            KeyValue(key="rejections_wheel", value=str(rejections_count.get('wheel', 0))),
            KeyValue(key="rejections_imu", value=str(rejections_count.get('imu', 0))),
            KeyValue(key="rejections_visual", value=str(rejections_count.get('visual', 0))),
            KeyValue(key="last_rej_reason_wheel", value=last_rejection_reason.get('wheel', "NONE")),
            KeyValue(key="last_rej_reason_imu", value=last_rejection_reason.get('imu', "NONE")),
            KeyValue(key="last_rej_reason_visual", value=last_rejection_reason.get('visual', "NONE")),
        ]
        msg.status.append(s_sens)

        # 3. Monocular Scale Observation Status
        s_scale = DiagnosticStatus()
        s_scale.name = "naviguard_state_estimation: scale"
        s_scale.hardware_id = self.hardware_id
        scale_valid = bool(scale_data.get('scale_valid', False))
        scale_status = str(scale_data.get('scale_status', "UNOBSERVED"))
        s_scale.level = DiagnosticStatus.OK if scale_valid else DiagnosticStatus.WARN
        s_scale.message = scale_status

        s_scale.values = [
            KeyValue(key="scale_valid", value=str(scale_valid)),
            KeyValue(key="scale_status", value=scale_status),
            KeyValue(key="observed_scale_m", value=f"{float(scale_data.get('observed_scale_m', 0.0)):.4f}"),
            KeyValue(key="wheel_disp_m", value=f"{float(scale_data.get('wheel_disp_m', 0.0)):.4f}"),
            KeyValue(key="direction_error_deg", value=f"{float(scale_data.get('direction_error_deg', 0.0)):.2f}"),
            KeyValue(key="metric_vx_mps", value=f"{float(scale_data.get('metric_vx', 0.0)):.4f}"),
        ]
        msg.status.append(s_scale)

        # 4. Cross-Sensor Consistency Status
        s_cons = DiagnosticStatus()
        s_cons.name = "naviguard_state_estimation: consistency"
        s_cons.hardware_id = self.hardware_id
        cons_dict = consistency.to_dict()
        all_consistent = (
            cons_dict['is_vis_wheel_consistent']
            and cons_dict['is_vis_imu_consistent']
            and cons_dict['is_wheel_imu_consistent']
        )
        s_cons.level = DiagnosticStatus.OK if all_consistent else DiagnosticStatus.WARN
        s_cons.message = "Sensors physically consistent" if all_consistent else f"Divergence: {cons_dict['warning_notes']}"

        s_cons.values = [
            KeyValue(key="res_wz_vis_wheel_radps", value=f"{cons_dict['res_wz_vis_wheel']:.4f}"),
            KeyValue(key="res_wz_vis_imu_radps", value=f"{cons_dict['res_wz_vis_imu']:.4f}"),
            KeyValue(key="res_wz_wheel_imu_radps", value=f"{cons_dict['res_wz_wheel_imu']:.4f}"),
            KeyValue(key="res_accel_wheel_imu_mps2", value=f"{cons_dict['res_accel_wheel_imu']:.4f}"),
            KeyValue(key="res_direction_deg", value=f"{cons_dict['res_direction_deg']:.2f}"),
            KeyValue(key="is_vis_wheel_consistent", value=str(cons_dict['is_vis_wheel_consistent'])),
            KeyValue(key="is_vis_imu_consistent", value=str(cons_dict['is_vis_imu_consistent'])),
            KeyValue(key="is_wheel_imu_consistent", value=str(cons_dict['is_wheel_imu_consistent'])),
            KeyValue(key="warning_notes", value=str(cons_dict['warning_notes'])),
        ]
        msg.status.append(s_cons)

        return msg


class TrajectoryLogger:
    """Logs estimated state and sensor telemetry to a CSV file."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._file = None
        self._writer = None

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self._file = open(self.file_path, mode='w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'timestamp_sec',
            'x_m', 'y_m', 'theta_rad', 'vx_mps', 'wz_radps',
            'cov_xx', 'cov_yy', 'cov_thth', 'cov_vxvx', 'cov_wzwz',
            'scale_valid', 'observed_scale_m', 'metric_vx_mps',
            'res_wz_vis_wheel', 'res_wz_wheel_imu',
        ])
        self._file.flush()

    def log_row(
        self,
        stamp_sec: float,
        state: RobotState,
        scale_data: Dict[str, Any],
        consistency: ConsistencyMetrics,
    ) -> None:
        """Write single row to CSV log."""
        if self._writer is None or self._file is None:
            return

        self._writer.writerow([
            f"{stamp_sec:.4f}",
            f"{state.x:.4f}", f"{state.y:.4f}", f"{state.theta:.4f}",
            f"{state.vx:.4f}", f"{state.wz:.4f}",
            f"{state.cov[0, 0]:.6f}", f"{state.cov[1, 1]:.6f}",
            f"{state.cov[2, 2]:.6f}", f"{state.cov[3, 3]:.6f}", f"{state.cov[4, 4]:.6f}",
            str(scale_data.get('scale_valid', False)),
            f"{float(scale_data.get('observed_scale_m', 0.0)):.4f}",
            f"{float(scale_data.get('metric_vx', 0.0)):.4f}",
            f"{consistency.res_wz_vis_wheel or 0.0:.4f}",
            f"{consistency.res_wz_wheel_imu or 0.0:.4f}",
        ])
        self._file.flush()

    def close(self) -> None:
        """Flush and close open file handle."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
