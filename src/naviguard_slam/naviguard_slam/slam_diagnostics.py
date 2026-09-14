"""Diagnostics builder and evaluation logging for NAVIGUARD SLAM.

Publishes structured DiagnosticArray messages on /slam/diagnostics detailing
tracking state, graph metrics, loop closures, and sensor health.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import numpy as np


class SlamDiagnosticsBuilder:
    """Builds DiagnosticArray messages for SLAM health and localization tracking."""

    def __init__(self, hardware_id: str = "naviguard_ugv_slam") -> None:
        self.hardware_id = hardware_id

    def build_diagnostic_array(
        self,
        stamp_sec: float,
        mode: str,
        tracking_state: str,
        pose_map: Tuple[float, float, float],
        features_tracked: int,
        reprojection_error_px: float,
        num_keyframes: int,
        num_landmarks: int,
        loop_closures_count: int,
        opt_error: float,
        map_to_odom_offset: Tuple[float, float, float],
        sensor_rates: Dict[str, float],
        failure_reason: str = "",
    ) -> DiagnosticArray:
        """Construct full DiagnosticArray message."""
        msg = DiagnosticArray()
        msg.header.stamp.sec = int(stamp_sec)
        msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
        msg.header.frame_id = "map"

        # 1. Tracking Status
        s_track = DiagnosticStatus()
        s_track.name = "naviguard_slam: tracking"
        s_track.hardware_id = self.hardware_id
        if tracking_state == "OK":
            s_track.level = DiagnosticStatus.OK
            s_track.message = "Nominal SLAM tracking"
        elif tracking_state in ("DEGRADED", "INITIALIZING"):
            s_track.level = DiagnosticStatus.WARN
            s_track.message = f"Tracking {tracking_state}: {failure_reason}"
        else:
            s_track.level = DiagnosticStatus.ERROR
            s_track.message = f"Tracking Lost: {failure_reason}"

        s_track.values = [
            KeyValue(key="slam_mode", value=mode),
            KeyValue(key="tracking_state", value=tracking_state),
            KeyValue(key="estimated_x_m", value=f"{pose_map[0]:.4f}"),
            KeyValue(key="estimated_y_m", value=f"{pose_map[1]:.4f}"),
            KeyValue(key="estimated_yaw_deg", value=f"{np.degrees(pose_map[2]):.2f}"),
            KeyValue(key="features_tracked", value=str(features_tracked)),
            KeyValue(key="reprojection_error_px", value=f"{reprojection_error_px:.2f}"),
            KeyValue(key="failure_reason", value=failure_reason if failure_reason else "NONE"),
        ]
        msg.status.append(s_track)

        # 2. Graph & Map Metrics
        s_graph = DiagnosticStatus()
        s_graph.name = "naviguard_slam: graph"
        s_graph.hardware_id = self.hardware_id
        s_graph.level = DiagnosticStatus.OK
        s_graph.message = "Graph consistent"

        s_graph.values = [
            KeyValue(key="num_keyframes", value=str(num_keyframes)),
            KeyValue(key="num_landmarks", value=str(num_landmarks)),
            KeyValue(key="loop_closures_count", value=str(loop_closures_count)),
            KeyValue(key="last_opt_error", value=f"{opt_error:.4f}"),
            KeyValue(key="map_to_odom_dx_m", value=f"{map_to_odom_offset[0]:.4f}"),
            KeyValue(key="map_to_odom_dy_m", value=f"{map_to_odom_offset[1]:.4f}"),
            KeyValue(key="map_to_odom_dyaw_deg", value=f"{np.degrees(map_to_odom_offset[2]):.2f}"),
        ]
        msg.status.append(s_graph)

        # 3. Sensor Health
        s_sens = DiagnosticStatus()
        s_sens.name = "naviguard_slam: sensors"
        s_sens.hardware_id = self.hardware_id
        cam_rate = sensor_rates.get('camera', 0.0)
        odom_rate = sensor_rates.get('odom', 0.0)
        imu_rate = sensor_rates.get('imu', 0.0)
        sens_ok = (cam_rate > 5.0 and odom_rate > 10.0 and imu_rate > 20.0)
        s_sens.level = DiagnosticStatus.OK if sens_ok else DiagnosticStatus.WARN
        s_sens.message = "Sensors operational" if sens_ok else "Low sensor rates detected"

        s_sens.values = [
            KeyValue(key="camera_rate_hz", value=f"{cam_rate:.1f}"),
            KeyValue(key="odom_rate_hz", value=f"{odom_rate:.1f}"),
            KeyValue(key="imu_rate_hz", value=f"{imu_rate:.1f}"),
        ]
        msg.status.append(s_sens)

        return msg


class SlamTrajectoryLogger:
    """Logs estimated vs ground truth trajectories to CSV."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._file = None
        self._writer = None

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self._file = open(self.file_path, mode='w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'timestamp_sec',
            'estimated_x', 'estimated_y', 'estimated_yaw',
            'odom_x', 'odom_y', 'odom_yaw',
            'tracking_state', 'features_tracked',
            'num_keyframes', 'loop_closures',
        ])
        self._file.flush()

    def log_row(
        self,
        stamp_sec: float,
        pose_map: Tuple[float, float, float],
        pose_odom: Tuple[float, float, float],
        tracking_state: str,
        features_tracked: int,
        num_keyframes: int,
        loop_closures: int,
    ) -> None:
        if self._writer is None or self._file is None:
            return

        self._writer.writerow([
            f"{stamp_sec:.4f}",
            f"{pose_map[0]:.4f}", f"{pose_map[1]:.4f}", f"{pose_map[2]:.4f}",
            f"{pose_odom[0]:.4f}", f"{pose_odom[1]:.4f}", f"{pose_odom[2]:.4f}",
            tracking_state, features_tracked, num_keyframes, loop_closures,
        ])
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
