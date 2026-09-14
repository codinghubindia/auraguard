"""Optional CSV Data Logger for Sensor Synchronization Diagnostics."""

import csv
import os
from typing import Any, Dict, Optional


class SensorSyncDataLogger:
    """Logs timestamped sensor synchronization metrics to CSV when enabled."""

    CSV_HEADERS = [
        "timestamp_sec",
        "camera_rate_hz",
        "imu_rate_hz",
        "odom_rate_hz",
        "camera_mean_period_sec",
        "imu_mean_period_sec",
        "odom_mean_period_sec",
        "cam_imu_median_diff_ms",
        "cam_odom_median_diff_ms",
        "imu_odom_median_diff_ms",
        "cam_imu_pct_within_tol",
        "cam_odom_pct_within_tol",
        "imu_odom_pct_within_tol",
        "frame_status",
        "camera_calib_status",
        "imu_status",
        "odom_status",
        "overall_status",
    ]

    def __init__(self, enabled: bool = False, output_path: str = "/tmp/sensor_sync_log.csv") -> None:
        self.enabled = enabled
        self.output_path = output_path
        self._file = None
        self._writer = None

        if self.enabled:
            self._open_log()

    def _open_log(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self._file = open(self.output_path, mode="w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(self.CSV_HEADERS)
            self._file.flush()
        except Exception:
            self.enabled = False

    def log_record(self, data: Dict[str, Any]) -> None:
        """Write single row of diagnostic values to CSV file."""
        if not self.enabled or self._writer is None or self._file is None:
            return

        row = [data.get(header, "") for header in self.CSV_HEADERS]
        try:
            self._writer.writerow(row)
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Close log file."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None
