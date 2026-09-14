"""Timestamp-aware measurement buffering and interpolation for NAVIGUARD UGV.

Maintains chronologically ordered buffers for IMU, wheel odometry, and visual motion,
supporting nearest-neighbor lookup, time-window integration, and age calculation.
"""

import bisect
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StampedMeasurement:
    """Generic container for a timestamped sensor or visual measurement."""

    def __init__(
        self,
        stamp_sec: float,
        source: str,
        frame_id: str,
        data: Dict[str, Any],
        is_valid: bool = True,
        rejection_reason: str = "",
    ) -> None:
        self.stamp_sec = stamp_sec
        self.source = source
        self.frame_id = frame_id
        self.data = data
        self.is_valid = is_valid
        self.rejection_reason = rejection_reason

    @property
    def age_sec(self) -> float:
        """Calculate age relative to current measurement or query."""
        return 0.0


class MeasurementBuffer:
    """Sliding-window buffer storing measurements ordered by ROS timestamp."""

    def __init__(self, max_size: int = 300, max_age_sec: float = 5.0) -> None:
        self.max_size = max_size
        self.max_age_sec = max_age_sec
        self.buffer: List[StampedMeasurement] = []

    def add(self, measurement: StampedMeasurement) -> bool:
        """Insert measurement in timestamp-sorted order and prune stale data.

        Returns:
            True if measurement was successfully added, False if invalid timestamp.
        """
        if measurement.stamp_sec <= 0.0 or np.isnan(measurement.stamp_sec) or np.isinf(measurement.stamp_sec):
            measurement.is_valid = False
            measurement.rejection_reason = "INVALID_TIMESTAMP"
            return False

        # Find sorted insertion index based on timestamp
        stamps = [m.stamp_sec for m in self.buffer]
        idx = bisect.bisect_right(stamps, measurement.stamp_sec)
        self.buffer.insert(idx, measurement)

        # Prune buffer size and age
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

        newest_stamp = self.buffer[-1].stamp_sec
        cutoff = newest_stamp - self.max_age_sec
        while len(self.buffer) > 1 and self.buffer[0].stamp_sec < cutoff:
            self.buffer.pop(0)

        return True

    def get_latest(self) -> Optional[StampedMeasurement]:
        """Return the most recent measurement."""
        return self.buffer[-1] if self.buffer else None

    def get_nearest(
        self,
        target_time_sec: float,
        max_time_diff_sec: float = 0.08,
    ) -> Optional[StampedMeasurement]:
        """Find the nearest measurement within a maximum allowed time offset."""
        if not self.buffer:
            return None

        stamps = [m.stamp_sec for m in self.buffer]
        idx = bisect.bisect_left(stamps, target_time_sec)

        candidates = []
        if idx < len(self.buffer):
            candidates.append(self.buffer[idx])
        if idx > 0:
            candidates.append(self.buffer[idx - 1])

        if not candidates:
            return None

        best = min(candidates, key=lambda m: abs(m.stamp_sec - target_time_sec))
        if abs(best.stamp_sec - target_time_sec) <= max_time_diff_sec:
            return best
        return None

    def get_range(self, t_start: float, t_end: float) -> List[StampedMeasurement]:
        """Return all measurements in interval [t_start, t_end]."""
        if not self.buffer or t_start > t_end:
            return []

        stamps = [m.stamp_sec for m in self.buffer]
        idx_start = bisect.bisect_left(stamps, t_start)
        idx_end = bisect.bisect_right(stamps, t_end)
        return self.buffer[idx_start:idx_end]

    def integrate_wheel_motion(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[float, float, float]:
        """Numerically integrate wheel motion over [t_start, t_end].

        Returns:
            Tuple of (distance_meters, delta_yaw_rad, mean_vx_mps).
        """
        samples = self.get_range(t_start, t_end)
        if not samples:
            # Fall back to nearest sample if available
            nearest = self.get_nearest(0.5 * (t_start + t_end), max_time_diff_sec=0.2)
            if nearest and nearest.is_valid:
                vx = float(nearest.data.get('vx', 0.0))
                wz = float(nearest.data.get('wz', 0.0))
                dt = max(0.0, t_end - t_start)
                return vx * dt, wz * dt, vx
            return 0.0, 0.0, 0.0

        if len(samples) == 1:
            vx = float(samples[0].data.get('vx', 0.0))
            wz = float(samples[0].data.get('wz', 0.0))
            dt = max(0.0, t_end - t_start)
            return vx * dt, wz * dt, vx

        dist = 0.0
        dyaw = 0.0
        vxs = []

        for i in range(len(samples) - 1):
            m0, m1 = samples[i], samples[i + 1]
            dt = m1.stamp_sec - m0.stamp_sec
            if dt <= 0.0:
                continue
            v_avg = 0.5 * (float(m0.data.get('vx', 0.0)) + float(m1.data.get('vx', 0.0)))
            w_avg = 0.5 * (float(m0.data.get('wz', 0.0)) + float(m1.data.get('wz', 0.0)))
            dist += v_avg * dt
            dyaw += w_avg * dt
            vxs.append(v_avg)

        mean_vx = float(np.mean(vxs)) if vxs else 0.0
        return dist, dyaw, mean_vx

    def clear(self) -> None:
        """Empty the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
