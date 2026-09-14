"""Sensor Timestamp Monitor for NAVIGUARD UGV.

Tracks incoming message timestamps, evaluates monotonicity, measures inter-frame
periods, detects temporal gaps/jumps, and computes message arrival delays.
"""

from collections import deque
from typing import Deque, List, Optional
import numpy as np


class TimestampRecord:
    """Individual record of a received sensor message."""

    def __init__(
        self,
        stamp_sec: float,
        arrival_sec: float,
        frame_id: str,
        msg_type: str,
        seq: Optional[int] = None,
    ) -> None:
        self.stamp_sec = stamp_sec
        self.arrival_sec = arrival_sec
        self.frame_id = frame_id
        self.msg_type = msg_type
        self.seq = seq
        self.delay_sec = max(0.0, arrival_sec - stamp_sec)


class SensorTimestampStats:
    """Statistical summary of sensor timestamp characteristics."""

    def __init__(
        self,
        sensor_name: str,
        total_count: int = 0,
        current_rate_hz: float = 0.0,
        mean_period_sec: float = 0.0,
        min_period_sec: float = 0.0,
        max_period_sec: float = 0.0,
        std_period_sec: float = 0.0,
        is_monotonic: bool = True,
        num_gaps: int = 0,
        num_jumps: int = 0,
        mean_delay_sec: float = 0.0,
        last_stamp_sec: float = 0.0,
        last_frame_id: str = "",
        status: str = "NO_DATA",
    ) -> None:
        self.sensor_name = sensor_name
        self.total_count = total_count
        self.current_rate_hz = current_rate_hz
        self.mean_period_sec = mean_period_sec
        self.min_period_sec = min_period_sec
        self.max_period_sec = max_period_sec
        self.std_period_sec = std_period_sec
        self.is_monotonic = is_monotonic
        self.num_gaps = num_gaps
        self.num_jumps = num_jumps
        self.mean_delay_sec = mean_delay_sec
        self.last_stamp_sec = last_stamp_sec
        self.last_frame_id = last_frame_id
        self.status = status


class SensorTimestampMonitor:
    """Sliding-window monitor for a single sensor's timestamp behavior."""

    def __init__(
        self,
        sensor_name: str,
        expected_rate_hz: float,
        window_size: int = 100,
        gap_threshold_factor: float = 2.5,
        rate_tolerance_pct: float = 35.0,
    ) -> None:
        self.sensor_name = sensor_name
        self.expected_rate_hz = max(0.1, expected_rate_hz)
        self.nominal_period_sec = 1.0 / self.expected_rate_hz
        self.window_size = window_size
        self.gap_threshold_sec = gap_threshold_factor * self.nominal_period_sec
        self.rate_tolerance_pct = rate_tolerance_pct

        self.records: Deque[TimestampRecord] = deque(maxlen=window_size)
        self.total_count = 0
        self.is_monotonic = True
        self.num_gaps = 0
        self.num_jumps = 0

        self.last_record: Optional[TimestampRecord] = None

    def add_sample(
        self,
        stamp_sec: float,
        arrival_sec: float,
        frame_id: str,
        msg_type: str,
        seq: Optional[int] = None,
    ) -> None:
        """Add incoming message sample and evaluate temporal continuity."""
        record = TimestampRecord(stamp_sec, arrival_sec, frame_id, msg_type, seq)

        if self.last_record is not None:
            dt = stamp_sec - self.last_record.stamp_sec

            # Monotonicity check (allowing for tiny floating point noise <= 1e-7s)
            if dt < -1e-6:
                self.is_monotonic = False
                self.num_jumps += 1
            elif dt > self.gap_threshold_sec:
                self.num_gaps += 1

        self.records.append(record)
        self.last_record = record
        self.total_count += 1

    def compute_stats(self, current_time_sec: Optional[float] = None) -> SensorTimestampStats:
        """Compute rate, periods, variance, and status over recent sliding window."""
        if len(self.records) < 2:
            status = "NO_DATA" if self.total_count == 0 else "WARMUP"
            last_stamp = self.last_record.stamp_sec if self.last_record else 0.0
            last_frame = self.last_record.frame_id if self.last_record else ""
            return SensorTimestampStats(
                sensor_name=self.sensor_name,
                total_count=self.total_count,
                last_stamp_sec=last_stamp,
                last_frame_id=last_frame,
                status=status,
            )

        # Extract timestamps in window
        stamps = [r.stamp_sec for r in self.records]
        delays = [r.delay_sec for r in self.records]

        diffs = np.diff(stamps)
        positive_diffs = diffs[diffs > 1e-6]

        if len(positive_diffs) > 0:
            mean_period = float(np.mean(positive_diffs))
            min_period = float(np.min(positive_diffs))
            max_period = float(np.max(positive_diffs))
            std_period = float(np.std(positive_diffs))
            current_rate = 1.0 / mean_period if mean_period > 0.0 else 0.0
        else:
            mean_period = 0.0
            min_period = 0.0
            max_period = 0.0
            std_period = 0.0
            current_rate = 0.0

        mean_delay = float(np.mean(delays)) if delays else 0.0
        last_stamp = self.last_record.stamp_sec if self.last_record else 0.0
        last_frame = self.last_record.frame_id if self.last_record else ""

        # Status determination
        status = "OK"
        if not self.is_monotonic:
            status = "FAIL"
        elif current_time_sec is not None and self.last_record is not None:
            # Check staleness
            time_since_last = current_time_sec - self.last_record.arrival_sec
            if time_since_last > 4.0 * self.nominal_period_sec:
                status = "STALE"

        if status == "OK":
            # Rate deviation check
            rate_err_pct = abs(current_rate - self.expected_rate_hz) / self.expected_rate_hz * 100.0
            if rate_err_pct > self.rate_tolerance_pct or self.num_gaps > 0:
                status = "WARN"

        return SensorTimestampStats(
            sensor_name=self.sensor_name,
            total_count=self.total_count,
            current_rate_hz=current_rate,
            mean_period_sec=mean_period,
            min_period_sec=min_period,
            max_period_sec=max_period,
            std_period_sec=std_period,
            is_monotonic=self.is_monotonic,
            num_gaps=self.num_gaps,
            num_jumps=self.num_jumps,
            mean_delay_sec=mean_delay,
            last_stamp_sec=last_stamp,
            last_frame_id=last_frame,
            status=status,
        )

    def get_recent_timestamps(self) -> List[float]:
        """Return list of timestamps currently stored in sliding window."""
        return [r.stamp_sec for r in self.records]
