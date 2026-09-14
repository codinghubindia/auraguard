"""Cross-Sensor Synchronization Analyzer for NAVIGUARD UGV.

Computes nearest-neighbor timestamp pairing between sensor streams,
evaluates temporal alignment offsets, and quantifies synchronization tolerance compliance.
"""

import bisect
from typing import Dict, List, Optional
import numpy as np


class SyncPairStats:
    """Statistical summary of temporal alignment between two sensor streams."""

    def __init__(
        self,
        pair_name: str,
        num_paired_samples: int = 0,
        mean_abs_diff_ms: float = 0.0,
        median_abs_diff_ms: float = 0.0,
        max_abs_diff_ms: float = 0.0,
        min_abs_diff_ms: float = 0.0,
        std_diff_ms: float = 0.0,
        pct_within_tolerance: float = 0.0,
        tolerance_ms: float = 20.0,
        status: str = "INSUFFICIENT_DATA",
    ) -> None:
        self.pair_name = pair_name
        self.num_paired_samples = num_paired_samples
        self.mean_abs_diff_ms = mean_abs_diff_ms
        self.median_abs_diff_ms = median_abs_diff_ms
        self.max_abs_diff_ms = max_abs_diff_ms
        self.min_abs_diff_ms = min_abs_diff_ms
        self.std_diff_ms = std_diff_ms
        self.pct_within_tolerance = pct_within_tolerance
        self.tolerance_ms = tolerance_ms
        self.status = status


class CrossSensorSyncAnalyzer:
    """Analyzes timestamp offset distributions across multiple sensor streams."""

    def __init__(self, tolerance_ms: float = 20.0, max_association_window_sec: float = 0.5) -> None:
        self.tolerance_ms = tolerance_ms
        self.max_association_window_sec = max_association_window_sec

    def evaluate_pair(
        self,
        stamps_ref: List[float],
        stamps_target: List[float],
        pair_name: str,
    ) -> SyncPairStats:
        """Find nearest-neighbor timestamps in stamps_target for each stamp in stamps_ref.

        Args:
            stamps_ref: Reference timestamps in seconds (typically lower-rate sensor, e.g. camera).
            stamps_target: Target timestamps in seconds (typically higher-rate sensor, e.g. IMU/odom).
            pair_name: Identifier for sensor pair.

        Returns:
            SyncPairStats containing offset statistics and tolerance compliance.
        """
        if len(stamps_ref) < 3 or len(stamps_target) < 3:
            return SyncPairStats(
                pair_name=pair_name,
                num_paired_samples=0,
                tolerance_ms=self.tolerance_ms,
                status="INSUFFICIENT_DATA",
            )

        # Sort target timestamps for binary search
        sorted_targets = sorted(stamps_target)
        n_targets = len(sorted_targets)

        deltas_ms: List[float] = []

        for t_ref in stamps_ref:
            # Binary search for closest target timestamp
            idx = bisect.bisect_left(sorted_targets, t_ref)

            candidates = []
            if idx < n_targets:
                candidates.append(sorted_targets[idx])
            if idx > 0:
                candidates.append(sorted_targets[idx - 1])

            if not candidates:
                continue

            # Find closest candidate
            best_t = min(candidates, key=lambda t: abs(t - t_ref))
            diff_sec = abs(best_t - t_ref)

            # Only consider association if within reasonable window
            if diff_sec <= self.max_association_window_sec:
                deltas_ms.append(diff_sec * 1000.0)

        num_paired = len(deltas_ms)
        if num_paired < 3:
            return SyncPairStats(
                pair_name=pair_name,
                num_paired_samples=num_paired,
                tolerance_ms=self.tolerance_ms,
                status="INSUFFICIENT_DATA",
            )

        arr = np.array(deltas_ms, dtype=np.float64)
        mean_abs = float(np.mean(arr))
        median_abs = float(np.median(arr))
        max_abs = float(np.max(arr))
        min_abs = float(np.min(arr))
        std_diff = float(np.std(arr))

        within_tol = np.sum(arr <= self.tolerance_ms)
        pct_within = float(within_tol) / float(num_paired) * 100.0

        # Classification based on percentage within tolerance and median offset
        if pct_within >= 85.0 and median_abs <= self.tolerance_ms:
            status = "OK"
        elif pct_within >= 50.0:
            status = "WARN"
        else:
            status = "FAIL"

        return SyncPairStats(
            pair_name=pair_name,
            num_paired_samples=num_paired,
            mean_abs_diff_ms=mean_abs,
            median_abs_diff_ms=median_abs,
            max_abs_diff_ms=max_abs,
            min_abs_diff_ms=min_abs,
            std_diff_ms=std_diff,
            pct_within_tolerance=pct_within,
            tolerance_ms=self.tolerance_ms,
            status=status,
        )
