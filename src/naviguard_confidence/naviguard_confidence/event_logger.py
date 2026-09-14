"""Event and transition logging for NAVIGUARD confidence decisions.

Exports transition events and telemetry to CSV and JSON formats for audit,
benchmarking, and offline evaluation.
"""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

from naviguard_confidence.confidence_dimensions import DecisionResult, DecisionState


class ConfidenceEventLogger:
    """Logs state transitions and diagnostic snapshots."""

    def __init__(self, csv_path: str = "", json_path: str = "") -> None:
        self.csv_path = csv_path
        self.json_path = json_path
        self.transition_events: List[Dict[str, Any]] = []
        self.state_time_accum: Dict[str, float] = {
            "CONTINUE": 0.0,
            "VERIFY": 0.0,
            "RECOVER": 0.0,
        }
        self.last_step_time: Optional[float] = None
        self._init_csv()

    def _init_csv(self) -> None:
        if not self.csv_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_sec",
                "wall_time",
                "event_type",
                "prev_state",
                "new_state",
                "primary_reason",
                "overall_conf",
                "visual_conf",
                "loc_conf",
                "imu_conf",
                "wheel_conf",
                "temporal_conf",
                "cross_conf",
                "map_conf",
                "dwell_sec",
            ])

    def log_step(self, result: DecisionResult) -> None:
        """Accumulate dwell time and log transitions."""
        now = result.timestamp_sec
        if self.last_step_time is not None:
            dt = max(0.0, now - self.last_step_time)
            self.state_time_accum[result.state.to_string()] += dt
        self.last_step_time = now

        if result.transition_occurred and result.previous_state is not None:
            event = {
                "timestamp_sec": float(result.timestamp_sec),
                "wall_time": time.time(),
                "event_type": "STATE_TRANSITION",
                "prev_state": result.previous_state.to_string(),
                "new_state": result.state.to_string(),
                "primary_reason": result.primary_reason,
                "overall_conf": float(result.scores.overall),
                "visual_conf": float(result.scores.visual),
                "loc_conf": float(result.scores.localization),
                "imu_conf": float(result.scores.imu),
                "wheel_conf": float(result.scores.wheel),
                "temporal_conf": float(result.scores.temporal),
                "cross_conf": float(result.scores.cross_sensor),
                "map_conf": float(result.scores.map),
                "dwell_sec": float(result.dwell_time_sec),
            }
            self.transition_events.append(event)
            self._write_csv_row(event)

    def _write_csv_row(self, ev: Dict[str, Any]) -> None:
        if not self.csv_path:
            return
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{ev['timestamp_sec']:.3f}",
                f"{ev['wall_time']:.3f}",
                ev['event_type'],
                ev['prev_state'],
                ev['new_state'],
                ev['primary_reason'],
                f"{ev['overall_conf']:.3f}",
                f"{ev['visual_conf']:.3f}",
                f"{ev['loc_conf']:.3f}",
                f"{ev['imu_conf']:.3f}",
                f"{ev['wheel_conf']:.3f}",
                f"{ev['temporal_conf']:.3f}",
                f"{ev['cross_conf']:.3f}",
                f"{ev['map_conf']:.3f}",
                f"{ev['dwell_sec']:.3f}",
            ])

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for evaluation report."""
        total_time = sum(self.state_time_accum.values())
        return {
            "total_monitoring_time_sec": total_time,
            "total_transitions": len(self.transition_events),
            "state_dwell_times_sec": dict(self.state_time_accum),
            "state_percentages": {
                k: (v / total_time * 100.0) if total_time > 0 else 0.0
                for k, v in self.state_time_accum.items()
            },
            "transition_history": list(self.transition_events),
        }

    def save_json_summary(self) -> None:
        if not self.json_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.json_path)), exist_ok=True)
        summary = self.get_summary_statistics()
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
