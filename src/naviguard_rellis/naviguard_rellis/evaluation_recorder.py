"""
Evaluation Recorder and Metric Calculator for NAVIGUARD.

Computes standard robotics SLAM / visual odometry evaluation metrics:
- Absolute Trajectory Error (ATE) RMSE, Mean, Max, and Standard Deviation
- Relative Pose Error (RPE) Translation and Rotation Drift
- Final Position Error
- Trajectory Length and Tracking Continuity
- Exports complete metric reports to structured JSON
"""

import os
import json
import math
from typing import List, Dict, Optional, Tuple
import numpy as np


def compute_ate(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    align_origin: bool = True
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE) between ground truth and estimated positions.
    both arrays shape: (N, 3)
    """
    if len(gt_positions) == 0 or len(est_positions) == 0:
        return {
            'rmse': 0.0,
            'mean': 0.0,
            'max': 0.0,
            'std': 0.0,
            'count': 0
        }

    # Match lengths
    n = min(len(gt_positions), len(est_positions))
    gt = gt_positions[:n].copy()
    est = est_positions[:n].copy()

    if align_origin and n > 0:
        gt -= gt[0]
        est -= est[0]

    # Compute Euclidean distance errors at each step
    errors = np.linalg.norm(est - gt, axis=1)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    std_err = float(np.std(errors))

    return {
        'rmse': rmse,
        'mean': mean_err,
        'max': max_err,
        'std': std_err,
        'count': n
    }


def compute_rpe(
    gt_positions: np.ndarray,
    est_positions: np.ndarray,
    step: int = 1
) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE) for translation over a fixed step interval.
    """
    n = min(len(gt_positions), len(est_positions))
    if n <= step:
        return {'rmse': 0.0, 'mean': 0.0, 'max': 0.0, 'count': 0}

    gt_deltas = gt_positions[step:n] - gt_positions[:n - step]
    est_deltas = est_positions[step:n] - est_positions[:n - step]

    rpe_errors = np.linalg.norm(est_deltas - gt_deltas, axis=1)

    return {
        'rmse': float(np.sqrt(np.mean(rpe_errors ** 2))),
        'mean': float(np.mean(rpe_errors)),
        'max': float(np.max(rpe_errors)),
        'count': len(rpe_errors)
    }


def compute_trajectory_length(positions: np.ndarray) -> float:
    """Compute cumulative path length in meters."""
    if len(positions) < 2:
        return 0.0
    deltas = np.diff(positions, axis=0)
    dists = np.linalg.norm(deltas, axis=1)
    return float(np.sum(dists))


class EvaluationRecorder:
    """Records synchronized trajectory poses and calculates evaluation metrics."""

    def __init__(self, sequence_name: str = "00000"):
        self.sequence_name = sequence_name
        self.gt_timestamps: List[float] = []
        self.gt_positions: List[Tuple[float, float, float]] = []
        self.est_timestamps: List[float] = []
        self.est_positions: List[Tuple[float, float, float]] = []
        self.confidence_records: List[Dict] = []
        self.tracking_active_count = 0
        self.total_frames_evaluated = 0

    def add_ground_truth(self, t: float, x: float, y: float, z: float) -> None:
        self.gt_timestamps.append(t)
        self.gt_positions.append((x, y, z))

    def add_estimate(self, t: float, x: float, y: float, z: float) -> None:
        self.est_timestamps.append(t)
        self.est_positions.append((x, y, z))

    def record_confidence(self, t: float, overall_score: float, decision: str) -> None:
        self.confidence_records.append({
            'timestamp': t,
            'score': overall_score,
            'decision': decision
        })

    def get_metrics(self) -> Dict:
        """Calculate and return complete metrics dictionary."""
        gt_arr = np.array(self.gt_positions, dtype=np.float64) if self.gt_positions else np.zeros((0, 3))
        est_arr = np.array(self.est_positions, dtype=np.float64) if self.est_positions else np.zeros((0, 3))

        ate_metrics = compute_ate(gt_arr, est_arr, align_origin=True)
        rpe_metrics = compute_rpe(gt_arr, est_arr, step=1)

        gt_len = compute_trajectory_length(gt_arr)
        est_len = compute_trajectory_length(est_arr)

        final_pos_err = 0.0
        if len(gt_arr) > 0 and len(est_arr) > 0:
            n = min(len(gt_arr), len(est_arr))
            aligned_gt = gt_arr[n - 1] - gt_arr[0]
            aligned_est = est_arr[n - 1] - est_arr[0]
            final_pos_err = float(np.linalg.norm(aligned_est - aligned_gt))

        # Continuity
        continuity_pct = 100.0
        if self.total_frames_evaluated > 0:
            continuity_pct = float((self.tracking_active_count / self.total_frames_evaluated) * 100.0)
        elif len(est_arr) > 0:
            continuity_pct = 100.0

        avg_confidence = 0.0
        if self.confidence_records:
            avg_confidence = float(np.mean([c['score'] for c in self.confidence_records]))

        return {
            'sequence': self.sequence_name,
            'gt_samples': len(self.gt_positions),
            'est_samples': len(self.est_positions),
            'gt_trajectory_length_m': gt_len,
            'est_trajectory_length_m': est_len,
            'ate_rmse_m': ate_metrics['rmse'],
            'ate_mean_m': ate_metrics['mean'],
            'ate_max_m': ate_metrics['max'],
            'ate_std_m': ate_metrics['std'],
            'rpe_rmse_m': rpe_metrics['rmse'],
            'rpe_mean_m': rpe_metrics['mean'],
            'final_position_error_m': final_pos_err,
            'tracking_continuity_pct': continuity_pct,
            'average_confidence_score': avg_confidence
        }

    def save_json(self, output_path: str) -> None:
        """Save evaluation metrics to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        metrics = self.get_metrics()
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
