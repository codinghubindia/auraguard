import math
import numpy as np
from naviguard_rellis.evaluation_recorder import (
    compute_ate,
    compute_rpe,
    compute_trajectory_length,
    EvaluationRecorder
)


def test_compute_trajectory_length():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 4.0, 0.0]
    ])
    length = compute_trajectory_length(pts)
    assert math.isclose(length, 7.0, rel_tol=1e-5)


def test_compute_ate_identical():
    pts = np.array([[i * 1.0, 0.0, 0.0] for i in range(10)])
    ate = compute_ate(pts, pts)
    assert math.isclose(ate['rmse'], 0.0, abs_tol=1e-6)
    assert math.isclose(ate['max'], 0.0, abs_tol=1e-6)


def test_compute_ate_with_known_drift():
    gt = np.array([[float(i), 0.0, 0.0] for i in range(5)])
    # Drift 0.2m at each point except origin (which is zeroed out by align_origin)
    # If align_origin: gt[0]=(0,0,0), est[0]=(0,0,0)
    # Suppose est has constant error of 0.2 in Y relative to origin:
    est = np.array([[float(i), 0.2 * float(i), 0.0] for i in range(5)])
    ate = compute_ate(gt, est, align_origin=True)
    assert ate['rmse'] > 0.0
    assert ate['count'] == 5


def test_evaluation_recorder():
    recorder = EvaluationRecorder(sequence_name="test_seq")
    for i in range(10):
        t = float(i) * 0.1
        recorder.add_ground_truth(t, float(i), 0.0, 0.0)
        recorder.add_estimate(t, float(i) * 1.01, 0.02, 0.0)
        recorder.record_confidence(t, 0.95, "CONTINUE")

    metrics = recorder.get_metrics()
    assert metrics['sequence'] == "test_seq"
    assert metrics['gt_samples'] == 10
    assert metrics['est_samples'] == 10
    assert metrics['ate_rmse_m'] >= 0.0
    assert metrics['tracking_continuity_pct'] == 100.0
    assert math.isclose(metrics['average_confidence_score'], 0.95, rel_tol=1e-4)
