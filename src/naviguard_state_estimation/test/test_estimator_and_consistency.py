import pytest
import numpy as np

from naviguard_state_estimation.state_model import KinematicStateModel, RobotState
from naviguard_state_estimation.estimator import NaviguardStateEstimator, EstimatorConfig
from naviguard_state_estimation.consistency_checker import CrossSensorConsistencyChecker, ConsistencyConfig
from naviguard_state_estimation.measurement_buffer import StampedMeasurement


def test_kinematic_model_prediction():
    model = KinematicStateModel(q_pos=0.01, q_theta=0.005, q_vx=0.05, q_wz=0.02)
    # Robot starting at (0, 0, 0) moving at 1.0 m/s forward
    state = RobotState(x=0.0, y=0.0, theta=0.0, vx=1.0, wz=0.0, stamp_sec=0.0)

    # Predict forward 1 second
    pred_state = model.predict(state, dt=1.0)
    assert pytest.approx(pred_state.x, rel=1e-3) == 1.0
    assert pytest.approx(pred_state.y, abs=1e-5) == 0.0
    assert pytest.approx(pred_state.theta, abs=1e-5) == 0.0

    # Verify covariance grew
    assert pred_state.cov[0, 0] > state.cov[0, 0]
    assert pred_state.cov[1, 1] > state.cov[1, 1]


def test_ekf_wheel_and_imu_updates():
    estimator = NaviguardStateEstimator()
    estimator.initialize_pose(0.0, 0.0, 0.0, stamp_sec=1.0)

    # Wheel odometry update: vx = 0.5 m/s, wz = 0.0
    acc, reason = estimator.update_wheel_odometry(vx=0.5, wz=0.0, vx_cov=0.02, wz_cov=0.04, stamp_sec=1.1)
    assert acc is True
    assert 0.3 < estimator.state.vx <= 0.5

    # IMU update: wz = 0.2 rad/s
    acc, reason = estimator.update_imu(wz=0.2, wz_cov=0.005, stamp_sec=1.15)
    assert acc is True
    assert 0.1 < estimator.state.wz <= 0.2

    # Verify Joseph form covariance symmetry and positive-definiteness
    cov = estimator.state.cov
    assert np.allclose(cov, cov.T, atol=1e-8)
    eigenvalues = np.linalg.eigvals(cov)
    assert np.all(eigenvalues > 0.0)


def test_ekf_visual_updates():
    estimator = NaviguardStateEstimator()
    estimator.initialize_pose(0.0, 0.0, 0.0, stamp_sec=1.0)

    # 1-DOF visual update (only yaw rate)
    acc, reason = estimator.update_visual(wz_vis=0.15, wz_cov=0.05, stamp_sec=1.1)
    assert acc is True
    assert estimator.updates_count['visual'] == 1

    # 2-DOF visual update (scaled linear velocity + yaw rate)
    acc, reason = estimator.update_visual(
        wz_vis=0.15,
        wz_cov=0.05,
        stamp_sec=1.2,
        metric_vx=0.6,
        vx_cov=0.05,
    )
    assert acc is True
    assert estimator.updates_count['visual'] == 2


def test_ekf_mahalanobis_outlier_rejection():
    # Set tighter gate for testing
    estimator = NaviguardStateEstimator(EstimatorConfig(mahalanobis_gate_1d=4.0, mahalanobis_gate_2d=6.0))
    estimator.initialize_pose(0.0, 0.0, 0.0, stamp_sec=1.0)

    # First update establishes 0.0 m/s
    estimator.update_wheel_odometry(vx=0.0, wz=0.0, vx_cov=0.01, wz_cov=0.01, stamp_sec=1.05)

    # Absurd outlier jump: vx = 50.0 m/s
    acc, reason = estimator.update_wheel_odometry(vx=50.0, wz=0.0, vx_cov=0.01, wz_cov=0.01, stamp_sec=1.10)
    assert acc is False
    assert "MAHALANOBIS_GATE_EXCEEDED" in reason
    assert estimator.rejections_count['wheel'] == 1

    # Absurd IMU outlier jump: wz = 20.0 rad/s
    acc, reason = estimator.update_imu(wz=20.0, wz_cov=0.01, stamp_sec=1.12)
    assert acc is False
    assert "MAHALANOBIS_GATE_EXCEEDED" in reason
    assert estimator.rejections_count['imu'] == 1


def test_cross_sensor_consistency_checker():
    checker = CrossSensorConsistencyChecker(ConsistencyConfig(max_yaw_rate_residual_rad_s=0.3))

    wheel_meas = StampedMeasurement(1.0, "wheel", "odom", {'vx': 0.5, 'wz': 0.1})
    imu_meas = StampedMeasurement(1.0, "imu", "imu_link", {'wz': 0.12, 'ax': 0.0})

    checker.check_wheel_vs_imu(wheel_meas, imu_meas)
    assert checker.metrics.is_wheel_imu_consistent is True
    assert pytest.approx(checker.metrics.res_wz_wheel_imu, abs=1e-3) == 0.02

    # Inconsistent IMU measurement (divergent yaw rate)
    imu_bad = StampedMeasurement(1.05, "imu", "imu_link", {'wz': 1.0, 'ax': 0.0})
    checker.check_wheel_vs_imu(wheel_meas, imu_bad)
    assert checker.metrics.is_wheel_imu_consistent is False
    assert "WHEEL_IMU_WZ_DIVERGENCE" in checker.metrics.warning_notes['wheel_imu_wz']

    # Visual vs Wheel direction check
    vis_meas = StampedMeasurement(
        1.10, "visual", "camera_link",
        {'wz': 0.1, 'direction_error_deg': 60.0, 'scale_valid': False}
    )
    checker.check_visual_vs_others(vis_meas, wheel_meas, imu_meas)
    assert checker.metrics.is_vis_wheel_consistent is False
    assert "VIS_DIRECTION_DIVERGENCE" in checker.metrics.warning_notes['vis_wheel_dir']
