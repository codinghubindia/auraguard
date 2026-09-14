import pytest
from naviguard_confidence.sensor_monitors import (
    CompositeConfidenceEngine,
    CrossSensorMonitor,
    ImuMonitor,
    LocalizationMonitor,
    MapMonitor,
    TemporalMonitor,
    VisualMonitor,
    WheelMonitor,
)


def test_visual_monitor_nominal_and_degraded():
    mon = VisualMonitor()

    # Nominal
    telemetry_ok = {
        "geom_num_inliers": "45",
        "geom_is_valid": "true",
        "geom_inlier_ratio": "0.65",
        "disp_std_px": "5.0",
    }
    s_ok, r_ok = mon.evaluate(telemetry_ok, 10.0, 10.1)
    assert s_ok >= 0.90
    assert len(r_ok) == 0

    # Degraded inliers
    telemetry_deg = {
        "geom_num_inliers": "20",
        "geom_is_valid": "true",
        "geom_inlier_ratio": "0.50",
    }
    s_deg, r_deg = mon.evaluate(telemetry_deg, 10.0, 10.1)
    assert 0.40 <= s_deg <= 0.75
    assert any("DEGRADED_INLIERS" in r for r in r_deg)

    # Insufficient inliers
    telemetry_loss = {
        "geom_num_inliers": "5",
        "geom_is_valid": "false",
    }
    s_loss, r_loss = mon.evaluate(telemetry_loss, 10.0, 10.1)
    assert s_loss <= 0.20
    assert any("INSUFFICIENT_INLIERS" in r for r in r_loss)

    # Timeout
    s_to, r_to = mon.evaluate(telemetry_ok, 10.0, 12.5)
    assert s_to == 0.0
    assert any("TIMEOUT" in r for r in r_to)


def test_localization_monitor():
    mon = LocalizationMonitor()

    # Nominal
    diag_ok = {"tracking_state": "OK", "reprojection_error_px": "1.2", "num_landmarks": "25"}
    s_ok, r_ok = mon.evaluate(diag_ok, 10.0, 10.1)
    assert s_ok >= 0.90

    # Degraded
    diag_deg = {"tracking_state": "DEGRADED", "failure_reason": "LOW_KEYPOINTS"}
    s_deg, r_deg = mon.evaluate(diag_deg, 10.0, 10.1)
    assert 0.40 <= s_deg <= 0.60
    assert any("DEGRADED" in r for r in r_deg)

    # Tracking Lost
    diag_lost = {"tracking_state": "TRACKING_LOST", "failure_reason": "FEATURES_ZERO"}
    s_lost, r_lost = mon.evaluate(diag_lost, 10.0, 10.1)
    assert s_lost <= 0.10
    assert any("LOST" in r for r in r_lost)


def test_imu_monitor():
    mon = ImuMonitor()

    # Nominal gravity
    s_ok, r_ok = mon.evaluate((0.0, 0.0, 9.81), (0.0, 0.0, 0.0), 10.0, 10.05)
    assert s_ok == 1.0
    assert len(r_ok) == 0

    # Impact / shock
    s_shock, r_shock = mon.evaluate((20.0, 15.0, 10.0), (0.0, 0.0, 0.0), 10.0, 10.05)
    assert s_shock <= 0.25
    assert any("SHOCK" in r for r in r_shock)

    # NaN / Inf
    s_nan, r_nan = mon.evaluate((float("nan"), 0.0, 9.81), (0.0, 0.0, 0.0), 10.0, 10.05)
    assert s_nan == 0.0
    assert any("NAN_OR_INF" in r for r in r_nan)

    # Timeout
    s_to, r_to = mon.evaluate((0.0, 0.0, 9.81), (0.0, 0.0, 0.0), 10.0, 11.0)
    assert s_to == 0.0


def test_wheel_monitor():
    mon = WheelMonitor()

    # Nominal forward motion
    s_ok, r_ok = mon.evaluate(0.5, 0.0, 0.1, 10.0, 10.05)
    assert s_ok == 1.0

    # Severe lateral slip
    s_slip, r_slip = mon.evaluate(0.5, 0.4, 0.1, 10.0, 10.05)
    assert s_slip <= 0.40
    assert any("LATERAL_SLIP" in r for r in r_slip)

    # Overspeed
    s_speed, r_speed = mon.evaluate(2.8, 0.0, 0.0, 10.0, 10.05)
    assert s_speed <= 0.20
    assert any("OVERSPEED" in r for r in r_speed)


def test_temporal_monitor():
    mon = TemporalMonitor()

    # Nominal
    s_ok, _ = mon.evaluate({"overall_status": "PASS", "cam_imu_median_ms": "12.0"}, 10.0, 10.1)
    assert s_ok == 1.0

    # Jitter warning
    s_warn, r_warn = mon.evaluate({"overall_status": "WARNING", "cam_imu_median_ms": "75.0"}, 10.0, 10.1)
    assert s_warn <= 0.65
    assert any("WARNING" in r or "LATENCY" in r for r in r_warn)


def test_cross_sensor_monitor():
    mon = CrossSensorMonitor()

    # Nominal
    diag_ok = {
        "is_vis_wheel_consistent": "true",
        "is_vis_imu_consistent": "true",
        "is_wheel_imu_consistent": "true",
        "res_wz_wheel_imu_radps": "0.02",
        "rejections_visual": "0",
    }
    s_ok, _ = mon.evaluate(diag_ok, 10.0, 10.1)
    assert s_ok == 1.0

    # Wheel-IMU Disagreement
    diag_bad = dict(diag_ok)
    diag_bad["is_wheel_imu_consistent"] = "false"
    diag_bad["res_wz_wheel_imu_radps"] = "0.75"
    s_bad, r_bad = mon.evaluate(diag_bad, 10.0, 10.1)
    assert s_bad <= 0.40
    assert any("DISAGREEMENT" in r or "RESIDUAL" in r for r in r_bad)


def test_map_monitor():
    mon = MapMonitor()

    # Grid 10x10, resolution 0.1m, origin -0.5, -0.5
    # Robot at (0.0, 0.0) -> cell (5, 5)
    grid = [0] * 100

    # Clear map
    s_clear, _ = mon.evaluate(grid, 0.1, 10, 10, -0.5, -0.5, 0.0, 0.0, 10.0, 10.1)
    assert s_clear == 1.0

    # Obstacle at cell (5, 7) -> 0.2m away -> critical collision zone
    grid_crit = list(grid)
    grid_crit[7 * 10 + 5] = 100
    s_crit, r_crit = mon.evaluate(grid_crit, 0.1, 10, 10, -0.5, -0.5, 0.0, 0.0, 10.0, 10.1)
    assert s_crit <= 0.10
    assert any("CRITICAL" in r for r in r_crit)


def test_composite_confidence_engine_safety_gating():
    engine = CompositeConfidenceEngine()

    # Nominal case
    scores, prim_reason, _ = engine.compute(
        current_time=10.0,
        vo_telemetry={"geom_num_inliers": "50", "geom_is_valid": "true"},
        vo_last_time=9.95,
        slam_diag={"tracking_state": "OK", "reprojection_error_px": "1.0"},
        slam_last_time=9.95,
        imu_accel=(0.0, 0.0, 9.81),
        imu_gyro=(0.0, 0.0, 0.0),
        imu_last_time=9.98,
        wheel_twist=(0.3, 0.0, 0.0),
        wheel_last_time=9.98,
        sync_diag={"overall_status": "PASS"},
        sync_last_time=9.90,
        state_diag={"is_wheel_imu_consistent": "true", "is_vis_wheel_consistent": "true", "is_vis_imu_consistent": "true"},
        state_last_time=9.95,
        map_data=[0] * 100,
        map_res=0.1,
        map_w=10,
        map_h=10,
        map_ox=-0.5,
        map_oy=-0.5,
        robot_pose=(0.0, 0.0),
        map_last_time=9.90,
    )
    assert scores.overall >= 0.90
    assert prim_reason == "NOMINAL_OPERATION"

    # Critical IMU timeout -> triggers safety floor gating
    scores_imu_fail, prim_fail, _ = engine.compute(
        current_time=15.0,  # 5 seconds later without IMU update
        vo_telemetry={"geom_num_inliers": "50", "geom_is_valid": "true"},
        vo_last_time=14.95,
        slam_diag={"tracking_state": "OK"},
        slam_last_time=14.95,
        imu_accel=(0.0, 0.0, 9.81),
        imu_gyro=(0.0, 0.0, 0.0),
        imu_last_time=9.98,  # Old
        wheel_twist=(0.3, 0.0, 0.0),
        wheel_last_time=14.98,
        sync_diag={"overall_status": "PASS"},
        sync_last_time=14.90,
        state_diag={"is_wheel_imu_consistent": "true", "is_vis_wheel_consistent": "true", "is_vis_imu_consistent": "true"},
        state_last_time=14.95,
        map_data=[0] * 100,
        map_res=0.1,
        map_w=10,
        map_h=10,
        map_ox=-0.5,
        map_oy=-0.5,
        robot_pose=(0.0, 0.0),
        map_last_time=14.90,
    )
    assert scores_imu_fail.imu == 0.0
    assert scores_imu_fail.overall <= 0.25
    assert "IMU_TIMEOUT" in prim_fail
