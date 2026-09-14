import numpy as np
from naviguard_slam.slam_diagnostics import SlamDiagnosticsBuilder


def test_diagnostics_nominal_state():
    builder = SlamDiagnosticsBuilder()

    msg = builder.build_diagnostic_array(
        stamp_sec=10.0,
        mode="MAPPING",
        tracking_state="OK",
        pose_map=(1.5, 2.5, 0.3),
        features_tracked=120,
        reprojection_error_px=1.2,
        num_keyframes=15,
        num_landmarks=350,
        loop_closures_count=2,
        opt_error=0.045,
        map_to_odom_offset=(0.1, 0.05, 0.02),
        sensor_rates={'camera': 25.0, 'odom': 50.0, 'imu': 100.0},
        failure_reason="",
    )

    s_track = msg.status[0]
    lvl = ord(s_track.level) if isinstance(s_track.level, bytes) else s_track.level
    assert lvl == 0  # OK
    vals = {kv.key: kv.value for kv in s_track.values}
    assert vals['slam_mode'] == "MAPPING"
    assert vals['tracking_state'] == "OK"
    assert vals['features_tracked'] == "120"


def test_diagnostics_tracking_lost():
    builder = SlamDiagnosticsBuilder()

    msg = builder.build_diagnostic_array(
        stamp_sec=11.0,
        mode="MAPPING",
        tracking_state="TRACKING_LOST",
        pose_map=(1.5, 2.5, 0.3),
        features_tracked=4,
        reprojection_error_px=9.9,
        num_keyframes=15,
        num_landmarks=350,
        loop_closures_count=2,
        opt_error=0.045,
        map_to_odom_offset=(0.1, 0.05, 0.02),
        sensor_rates={'camera': 0.0, 'odom': 50.0, 'imu': 100.0},
        failure_reason="INSUFFICIENT_FEATURES_4",
    )

    s_track = msg.status[0]
    lvl = ord(s_track.level) if isinstance(s_track.level, bytes) else s_track.level
    assert lvl == 2  # ERROR
    vals = {kv.key: kv.value for kv in s_track.values}
    assert vals['tracking_state'] == "TRACKING_LOST"
    assert "INSUFFICIENT_FEATURES" in vals['failure_reason']
