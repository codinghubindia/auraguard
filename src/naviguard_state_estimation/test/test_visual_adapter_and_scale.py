import pytest
import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from naviguard_state_estimation.measurement_buffer import MeasurementBuffer, StampedMeasurement
from naviguard_state_estimation.visual_measurement_adapter import (
    VisualMeasurementAdapter,
    VisualAdapterConfig,
)


def make_vo_diagnostic_msg(
    stamp_sec: float,
    geom_valid: bool = True,
    inliers: int = 40,
    unit_tx: float = 0.0,
    unit_ty: float = 0.0,
    unit_tz: float = 1.0,
    rot_yaw_deg: float = 0.0,
    dt_sec: float = 0.04,
) -> DiagnosticArray:
    msg = DiagnosticArray()
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int((stamp_sec - int(stamp_sec)) * 1e9)
    msg.header.frame_id = "camera_link"

    st = DiagnosticStatus()
    st.name = "naviguard_visual_odometry: geometry"
    st.values = [
        KeyValue(key="geom_is_valid", value="true" if geom_valid else "false"),
        KeyValue(key="geom_status", value="SUCCESS" if geom_valid else "DEGENERATE"),
        KeyValue(key="geom_num_inliers", value=str(inliers)),
        KeyValue(key="dt_sec", value=str(dt_sec)),
        KeyValue(key="unit_tx", value=str(unit_tx)),
        KeyValue(key="unit_ty", value=str(unit_ty)),
        KeyValue(key="unit_tz", value=str(unit_tz)),
        KeyValue(key="rot_yaw_deg", value=str(rot_yaw_deg)),
        KeyValue(key="rot_angle_deg", value=str(abs(rot_yaw_deg))),
    ]
    msg.status.append(st)
    return msg


def test_coordinate_transformation():
    adapter = VisualMeasurementAdapter()

    # In optical frame:
    # unit_tz = 0.8 (forward along optical optical Z) -> base_link +X
    # unit_tx = 0.6 (right along optical X) -> base_link -Y
    # unit_ty = 0.0 -> base_link -Z
    # rot_yaw_deg = -10.0 (rightward turn in optical -> positive counter-clockwise turn in base_link)
    msg = make_vo_diagnostic_msg(
        stamp_sec=1.0,
        unit_tx=0.6,
        unit_ty=0.0,
        unit_tz=0.8,
        rot_yaw_deg=-10.0,
        dt_sec=0.05,
    )

    meas = adapter.process(msg, wheel_buffer=None)
    assert meas.is_valid is True
    assert pytest.approx(meas.data['unit_tx_base'], rel=1e-4) == 0.8
    assert pytest.approx(meas.data['unit_ty_base'], rel=1e-4) == -0.6
    assert pytest.approx(meas.data['unit_tz_base'], rel=1e-4) == 0.0

    # Yaw rate: yaw_base = -rot_yaw_opt = +10 deg = 0.17453 rad -> wz = 0.17453 / 0.05 = 3.49 rad/s
    expected_wz = np.radians(10.0) / 0.05
    assert pytest.approx(meas.data['wz'], rel=1e-3) == expected_wz


def test_scale_observation_valid_forward_motion():
    adapter = VisualMeasurementAdapter(VisualAdapterConfig(min_inliers=15, min_wheel_disp_m=0.02))

    wheel_buf = MeasurementBuffer()
    wheel_buf.add(StampedMeasurement(1.00, "wheel", "odom", {'vx': 0.5, 'wz': 0.0}))
    wheel_buf.add(StampedMeasurement(1.05, "wheel", "odom", {'vx': 0.5, 'wz': 0.0}))

    # Visual translation pointing purely forward in optical frame (+Z) -> base_link +X
    # First measurement at t=1.00
    msg0 = make_vo_diagnostic_msg(stamp_sec=1.00, unit_tz=1.0, dt_sec=0.05)
    adapter.process(msg0, wheel_buffer=wheel_buf)

    # Second measurement at t=1.05 (wheel traveled 0.5 * 0.05 = 0.025 m)
    msg1 = make_vo_diagnostic_msg(stamp_sec=1.05, unit_tz=1.0, dt_sec=0.05)
    meas1 = adapter.process(msg1, wheel_buffer=wheel_buf)

    assert meas1.is_valid is True
    assert meas1.data['scale_valid'] is True
    assert meas1.data['scale_status'] == "LOCAL_SCALE_OBSERVED"
    assert pytest.approx(meas1.data['wheel_disp_m'], rel=1e-2) == 0.025
    assert pytest.approx(meas1.data['metric_vx'], rel=1e-2) == 0.5


def test_scale_rejection_when_robot_stationary():
    adapter = VisualMeasurementAdapter(VisualAdapterConfig(min_wheel_disp_m=0.02))

    wheel_buf = MeasurementBuffer()
    wheel_buf.add(StampedMeasurement(1.00, "wheel", "odom", {'vx': 0.0, 'wz': 0.0}))
    wheel_buf.add(StampedMeasurement(1.05, "wheel", "odom", {'vx': 0.0, 'wz': 0.0}))

    msg = make_vo_diagnostic_msg(stamp_sec=1.05, unit_tz=1.0, dt_sec=0.05)
    meas = adapter.process(msg, wheel_buffer=wheel_buf)

    # Measurement remains valid for yaw, but scale observation is rejected
    assert meas.is_valid is True
    assert meas.data['scale_valid'] is False
    assert "ZERO_WHEEL_MOTION" in meas.data['scale_status']


def test_scale_rejection_on_direction_mismatch():
    adapter = VisualMeasurementAdapter(VisualAdapterConfig(max_direction_error_deg=45.0))

    wheel_buf = MeasurementBuffer()
    wheel_buf.add(StampedMeasurement(1.00, "wheel", "odom", {'vx': 0.5, 'wz': 0.0}))
    wheel_buf.add(StampedMeasurement(1.05, "wheel", "odom", {'vx': 0.5, 'wz': 0.0}))

    # Unit vector in optical frame pointing strongly lateral (unit_tx = 0.9, unit_tz = 0.1)
    # in base_link: t_x = 0.1, t_y = -0.9 -> angle ~ 83.6 degrees off forward heading (> 45 deg)
    msg = make_vo_diagnostic_msg(stamp_sec=1.05, unit_tx=0.9, unit_tz=0.1, dt_sec=0.05)
    meas = adapter.process(msg, wheel_buffer=wheel_buf)

    assert meas.is_valid is True
    assert meas.data['scale_valid'] is False
    assert "INCONSISTENT_DIRECTION" in meas.data['scale_status']


def test_measurement_rejection_on_degenerate_geometry():
    adapter = VisualMeasurementAdapter()

    msg = make_vo_diagnostic_msg(stamp_sec=1.05, geom_valid=False, inliers=5)
    meas = adapter.process(msg, wheel_buffer=None)

    assert meas.is_valid is False
    assert "GEOM" in meas.rejection_reason
