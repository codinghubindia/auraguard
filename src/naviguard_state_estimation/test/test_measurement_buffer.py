import pytest
import numpy as np
from naviguard_state_estimation.measurement_buffer import MeasurementBuffer, StampedMeasurement


def test_buffer_insertion_and_ordering():
    buf = MeasurementBuffer(max_size=10, max_age_sec=5.0)

    m1 = StampedMeasurement(1.0, "wheel", "odom", {'vx': 0.1})
    m3 = StampedMeasurement(3.0, "wheel", "odom", {'vx': 0.3})
    m2 = StampedMeasurement(2.0, "wheel", "odom", {'vx': 0.2})

    assert buf.add(m1) is True
    assert buf.add(m3) is True
    assert buf.add(m2) is True

    # Buffer should be sorted by stamp
    assert len(buf) == 3
    assert [m.stamp_sec for m in buf.buffer] == [1.0, 2.0, 3.0]
    assert buf.get_latest().stamp_sec == 3.0


def test_buffer_invalid_timestamp():
    buf = MeasurementBuffer()
    m_zero = StampedMeasurement(0.0, "imu", "imu_link", {})
    m_neg = StampedMeasurement(-1.0, "imu", "imu_link", {})
    m_nan = StampedMeasurement(float('nan'), "imu", "imu_link", {})

    assert buf.add(m_zero) is False
    assert buf.add(m_neg) is False
    assert buf.add(m_nan) is False
    assert len(buf) == 0


def test_buffer_capacity_and_age_pruning():
    buf = MeasurementBuffer(max_size=5, max_age_sec=2.0)

    for i in range(10):
        buf.add(StampedMeasurement(10.0 + i * 0.1, "imu", "imu_link", {'i': i}))

    # Max size is 5
    assert len(buf) <= 5
    assert buf.buffer[-1].stamp_sec == 10.9

    # Add measurement with large time jump: old ones beyond max_age_sec should prune
    buf.add(StampedMeasurement(20.0, "imu", "imu_link", {}))
    assert buf.buffer[0].stamp_sec >= 18.0


def test_buffer_nearest_neighbor():
    buf = MeasurementBuffer()
    for t in [1.0, 1.05, 1.10, 1.15, 1.20]:
        buf.add(StampedMeasurement(t, "wheel", "odom", {'t': t}))

    # Query close to 1.09
    nearest = buf.get_nearest(1.09, max_time_diff_sec=0.03)
    assert nearest is not None
    assert nearest.stamp_sec == 1.10

    # Query with exceeding time diff
    too_far = buf.get_nearest(1.50, max_time_diff_sec=0.05)
    assert too_far is None


def test_wheel_motion_integration():
    buf = MeasurementBuffer()
    # Robot traveling at constant 0.5 m/s, turning at 0.1 rad/s
    t_vals = np.linspace(10.0, 11.0, 11)  # 10.0, 10.1, ..., 11.0
    for t in t_vals:
        buf.add(StampedMeasurement(t, "wheel", "odom", {'vx': 0.5, 'wz': 0.1}))

    dist, dyaw, mean_vx = buf.integrate_wheel_motion(10.0, 11.0)
    assert pytest.approx(dist, rel=1e-2) == 0.5
    assert pytest.approx(dyaw, rel=1e-2) == 0.1
    assert pytest.approx(mean_vx, rel=1e-2) == 0.5
