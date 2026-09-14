"""Unit tests for SensorTimestampMonitor."""

import pytest
from naviguard_sensor_sync.timestamp_monitor import SensorTimestampMonitor


def test_monotonic_timestamps():
    """Verify strictly increasing timestamps are classified as monotonic."""
    monitor = SensorTimestampMonitor('test_sensor', expected_rate_hz=30.0)

    t0 = 100.0
    for i in range(20):
        t = t0 + i * (1.0 / 30.0)
        monitor.add_sample(stamp_sec=t, arrival_sec=t + 0.002, frame_id='test_frame', msg_type='test_msg')

    stats = monitor.compute_stats(current_time_sec=t0 + 20 * (1.0 / 30.0))
    assert stats.is_monotonic is True
    assert stats.num_jumps == 0
    assert stats.status == "OK"
    assert stats.total_count == 20


def test_non_monotonic_and_jumps():
    """Verify backward timestamp jumps are flagged and set status to FAIL."""
    monitor = SensorTimestampMonitor('test_sensor', expected_rate_hz=30.0)

    monitor.add_sample(stamp_sec=10.0, arrival_sec=10.001, frame_id='test_frame', msg_type='test_msg')
    monitor.add_sample(stamp_sec=10.033, arrival_sec=10.034, frame_id='test_frame', msg_type='test_msg')
    # Backward jump: 9.5 < 10.033
    monitor.add_sample(stamp_sec=9.5, arrival_sec=10.067, frame_id='test_frame', msg_type='test_msg')

    stats = monitor.compute_stats()
    assert stats.is_monotonic is False
    assert stats.num_jumps == 1
    assert stats.status == "FAIL"


def test_timestamp_gap_detection():
    """Verify temporal gaps exceeding gap_threshold_factor flag a warning."""
    # 50 Hz -> nominal period 0.02s. gap_threshold_factor=2.5 -> gap thresh 0.05s
    monitor = SensorTimestampMonitor('test_sensor', expected_rate_hz=50.0, gap_threshold_factor=2.5)

    monitor.add_sample(stamp_sec=1.00, arrival_sec=1.001, frame_id='test_frame', msg_type='test_msg')
    monitor.add_sample(stamp_sec=1.02, arrival_sec=1.021, frame_id='test_frame', msg_type='test_msg')
    # Large gap: 1.12 - 1.02 = 0.10s (> 0.05s)
    monitor.add_sample(stamp_sec=1.12, arrival_sec=1.121, frame_id='test_frame', msg_type='test_msg')

    stats = monitor.compute_stats()
    assert stats.num_gaps == 1
    assert stats.status == "WARN"


def test_sensor_rate_calculation():
    """Verify rate and period statistics for a known 100 Hz signal."""
    monitor = SensorTimestampMonitor('imu', expected_rate_hz=100.0)

    t0 = 50.0
    dt = 0.010  # 10ms = 100 Hz
    for i in range(50):
        t = t0 + i * dt
        monitor.add_sample(stamp_sec=t, arrival_sec=t + 0.001, frame_id='imu_link', msg_type='sensor_msgs/Imu')

    stats = monitor.compute_stats(current_time_sec=t0 + 49 * dt)
    assert stats.current_rate_hz == pytest.approx(100.0, abs=1.0)
    assert stats.mean_period_sec == pytest.approx(0.010, abs=1e-4)
    assert stats.std_period_sec == pytest.approx(0.0, abs=1e-5)
    assert stats.status == "OK"


def test_stale_sensor_detection():
    """Verify sensor with no recent arrivals is flagged as STALE."""
    monitor = SensorTimestampMonitor('test_sensor', expected_rate_hz=30.0)

    monitor.add_sample(stamp_sec=1.00, arrival_sec=1.00, frame_id='test_frame', msg_type='test_msg')
    monitor.add_sample(stamp_sec=1.033, arrival_sec=1.033, frame_id='test_frame', msg_type='test_msg')

    # Query stats 2.0s later (nominal period is ~0.033s)
    stats = monitor.compute_stats(current_time_sec=3.0)
    assert stats.status == "STALE"
