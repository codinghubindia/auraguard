"""Unit tests for CrossSensorSyncAnalyzer."""

import pytest
from naviguard_sensor_sync.sync_analyzer import CrossSensorSyncAnalyzer


def test_nearest_timestamp_pairing():
    """Verify pairing finds nearest target sample and calculates offsets accurately."""
    analyzer = CrossSensorSyncAnalyzer(tolerance_ms=20.0)

    # Reference at 10 Hz (every 100ms)
    stamps_ref = [1.000, 1.100, 1.200, 1.300]
    # Target at 50 Hz (offset by +5ms from each reference)
    stamps_target = [0.985, 1.005, 1.025, 1.085, 1.105, 1.125, 1.185, 1.205, 1.285, 1.305]

    stats = analyzer.evaluate_pair(stamps_ref, stamps_target, "test_pair")
    assert stats.status == "OK"
    assert stats.num_paired_samples == 4
    # All nearest distances should be exactly 5.0 ms
    assert stats.mean_abs_diff_ms == pytest.approx(5.0, abs=0.1)
    assert stats.median_abs_diff_ms == pytest.approx(5.0, abs=0.1)
    assert stats.pct_within_tolerance == 100.0


def test_sync_tolerance_classification_pass():
    """Verify high percentage within tolerance produces status OK."""
    analyzer = CrossSensorSyncAnalyzer(tolerance_ms=20.0)

    stamps_ref = [10.0, 10.1, 10.2, 10.3, 10.4]
    # All offsets around 8ms
    stamps_target = [t + 0.008 for t in stamps_ref]

    stats = analyzer.evaluate_pair(stamps_ref, stamps_target, "pass_pair")
    assert stats.status == "OK"
    assert stats.pct_within_tolerance == 100.0


def test_sync_tolerance_classification_fail():
    """Verify offsets well beyond tolerance trigger status FAIL."""
    analyzer = CrossSensorSyncAnalyzer(tolerance_ms=20.0)

    stamps_ref = [10.0, 10.1, 10.2, 10.3, 10.4]
    # Offsets of 60ms (> 20ms tolerance)
    stamps_target = [t + 0.060 for t in stamps_ref]

    stats = analyzer.evaluate_pair(stamps_ref, stamps_target, "fail_pair")
    assert stats.status == "FAIL"
    assert stats.pct_within_tolerance == 0.0


def test_insufficient_samples_handling():
    """Verify graceful handling when fewer than 3 samples are provided."""
    analyzer = CrossSensorSyncAnalyzer(tolerance_ms=20.0)

    stats = analyzer.evaluate_pair([1.0], [1.0], "empty_pair")
    assert stats.status == "INSUFFICIENT_DATA"
    assert stats.num_paired_samples == 0
