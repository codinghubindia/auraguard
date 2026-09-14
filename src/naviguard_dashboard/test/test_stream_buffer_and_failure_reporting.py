import time
import pytest
from naviguard_dashboard.state_cache import StateCache, StreamBuffer


def test_stream_buffer_size_one_latest_frame_overwrite():
    """Verify StreamBuffer keeps only size=1 latest frame and drops older unread frames."""
    buf = StreamBuffer(name="test_stream", target_fps=15.0)

    # Push frame 1
    buf.update_frame(b"frame_1", timestamp=10.0, encode_latency_ms=4.2, source_fps=15.0)
    # Push frame 2 without consuming frame 1
    buf.update_frame(b"frame_2", timestamp=10.1, encode_latency_ms=3.8, source_fps=15.0)

    # Consume
    data, ts, age_ms, status = buf.consume_latest()
    assert data == b"frame_2"
    assert ts == 10.1


def test_stream_buffer_degraded_threshold():
    """Verify status switches to DEGRADED when frame age exceeds 350ms."""
    buf = StreamBuffer(name="test_stream", target_fps=15.0)
    past_time = time.time() - 1.5
    buf.update_frame(b"old_frame", timestamp=past_time, encode_latency_ms=5.0, source_fps=10.0)

    data, ts, age_ms, status = buf.consume_latest()
    assert data == b"old_frame"
    assert status == "DEGRADED"
    assert age_ms >= 1000.0


def test_state_cache_failure_and_success_records():
    """Verify StateCache correctly records and exports structured failure and success reports."""
    cache = StateCache()

    # Initial snapshot has None for records
    snap = cache.get_snapshot()
    assert snap.get("failure_record") is None
    assert snap.get("success_record") is None

    # Update failure record
    fail_rec = {
        "failure_code": "NO_SAFE_PATH",
        "human_reason": "No safe collision-free path exists to the goal.",
        "category": "NAVIGATION",
        "recovery_attempts": 3,
        "recovery_max_attempts": 3,
    }
    cache.set_failure_record(fail_rec)

    snap = cache.get_snapshot()
    assert snap["failure_record"] == fail_rec
    assert snap["failure_record"]["failure_code"] == "NO_SAFE_PATH"
    assert snap["mission_state"] == "FAILED"

    # Reset clears records
    cache.clear_failure_and_success()
    snap = cache.get_snapshot()
    assert snap.get("failure_record") is None
