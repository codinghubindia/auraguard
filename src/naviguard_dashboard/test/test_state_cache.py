import pytest
from naviguard_dashboard.state_cache import StateCache


def test_state_cache_snapshot():
    cache = StateCache(max_events=10)
    cache.update_robot_pose(1.0, 2.0, 0.0, 0.5)
    cache.mission_state = "NAVIGATING"
    cache.confidence_state = "CONTINUE"
    cache.confidence_scores["overall"] = 0.92

    snap = cache.get_snapshot()
    assert snap["mission_state"] == "NAVIGATING"
    assert snap["confidence_state"] == "CONTINUE"
    assert snap["confidence_scores"]["overall"] == 0.92
    assert snap["robot_pose"]["x"] == 1.0
    assert snap["robot_pose"]["y"] == 2.0


def test_event_log_rolling_limit():
    cache = StateCache(max_events=5)
    for i in range(10):
        cache.add_event("TEST", f"Event {i}")

    snap = cache.get_snapshot()
    assert len(snap["events"]) == 5
    assert "Event 9" in snap["events"][0]["text"]


def test_subsystem_health_aggregation():
    cache = StateCache()
    # Initially all offline -> OFFLINE
    snap = cache.get_snapshot()
    assert snap["system_status"] == "OFFLINE"

    # Turn on 9 subsystems -> ONLINE
    for name in list(cache.subsystems.keys())[:9]:
        cache.update_subsystem_rate(name, 20.0, is_online=True)

    snap = cache.get_snapshot()
    assert snap["system_status"] == "ONLINE"
