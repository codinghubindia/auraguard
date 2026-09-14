import pytest
from naviguard_navigation.goal_checker import GoalChecker
from naviguard_navigation.goal_manager import NavigationGoal


def test_goal_checker_spatial_tolerance():
    checker = GoalChecker(xy_tolerance_m=0.20, yaw_tolerance_rad=0.30, required_dwell_sec=1.0)
    goal = NavigationGoal(x=2.0, y=1.0, yaw=0.0, frame_id="map", timestamp=0.0)

    # 1. Too far away
    assert checker.is_goal_reached(1.5, 1.0, 0.0, goal, now_sec=10.0) is False

    # 2. Inside spatial tolerance, but dwell hasn't elapsed
    assert checker.is_goal_reached(1.95, 1.02, 0.05, goal, now_sec=10.0) is False

    # 3. Inside tolerance, 0.5s elapsed (< 1.0s required)
    assert checker.is_goal_reached(1.95, 1.02, 0.05, goal, now_sec=10.5) is False

    # 4. Inside tolerance, 1.1s elapsed (>= 1.0s required)
    assert checker.is_goal_reached(1.95, 1.02, 0.05, goal, now_sec=11.1) is True


def test_goal_checker_dwell_resets_on_departure():
    checker = GoalChecker(xy_tolerance_m=0.20, required_dwell_sec=1.0)
    goal = NavigationGoal(x=2.0, y=1.0, yaw=None, frame_id="map", timestamp=0.0)

    # Enter goal region
    assert checker.is_goal_reached(2.0, 1.0, 0.0, goal, now_sec=1.0) is False

    # Robot drifts outside goal region
    assert checker.is_goal_reached(2.5, 1.0, 0.0, goal, now_sec=1.5) is False

    # Robot re-enters goal region: dwell timer must restart
    assert checker.is_goal_reached(2.0, 1.0, 0.0, goal, now_sec=2.0) is False
    assert checker.is_goal_reached(2.0, 1.0, 0.0, goal, now_sec=2.5) is False
    assert checker.is_goal_reached(2.0, 1.0, 0.0, goal, now_sec=3.1) is True
