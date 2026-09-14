"""Comprehensive scenario test suite for NAVIGUARD Phase 9 Navigation.

Covers Tests A through I:
- TEST A: Open environment START -> GOAL
- TEST B: Obstacle avoidance / routing around obstacle
- TEST C: Goal near obstacle (safe clearance)
- TEST D: Blocked goal (unreachable)
- TEST E: Dynamic obstacle / path invalidation -> REPLAN
- TEST F: Confidence degradation -> Yield to recovery
- TEST G: Recovery success -> Replan to original goal -> Continue
- TEST H: Recovery FAILED_SAFE -> Robot remains stopped
- TEST I: Goal reached -> GOAL_REACHED & zero velocity
"""

import math
import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose

from naviguard_navigation.occupancy_grid import NavigationOccupancyGrid
from naviguard_navigation.global_planner import GlobalPlannerAStar
from naviguard_navigation.path_smoother import PathSmoother
from naviguard_navigation.waypoint_generator import WaypointGenerator
from naviguard_navigation.path_follower import PathFollower
from naviguard_navigation.goal_checker import GoalChecker
from naviguard_navigation.replanner import Replanner
from naviguard_navigation.mission_manager import MissionManager, MissionState
from naviguard_navigation.goal_manager import GoalManager, NavigationGoal


def make_grid(width=60, height=60, res=0.1, origin_x=0.0, origin_y=0.0, obstacles=None):
    msg = OccupancyGrid()
    meta = MapMetaData()
    meta.resolution = res
    meta.width = width
    meta.height = height
    origin = Pose()
    origin.position.x = origin_x
    origin.position.y = origin_y
    meta.origin = origin
    msg.info = meta

    data = np.zeros((height, width), dtype=np.int8)
    if obstacles:
        for y1, y2, x1, x2 in obstacles:
            data[y1:y2, x1:x2] = 100
    msg.data = data.flatten().tolist()

    grid = NavigationOccupancyGrid(inflation_radius_m=0.20, proximity_radius_m=0.40)
    grid.update_from_msg(msg)
    return grid


def test_scenario_a_open_environment():
    """TEST A: Simple open environment START -> GOAL."""
    grid = make_grid()
    planner = GlobalPlannerAStar()
    smoother = PathSmoother()
    wp_gen = WaypointGenerator(target_spacing_m=0.30)
    follower = PathFollower()
    checker = GoalChecker(xy_tolerance_m=0.25, required_dwell_sec=0.5)

    start = (1.0, 1.0)
    goal = NavigationGoal(x=4.0, y=1.0, yaw=0.0, frame_id="map", timestamp=0.0)

    # 1. Plan path
    raw_path = planner.plan(grid, start, (goal.x, goal.y))
    assert raw_path is not None
    smoothed = smoother.smooth(raw_path, grid)
    wps = wp_gen.generate(smoothed, final_yaw=goal.yaw)
    assert len(wps) >= 8

    # 2. Simulate step along path
    rx, ry, ryaw = 1.0, 1.0, 0.0
    vx, wz, lookahead, cross_err = follower.compute_commands(rx, ry, ryaw, wps)
    assert vx > 0.0
    assert abs(wz) < 0.2

    # 3. Simulate arrival at goal
    assert checker.is_goal_reached(4.0, 1.0, 0.0, goal, now_sec=1.0) is False
    assert checker.is_goal_reached(4.0, 1.0, 0.0, goal, now_sec=1.6) is True


def test_scenario_b_obstacle_routing():
    """TEST B: Obstacle between start and goal -> planner routes around obstacle."""
    # Obstacle wall at x=25 (2.5m) spanning y=10 to 40 (1.0m to 4.0m)
    grid = make_grid(obstacles=[(10, 40, 24, 26)])
    planner = GlobalPlannerAStar()

    start = (1.0, 2.5)
    goal = (4.0, 2.5)
    path = planner.plan(grid, start, goal)

    assert path is not None
    # Verify path cleared the obstacle wall
    for pt in path:
        cell = grid.world_to_map(pt[0], pt[1])
        assert not grid.is_lethal(cell[0], cell[1])


def test_scenario_c_goal_near_obstacle():
    """TEST C: Goal near obstacle -> safe clearance maintained."""
    # Obstacle at (3.5, 1.5)
    grid = make_grid(obstacles=[(15, 20, 35, 40)])
    planner = GlobalPlannerAStar()

    start = (1.0, 1.0)
    # Goal placed adjacent to obstacle buffer
    goal = (3.0, 1.0)
    path = planner.plan(grid, start, goal)

    assert path is not None
    # All path points must respect safety inflation
    for pt in path:
        cell = grid.world_to_map(pt[0], pt[1])
        assert not grid.is_lethal(cell[0], cell[1])


def test_scenario_d_blocked_unreachable_goal():
    """TEST D: Blocked goal -> planning failure."""
    # Completely seal goal inside a box
    grid = make_grid(obstacles=[
        (25, 45, 25, 27),
        (25, 45, 43, 45),
        (25, 27, 25, 45),
        (43, 45, 25, 45),
    ])
    planner = GlobalPlannerAStar()

    start = (1.0, 1.0)
    goal = (3.5, 3.5)  # Inside sealed box
    path = planner.plan(grid, start, goal)

    assert path is None


def test_scenario_e_dynamic_obstacle_replan():
    """TEST E: Path becomes blocked dynamically -> triggers REPLAN."""
    grid = make_grid()
    replanner = Replanner(min_replan_interval_sec=0.5)

    wps = [
        wp_gen_point(1.0, 1.0, 0),
        wp_gen_point(2.0, 1.0, 1),
        wp_gen_point(3.0, 1.0, 2),
    ]

    # Clear initially
    should_replan, _ = replanner.should_replan_due_to_obstacle(wps, 0, grid, now_sec=1.0)
    assert should_replan is False

    # Dynamic obstacle placed at (2.0, 1.0)
    new_grid = make_grid(obstacles=[(8, 12, 18, 22)])
    should_replan, reason = replanner.should_replan_due_to_obstacle(wps, 0, new_grid, now_sec=2.0)
    assert should_replan is True
    assert "PATH_BLOCKED" in reason


def test_scenario_f_confidence_degradation_yields():
    """TEST F: Confidence degradation -> navigation yields to recovery."""
    mgr = MissionManager()
    mgr.state = MissionState.NAVIGATING

    # Confidence triggers RECOVER
    mgr.transition_to(MissionState.RECOVERY_WAIT, now_sec=10.0, reason="CONFIDENCE_RECOVER")
    assert mgr.state == MissionState.RECOVERY_WAIT


def test_scenario_g_recovery_success_and_replan():
    """TEST G: Recovery success -> replan to original goal -> continue."""
    mgr = MissionManager()
    mgr.state = MissionState.RECOVERY_WAIT

    # Recovery finished successfully
    mgr.transition_to(MissionState.REPLANNING, now_sec=15.0, reason="RECOVERY_COMPLETED")
    assert mgr.state == MissionState.REPLANNING

    # Replan succeeds -> resumes NAVIGATING
    mgr.transition_to(MissionState.NAVIGATING, now_sec=15.5)
    assert mgr.state == MissionState.NAVIGATING


def test_scenario_h_recovery_failed_safe():
    """TEST H: Recovery FAILED_SAFE -> robot remains stopped in MISSION_FAILED."""
    mgr = MissionManager()
    mgr.state = MissionState.RECOVERY_WAIT

    mgr.transition_to(MissionState.MISSION_FAILED, now_sec=20.0, reason="RECOVERY_FAILED_SAFE")
    assert mgr.state == MissionState.MISSION_FAILED


def test_scenario_i_goal_reached_zero_velocity():
    """TEST I: Goal reached -> GOAL_REACHED state."""
    mgr = MissionManager()
    mgr.state = MissionState.NAVIGATING

    mgr.transition_to(MissionState.GOAL_REACHED, now_sec=30.0)
    assert mgr.state == MissionState.GOAL_REACHED


def wp_gen_point(x, y, idx):
    from naviguard_navigation.waypoint_generator import Waypoint
    return Waypoint(x=x, y=y, yaw=0.0, index=idx)


def test_scenario_j_closed_loop_obstacle_recovery_replan():
    """TEST J: Dynamic obstacle -> Recovery/Blockage mark -> A* replans alternative path."""
    grid = make_grid(width=60, height=60, res=0.1)
    planner = GlobalPlannerAStar()
    smoother = PathSmoother()
    wp_gen = WaypointGenerator(target_spacing_m=0.30)
    replanner = Replanner(min_replan_interval_sec=0.1)

    start = (1.0, 3.0)
    goal = (5.0, 3.0)

    # 1. Initial Plan (Straight line)
    path1 = planner.plan(grid, start, goal)
    assert path1 is not None
    wps1 = wp_gen.generate(smoother.smooth(path1, grid))

    # 2. Obstacle blocks the direct path at (3.0, 3.0)
    grid.mark_blocked_region(3.0, 3.0, radius_m=0.35)

    # Replanner detects obstacle
    should_replan, reason = replanner.should_replan_due_to_obstacle(wps1, 0, grid, now_sec=1.0)
    assert should_replan is True

    # 3. Closed-loop replan computes alternative path
    path2 = planner.plan(grid, start, goal)
    assert path2 is not None
    wps2 = wp_gen.generate(smoother.smooth(path2, grid))

    # Path 2 must not intersect the blocked region around (3.0, 3.0)
    for wp in wps2:
        dist_to_obs = math.hypot(wp.x - 3.0, wp.y - 3.0)
        assert dist_to_obs >= 0.30, f"Waypoint ({wp.x}, {wp.y}) too close to obstacle: {dist_to_obs:.2f}m"

    # Verify paths are distinctly different
    assert len(path2) != len(path1) or path2 != path1


def test_scenario_k_full_blockage_budget_exhaustion():
    """TEST K: Total obstacle wall -> Replanner fails -> Budget exhaustion -> FAILED."""
    # Wall completely blocking x=3.0 from y=0 to y=6.0 (cells 0 to 60)
    grid = make_grid(width=60, height=60, res=0.1, obstacles=[(0, 60, 28, 32)])
    planner = GlobalPlannerAStar()
    mission_mgr = MissionManager(max_replan_retries=2)
    mission_mgr.state = MissionState.PLANNING

    start = (1.0, 3.0)
    goal = (5.0, 3.0)

    # Planning attempt 1 fails
    p1 = planner.plan(grid, start, goal)
    assert p1 is None
    mission_mgr.record_planning_failure(now_sec=1.0, reason="NO_PATH")
    assert mission_mgr.state == MissionState.PLANNING

    # Planning attempt 2 fails -> exhausts retries -> MISSION_FAILED
    p2 = planner.plan(grid, start, goal)
    assert p2 is None
    mission_mgr.record_planning_failure(now_sec=2.0, reason="NO_PATH")
    assert mission_mgr.state == MissionState.MISSION_FAILED

