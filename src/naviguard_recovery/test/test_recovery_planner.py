import pytest
from naviguard_recovery.recovery_planner import RecoveryPlanner, RecoveryStrategy
from naviguard_recovery.trusted_state_manager import TrustedCheckpoint


def test_recovery_planner_strategy_selection():
    planner = RecoveryPlanner()

    # 1. Critical hardware timeout -> STOP_AND_RELOCALIZE
    s1 = planner.select_strategy("IMU_TIMEOUT_0.60S", attempt_number=1, has_checkpoint=True)
    assert s1 == RecoveryStrategy.STOP_AND_RELOCALIZE

    # 2. Obstacle proximity -> SHORT_BACKTRACK
    s2 = planner.select_strategy("OBSTACLE_COLLISION_CRITICAL_0.20M", attempt_number=1, has_checkpoint=True)
    assert s2 == RecoveryStrategy.SHORT_BACKTRACK

    # 3. Visual degradation -> attempt 1 STOP_AND_RELOCALIZE, attempt 2 ROTATE_FOR_VISUAL_REACQUISITION
    s3_1 = planner.select_strategy("VO_DEGRADED_INLIERS_20", attempt_number=1, has_checkpoint=True)
    assert s3_1 == RecoveryStrategy.STOP_AND_RELOCALIZE
    s3_2 = planner.select_strategy("VO_DEGRADED_INLIERS_20", attempt_number=2, has_checkpoint=True)
    assert s3_2 == RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION

    # 4. SLAM lost with checkpoint -> SHORT_BACKTRACK
    s4 = planner.select_strategy("SLAM_TRACKING_LOST_FEATURES_0", attempt_number=1, has_checkpoint=True)
    assert s4 == RecoveryStrategy.SHORT_BACKTRACK

    # 5. SLAM lost without checkpoint -> ROTATE_FOR_VISUAL_REACQUISITION
    s5 = planner.select_strategy("SLAM_TRACKING_LOST_FEATURES_0", attempt_number=1, has_checkpoint=False)
    assert s5 == RecoveryStrategy.ROTATE_FOR_VISUAL_REACQUISITION


def test_backtrack_trajectory_and_collision():
    planner = RecoveryPlanner()
    ckpt = TrustedCheckpoint(
        checkpoint_id=0,
        timestamp_sec=1.0,
        map_pose=(0.0, 0.0, 0.0),
        odom_pose=(0.0, 0.0, 0.0),
        confidence_overall=0.9,
        confidence_loc=0.9,
        confidence_vis=0.9,
        velocity=(0.0, 0.0),
    )

    # Clear line from (1.0, 0.0, 0.0) back to (0.0, 0.0, 0.0)
    grid_clear = [0] * 100
    ok, wps, msg = planner.plan_backtrack(
        current_pose=(1.0, 0.0, 0.0),
        target_checkpoint=ckpt,
        grid_data=grid_clear,
        grid_res=0.2,
        grid_w=10,
        grid_h=10,
        grid_ox=-1.0,
        grid_oy=-1.0,
    )
    assert ok is True
    assert len(wps) >= 10
    assert wps[0][0] == 1.0
    assert abs(wps[-1][0] - 0.0) < 1e-4

    # Grid with obstacle blocking backtrack path at (0.4, 0.0)
    # Cell index: (0.4 - (-1.0)) / 0.2 = 7, cy = 5 -> idx = 5*10 + 7 = 57
    grid_blocked = list(grid_clear)
    grid_blocked[5 * 10 + 7] = 100
    ok_block, _, msg_block = planner.plan_backtrack(
        current_pose=(1.0, 0.0, 0.0),
        target_checkpoint=ckpt,
        grid_data=grid_blocked,
        grid_res=0.2,
        grid_w=10,
        grid_h=10,
        grid_ox=-1.0,
        grid_oy=-1.0,
    )
    assert ok_block is False
    assert "OBSTACLE" in msg_block


def test_360_lookaround_and_passages():
    planner = RecoveryPlanner()

    # 1. Blocked path triggers 360 lookaround
    s_blocked = planner.select_strategy("PATH_BLOCKED_AHEAD", attempt_number=1, has_checkpoint=True)
    assert s_blocked == RecoveryStrategy.LOOKAROUND_360_SCAN

    # 2. Obstacle retry on attempt 2 triggers 360 lookaround
    s_retry = planner.select_strategy("OBSTACLE_COLLISION_CRITICAL", attempt_number=2, has_checkpoint=True)
    assert s_retry == RecoveryStrategy.LOOKAROUND_360_SCAN

    # 3. Explicit lookaround request
    s_look = planner.select_strategy("LOOKAROUND_360_SCAN_REQUEST", attempt_number=1, has_checkpoint=True)
    assert s_look == RecoveryStrategy.LOOKAROUND_360_SCAN

    # 4. Plan 360 rotation sweep
    dyaw, tyaw, name = planner.plan_360_lookaround(current_yaw=0.5, direction=1)
    assert abs(dyaw - 2.0 * 3.14159265) < 1e-3
    assert abs(tyaw - (0.5 + 2.0 * 3.14159265)) < 1e-3
    assert name == "LOOKAROUND_360_SCAN"

    # 5. Evaluate 360 passages with a clear corridor towards +X
    # 20x20 grid, resolution 0.1m, origin (-1.0, -1.0)
    # Robot at (0.0, 0.0)
    grid = [0] * 400
    # Add obstacles north and south, leaving an east-west passage (width 1.0m > 0.48m)
    for cx in range(20):
        grid[18 * 20 + cx] = 100  # North wall at y ~ 0.8m
        grid[2 * 20 + cx] = 100   # South wall at y ~ -0.8m

    eval_res = planner.evaluate_360_passages(
        current_pose=(0.0, 0.0, 0.0),
        grid_data=grid,
        grid_res=0.1,
        grid_w=20,
        grid_h=20,
        grid_ox=-1.0,
        grid_oy=-1.0,
        goal_pose=(2.0, 0.0),
        vehicle_width_m=0.48,
    )
    assert eval_res["scan_complete"] is True
    assert len(eval_res["passages"]) > 0
    # Best heading should be directed towards forward-right / goal quadrant (< 0.60 rad / ~34 deg)
    assert abs(eval_res["best_heading_rad"]) < 0.60

