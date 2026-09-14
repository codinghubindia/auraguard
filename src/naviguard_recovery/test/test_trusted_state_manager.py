import pytest
from naviguard_recovery.trusted_state_manager import (
    TrustedCheckpoint,
    TrustedStateManager,
    TrustedStateManagerConfig,
)


def test_trusted_state_manager_add_and_gate():
    cfg = TrustedStateManagerConfig(min_displacement_m=0.25)
    mgr = TrustedStateManager(cfg)

    # 1. Valid high-confidence CONTINUE state
    added1 = mgr.maybe_add_checkpoint(
        timestamp_sec=1.0,
        map_pose=(0.0, 0.0, 0.0),
        odom_pose=(0.0, 0.0, 0.0),
        conf_overall=0.92,
        conf_loc=0.90,
        conf_vis=0.88,
        velocity=(0.3, 0.0),
        phase7_state="CONTINUE",
    )
    assert added1 is True
    assert len(mgr.checkpoints) == 1

    # 2. Too close (< 0.25m) -> rejected by displacement gate
    added_too_close = mgr.maybe_add_checkpoint(
        timestamp_sec=1.1,
        map_pose=(0.05, 0.0, 0.0),
        odom_pose=(0.05, 0.0, 0.0),
        conf_overall=0.92,
        conf_loc=0.90,
        conf_vis=0.88,
        velocity=(0.3, 0.0),
        phase7_state="CONTINUE",
    )
    assert added_too_close is False
    assert len(mgr.checkpoints) == 1

    # 3. Sufficient displacement (0.35m) -> accepted
    added2 = mgr.maybe_add_checkpoint(
        timestamp_sec=2.0,
        map_pose=(0.35, 0.0, 0.0),
        odom_pose=(0.35, 0.0, 0.0),
        conf_overall=0.90,
        conf_loc=0.88,
        conf_vis=0.85,
        velocity=(0.3, 0.0),
        phase7_state="CONTINUE",
    )
    assert added2 is True
    assert len(mgr.checkpoints) == 2

    # 4. Low confidence -> rejected
    added_low_conf = mgr.maybe_add_checkpoint(
        timestamp_sec=3.0,
        map_pose=(0.70, 0.0, 0.0),
        odom_pose=(0.70, 0.0, 0.0),
        conf_overall=0.60,
        conf_loc=0.55,
        conf_vis=0.50,
        velocity=(0.3, 0.0),
        phase7_state="CONTINUE",
    )
    assert added_low_conf is False

    # 5. Non-CONTINUE state -> rejected
    added_verify = mgr.maybe_add_checkpoint(
        timestamp_sec=4.0,
        map_pose=(1.00, 0.0, 0.0),
        odom_pose=(1.00, 0.0, 0.0),
        conf_overall=0.90,
        conf_loc=0.88,
        conf_vis=0.85,
        velocity=(0.3, 0.0),
        phase7_state="VERIFY",
    )
    assert added_verify is False


def test_trusted_state_manager_scoring_selection():
    mgr = TrustedStateManager()

    # Add 3 checkpoints along x-axis at x=0.0, x=0.8, x=1.6
    mgr.maybe_add_checkpoint(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.95, 0.95, 0.90, (0.3, 0.0), "CONTINUE")
    mgr.maybe_add_checkpoint(2.0, (0.8, 0.0, 0.0), (0.8, 0.0, 0.0), 0.92, 0.90, 0.88, (0.3, 0.0), "CONTINUE")
    mgr.maybe_add_checkpoint(3.0, (1.6, 0.0, 0.0), (1.6, 0.0, 0.0), 0.88, 0.85, 0.82, (0.3, 0.0), "CONTINUE")

    # Robot fails at x = 1.70.
    # Checkpoint at x=0.8 is ~0.9m away (near optimal backtrack distance 0.8m)
    # Checkpoint at x=1.6 is only 0.1m away (too close)
    # Checkpoint at x=0.0 is 1.7m away
    best = mgr.select_best_checkpoint(
        current_pose=(1.70, 0.0, 0.0),
        current_time=3.5,
    )
    assert best is not None
    # Best scored should be checkpoint 1 (x=0.8) due to Gaussian distance scoring
    assert best.checkpoint_id == 1
    assert best.map_pose[0] == 0.8
