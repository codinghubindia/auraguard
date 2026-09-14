import pytest
from naviguard_recovery.relocalization_manager import RelocalizationManager


def test_relocalization_manager_dwell_requirement():
    mgr = RelocalizationManager(min_dwell_sec=1.0)
    mgr.reset()

    # First frame with healthy confidence -> starts dwell, returns False
    ok1, dwell1 = mgr.evaluate_verification(1.0, conf_overall=0.85, conf_loc=0.80, conf_vis=0.70, slam_tracking_state="OK")
    assert ok1 is False
    assert dwell1 == 0.0

    # 0.5s later -> still in dwell, returns False
    ok2, dwell2 = mgr.evaluate_verification(1.5, conf_overall=0.85, conf_loc=0.80, conf_vis=0.70, slam_tracking_state="OK")
    assert ok2 is False
    assert abs(dwell2 - 0.5) < 1e-4

    # 1.1s later (total dwell 1.1s >= 1.0s) -> returns True
    ok3, dwell3 = mgr.evaluate_verification(2.1, conf_overall=0.85, conf_loc=0.80, conf_vis=0.70, slam_tracking_state="OK")
    assert ok3 is True
    assert abs(dwell3 - 1.1) < 1e-4


def test_relocalization_manager_confidence_dip_resets_dwell():
    mgr = RelocalizationManager(min_dwell_sec=1.0)
    mgr.reset()

    # Frame 1: healthy
    mgr.evaluate_verification(1.0, 0.85, 0.80, 0.70, "OK")

    # Frame 2: healthy at 1.6s (dwell 0.6s)
    mgr.evaluate_verification(1.6, 0.85, 0.80, 0.70, "OK")

    # Frame 3: confidence dips below threshold at 1.8s
    ok_dip, dwell_dip = mgr.evaluate_verification(1.8, 0.50, 0.40, 0.40, "OK")
    assert ok_dip is False
    assert dwell_dip == 0.0
    assert mgr.dwell_start_time is None
