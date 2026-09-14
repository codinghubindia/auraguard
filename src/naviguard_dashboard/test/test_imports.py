import pytest

def test_imports():
    import naviguard_dashboard
    from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
    from naviguard_dashboard.goal_validator import GoalValidator
    from naviguard_dashboard.state_cache import StateCache
    from naviguard_dashboard.web_server import NaviguardRequestHandler
    assert MapCoordinateConverter is not None
    assert GoalValidator is not None
    assert StateCache is not None
    assert NaviguardRequestHandler is not None
