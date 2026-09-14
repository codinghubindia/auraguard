import inspect
from naviguard_dashboard.dashboard_node import NaviguardDashboardNode


def test_no_cmd_vel_publisher():
    """Verify that naviguard_dashboard NEVER publishes to /cmd_vel."""
    src = inspect.getsource(NaviguardDashboardNode)
    # Must NOT have create_publisher(..., '/cmd_vel', ...)
    assert "create_publisher(Twist, '/cmd_vel'" not in src
    assert "create_publisher(geometry_msgs.msg.Twist, '/cmd_vel'" not in src
    assert "'/cmd_vel'" not in src
