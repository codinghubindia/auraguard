"""Deterministic demo motion controller node for NAVIGUARD.

Provides smooth, bounded motion profiles for visualization and integration
testing. Subscribes to /recovery/state and immediately yields /cmd_vel whenever
the robot enters non-NORMAL states (e.g. SAFE_STOP, RECOVER, FAILED_SAFE).
"""

import json
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class NaviguardDemoMotionNode(Node):
    """Deterministic motion generator for NAVIGUARD demonstrations."""

    def __init__(self, node_name: str = 'naviguard_demo_motion') -> None:
        super().__init__(node_name)

        # Parameters
        self.declare_parameter('linear_velocity', 0.25)
        self.declare_parameter('angular_velocity', 0.25)
        self.declare_parameter('pattern', 'loop')  # 'loop', 'straight', 'turn'
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('enabled', True)

        self.linear_velocity = float(self.get_parameter('linear_velocity').value)
        self.angular_velocity = float(self.get_parameter('angular_velocity').value)
        self.pattern = str(self.get_parameter('pattern').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.enabled = bool(self.get_parameter('enabled').value)

        # Safety clamps
        self.linear_velocity = max(0.0, min(self.linear_velocity, 0.35))
        self.angular_velocity = max(0.0, min(self.angular_velocity, 0.35))

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/demo_motion/status', 10)

        # Subscribers
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.recovery_sub = self.create_subscription(
            String,
            '/recovery/state',
            self._recovery_state_callback,
            qos,
        )

        # State tracking
        self.recovery_state: str = 'NORMAL'
        self.is_active: bool = True
        self.current_vx: float = 0.0
        self.current_wz: float = 0.0

        # Pattern timing
        self.pattern_start_time: Optional[float] = None
        self.segment_duration_forward: float = 6.0  # seconds
        self.segment_duration_turn: float = 4.0     # seconds

        timer_period = 1.0 / max(1.0, self.publish_rate)
        self.timer = self.create_timer(timer_period, self._control_loop)

        self.get_logger().info(
            f'Naviguard Demo Motion Node initialized: pattern={self.pattern}, '
            f'v_max={self.linear_velocity:.2f} m/s, w_max={self.angular_velocity:.2f} rad/s'
        )

    def _recovery_state_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.recovery_state = data.get('recovery_state', 'NORMAL')
        except Exception:
            self.recovery_state = msg.data.strip()

        # If recovery is not NORMAL, immediately yield control
        if self.recovery_state not in ('NORMAL', 'VERIFY'):
            if self.is_active:
                self.get_logger().warn(
                    f'Recovery state is [{self.recovery_state}]. Demo motion yielding /cmd_vel immediately.'
                )
                self.is_active = False
                self.current_vx = 0.0
                self.current_wz = 0.0
        else:
            if not self.is_active and self.enabled:
                self.get_logger().info(
                    f'Recovery state resumed to [{self.recovery_state}]. Demo motion active.'
                )
                self.is_active = True
                self.pattern_start_time = None

    def _control_loop(self) -> None:
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        # When inactive or disabled, do not publish to /cmd_vel so recovery node has full control
        if not self.is_active or not self.enabled or self.recovery_state not in ('NORMAL', 'VERIFY'):
            status_msg = String()
            status_msg.data = json.dumps({
                'active': False,
                'recovery_state': self.recovery_state,
                'target_vx': 0.0,
                'target_wz': 0.0,
                'mode': 'YIELDING_TO_RECOVERY',
            })
            self.status_pub.publish(status_msg)
            return

        if self.pattern_start_time is None:
            self.pattern_start_time = now_sec

        elapsed = now_sec - self.pattern_start_time
        cycle_period = self.segment_duration_forward + self.segment_duration_turn
        cycle_time = elapsed % cycle_period

        target_vx = 0.0
        target_wz = 0.0
        mode_str = 'FORWARD'

        if self.pattern == 'straight':
            target_vx = self.linear_velocity
            target_wz = 0.0
            mode_str = 'STRAIGHT'
        elif self.pattern == 'turn':
            target_vx = 0.05
            target_wz = self.angular_velocity
            mode_str = 'TURN_ONLY'
        else:  # 'loop' (patrol sequence)
            if cycle_time < self.segment_duration_forward:
                target_vx = self.linear_velocity
                target_wz = 0.0
                mode_str = f'LOOP_FORWARD ({cycle_time:.1f}/{self.segment_duration_forward:.1f}s)'
            else:
                target_vx = 0.05
                target_wz = self.angular_velocity
                turn_elapsed = cycle_time - self.segment_duration_forward
                mode_str = f'LOOP_TURN ({turn_elapsed:.1f}/{self.segment_duration_turn:.1f}s)'

        # Slew rate filtering for smooth accelerations
        dt = 1.0 / max(1.0, self.publish_rate)
        max_accel_lin = 0.5 * dt  # 0.5 m/s^2
        max_accel_ang = 0.8 * dt  # 0.8 rad/s^2

        dvx = target_vx - self.current_vx
        self.current_vx += max(-max_accel_lin, min(max_accel_lin, dvx))

        dwz = target_wz - self.current_wz
        self.current_wz += max(-max_accel_ang, min(max_accel_ang, dwz))

        twist = Twist()
        twist.linear.x = float(self.current_vx)
        twist.angular.z = float(self.current_wz)
        self.cmd_vel_pub.publish(twist)

        status_msg = String()
        status_msg.data = json.dumps({
            'active': True,
            'recovery_state': self.recovery_state,
            'target_vx': round(self.current_vx, 3),
            'target_wz': round(self.current_wz, 3),
            'mode': mode_str,
            'cycle_time_sec': round(cycle_time, 2),
        })
        self.status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NaviguardDemoMotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
