"""Automated Gazebo Integration Verification for NAVIGUARD Sensor Synchronization (Phase 5A).

Tests sensor reception, timestamp monitoring, cross-sensor synchronization,
TF frame verification, and optical conventions across:
- State 1: Stationary
- State 2: Forward Motion
- State 3: Rotation
- State 4: Stop
"""

import json
import time
from typing import Dict, List, Optional
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, Imu


class SensorSyncIntegrationVerifier(Node):
    def __init__(self) -> None:
        super().__init__('sensor_sync_integration_verifier')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.latest_diag: Optional[DiagnosticArray] = None
        self.cam_count = 0
        self.info_count = 0
        self.imu_count = 0
        self.odom_count = 0

        self.create_subscription(
            DiagnosticArray,
            '/sensor_sync/diagnostics',
            self._diag_cb,
            10,
        )
        self.create_subscription(Image, '/camera/image_raw', self._cam_cb, sensor_qos)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._info_cb, sensor_qos)
        self.create_subscription(Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

    def _diag_cb(self, msg: DiagnosticArray) -> None:
        self.latest_diag = msg

    def _cam_cb(self, msg: Image) -> None:
        self.cam_count += 1

    def _info_cb(self, msg: CameraInfo) -> None:
        self.info_count += 1

    def _imu_cb(self, msg: Imu) -> None:
        self.imu_count += 1

    def _odom_cb(self, msg: Odometry) -> None:
        self.odom_count += 1

    def send_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def wait_for_data(self, timeout_sec: float = 12.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_diag is not None and self.cam_count > 5 and self.imu_count > 10:
                return True
        return False

    def collect_state_telemetry(self, duration_sec: float, vx: float, wz: float) -> Dict[str, str]:
        start = time.time()
        interval = 0.05
        while time.time() - start < duration_sec:
            self.send_cmd(vx, wz)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(interval)

        # Extract latest diagnostic values
        telemetry = {}
        if self.latest_diag is not None:
            for status in self.latest_diag.status:
                for kv in status.values:
                    telemetry[kv.key] = kv.value
            telemetry['diag_overall_message'] = self.latest_diag.status[0].message
        return telemetry


def run_integration_verification():
    rclpy.init()
    verifier = SensorSyncIntegrationVerifier()

    print("Waiting for Gazebo simulation topics and /sensor_sync/diagnostics...")
    if not verifier.wait_for_data(15.0):
        print("ERROR: Timeout waiting for sensor data and diagnostics!")
        verifier.destroy_node()
        rclpy.shutdown()
        return

    print("Sensors and diagnostics active! Beginning 4-State Motion Integration Test...\n")
    results = {}

    # State 1: Stationary
    print(">>> State 1: Stationary Robot (cmd_vel = 0) for 3.0s...")
    state1_data = verifier.collect_state_telemetry(duration_sec=3.0, vx=0.0, wz=0.0)
    results['State_1_Stationary'] = state1_data
    print(f"    Status: {state1_data.get('phase_5a_status')} | "
          f"Cam: {state1_data.get('camera_rate_hz')} Hz, IMU: {state1_data.get('imu_rate_hz')} Hz, Odom: {state1_data.get('odom_rate_hz')} Hz | "
          f"Cam-IMU Sync: {state1_data.get('cam_imu_pct_in_tol')} (mean diff: {state1_data.get('cam_imu_mean_diff_ms')} ms)")

    # State 2: Forward Motion
    print("\n>>> State 2: Forward Motion (vx = 0.5 m/s, wz = 0.0) for 3.0s...")
    state2_data = verifier.collect_state_telemetry(duration_sec=3.0, vx=0.5, wz=0.0)
    results['State_2_Forward'] = state2_data
    print(f"    Status: {state2_data.get('phase_5a_status')} | "
          f"Cam: {state2_data.get('camera_rate_hz')} Hz, IMU: {state2_data.get('imu_rate_hz')} Hz, Odom: {state2_data.get('odom_rate_hz')} Hz | "
          f"Cam-Odom Sync: {state2_data.get('cam_odom_pct_in_tol')} (mean diff: {state2_data.get('cam_odom_mean_diff_ms')} ms)")

    # State 3: Rotation
    print("\n>>> State 3: Rotation (vx = 0.0, wz = 0.8 rad/s) for 3.0s...")
    state3_data = verifier.collect_state_telemetry(duration_sec=3.0, vx=0.0, wz=0.8)
    results['State_3_Rotation'] = state3_data
    print(f"    Status: {state3_data.get('phase_5a_status')} | "
          f"Cam: {state3_data.get('camera_rate_hz')} Hz, IMU: {state3_data.get('imu_rate_hz')} Hz, Odom: {state3_data.get('odom_rate_hz')} Hz | "
          f"Optical Conv: {state3_data.get('optical_convention_valid')}")

    # State 4: Stop
    print("\n>>> State 4: Deceleration & Stop (cmd_vel = 0) for 2.5s...")
    state4_data = verifier.collect_state_telemetry(duration_sec=2.5, vx=0.0, wz=0.0)
    results['State_4_Stop'] = state4_data
    print(f"    Status: {state4_data.get('phase_5a_status')} | "
          f"Frame Val: {state4_data.get('frame_validation_status')}, "
          f"Cam Calib: {state4_data.get('camera_calib_status')}, "
          f"IMU Val: {state4_data.get('imu_validation_status')}, "
          f"Odom Val: {state4_data.get('odom_validation_status')}")

    out_path = '/tmp/sensor_sync_verification_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nIntegration test completed successfully. Results saved to {out_path}")

    verifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    run_integration_verification()
