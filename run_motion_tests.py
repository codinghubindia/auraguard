"""Automated motion verification suite for NAVIGUARD Visual Odometry (Phase 4 Geometric Motion).

Runs Tests A, B, C, D:
- Test A: Stationary baseline (residual pixel noise & degenerate baseline detection)
- Test B: Forward translation (v_x = 0.5 m/s, unit-t_z forward verification, scale UNKNOWN)
- Test C: Rotation (w_z = 0.8 rad/s, optical yaw rotation estimation)
- Test D: Deceleration & Stop recovery (cmd_vel = 0, recovery of stationary baseline)

Saves debug frames and outputs JSON telemetry for evaluation report.
"""

import json
import time
from typing import Dict, List, Optional
import cv2
from cv_bridge import CvBridge
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
from sensor_msgs.msg import Image


class MotionVerificationSuite(Node):
    def __init__(self):
        super().__init__('vo_motion_verifier')

        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # QoS Profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.latest_telemetry: Optional[Dict[str, float]] = None
        self.latest_telemetry_msg: Optional[DiagnosticArray] = None
        self.latest_debug_image: Optional[np.ndarray] = None
        self.latest_odom: Optional[Odometry] = None

        self.create_subscription(
            DiagnosticArray,
            '/visual_odometry/telemetry',
            self.telemetry_cb,
            10,
        )

        self.create_subscription(
            Image,
            '/visual_odometry/debug_image',
            self.debug_image_cb,
            sensor_qos,
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_cb,
            10,
        )

    def telemetry_cb(self, msg: DiagnosticArray):
        self.latest_telemetry_msg = msg
        if len(msg.status) > 0:
            vals = {}
            for kv in msg.status[0].values:
                try:
                    vals[kv.key] = float(kv.value)
                except ValueError:
                    vals[kv.key] = kv.value
            vals['status_msg'] = msg.status[0].message
            self.latest_telemetry = vals

    def debug_image_cb(self, msg: Image):
        try:
            self.latest_debug_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def send_cmd(self, vx: float, wz: float):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def wait_for_data(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_telemetry is not None and self.latest_debug_image is not None:
                return True
        return False

    def collect_samples(self, duration_sec: float, vx: float, wz: float, sample_rate_hz: float = 20.0):
        records = []
        interval = 1.0 / sample_rate_hz
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            self.send_cmd(vx, wz)
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.latest_telemetry is not None:
                rec = dict(self.latest_telemetry)
                if self.latest_odom is not None:
                    rec['actual_odom_vx'] = float(self.latest_odom.twist.twist.linear.x)
                    rec['actual_odom_wz'] = float(self.latest_odom.twist.twist.angular.z)
                records.append(rec)
            time.sleep(interval)

        return records


def run_suite():
    rclpy.init()
    suite = MotionVerificationSuite()

    print("Waiting for visual odometry telemetry and debug stream...")
    if not suite.wait_for_data(10.0):
        print("ERROR: Failed to receive VO data within 10s!")
        suite.destroy_node()
        rclpy.shutdown()
        return

    print("VO Data stream active! Beginning Phase 4 Geometric Motion Verification Tests...\n")
    results = {}

    # -------------------------------------------------------------
    # TEST A: Stationary Baseline
    # -------------------------------------------------------------
    print(">>> Executing TEST A: Stationary Robot Baseline (cmd_vel = 0) for 3.0s...")
    records_a = suite.collect_samples(duration_sec=3.0, vx=0.0, wz=0.0)
    if suite.latest_debug_image is not None:
        cv2.imwrite('/tmp/vo_stationary.png', suite.latest_debug_image)

    med_dx_a = [r['median_dx_px'] for r in records_a if 'median_dx_px' in r]
    med_dy_a = [r['median_dy_px'] for r in records_a if 'median_dy_px' in r]
    disp_a = [r['median_displacement_px'] for r in records_a if 'median_displacement_px' in r]
    inliers_a = [r['num_inliers'] for r in records_a if 'num_inliers' in r]
    geom_status_a = [r['geom_status'] for r in records_a if 'geom_status' in r]

    results['Test_A_Stationary'] = {
        'num_samples': len(records_a),
        'mean_inliers': float(np.mean(inliers_a)) if inliers_a else 0,
        'residual_noise_mean_dx_px': float(np.mean(med_dx_a)) if med_dx_a else 0,
        'residual_noise_mean_dy_px': float(np.mean(med_dy_a)) if med_dy_a else 0,
        'residual_disp_px': float(np.mean(disp_a)) if disp_a else 0,
        'dominant_geom_status': max(set(geom_status_a), key=geom_status_a.count) if geom_status_a else 'UNKNOWN',
        'translation_scale_status': 'UNKNOWN',
    }
    print(f"  Test A Finished: Mean inliers={results['Test_A_Stationary']['mean_inliers']:.1f}, "
          f"Residual disp={results['Test_A_Stationary']['residual_disp_px']:.3f} px, "
          f"Geom status={results['Test_A_Stationary']['dominant_geom_status']}")

    time.sleep(1.0)

    # -------------------------------------------------------------
    # TEST B: Forward Translation (v_x = 0.5 m/s)
    # -------------------------------------------------------------
    print("\n>>> Executing TEST B: Forward Motion (v_x = 0.5 m/s, w_z = 0.0) for 3.5s...")
    records_b = suite.collect_samples(duration_sec=3.5, vx=0.5, wz=0.0)
    if suite.latest_debug_image is not None:
        cv2.imwrite('/tmp/vo_forward.png', suite.latest_debug_image)

    disp_b = [r['median_displacement_px'] for r in records_b if 'median_displacement_px' in r]
    inliers_b = [r['num_inliers'] for r in records_b if 'num_inliers' in r]
    odom_vx_b = [r['actual_odom_vx'] for r in records_b if 'actual_odom_vx' in r]
    geom_status_b = [r['geom_status'] for r in records_b if 'geom_status' in r]
    geom_inl_b = [r['geom_num_inliers'] for r in records_b if 'geom_num_inliers' in r and r.get('geom_is_valid') == 'true']
    unit_tx_b = [r['unit_tx'] for r in records_b if 'unit_tx' in r and r.get('geom_is_valid') == 'true']
    unit_ty_b = [r['unit_ty'] for r in records_b if 'unit_ty' in r and r.get('geom_is_valid') == 'true']
    unit_tz_b = [r['unit_tz'] for r in records_b if 'unit_tz' in r and r.get('geom_is_valid') == 'true']
    rot_deg_b = [r['rot_angle_deg'] for r in records_b if 'rot_angle_deg' in r and r.get('geom_is_valid') == 'true']

    results['Test_B_Forward'] = {
        'num_samples': len(records_b),
        'mean_inliers': float(np.mean(inliers_b)) if inliers_b else 0,
        'mean_displacement_px': float(np.mean(disp_b)) if disp_b else 0,
        'reference_wheel_vx': float(np.mean(odom_vx_b)) if odom_vx_b else 0,
        'dominant_geom_status': max(set(geom_status_b), key=geom_status_b.count) if geom_status_b else 'UNKNOWN',
        'mean_geom_inliers': float(np.mean(geom_inl_b)) if geom_inl_b else 0,
        'mean_unit_tx': float(np.mean(unit_tx_b)) if unit_tx_b else 0.0,
        'mean_unit_ty': float(np.mean(unit_ty_b)) if unit_ty_b else 0.0,
        'mean_unit_tz': float(np.mean(unit_tz_b)) if unit_tz_b else 0.0,
        'mean_rot_deg': float(np.mean(rot_deg_b)) if rot_deg_b else 0.0,
        'translation_scale_status': 'UNKNOWN',
    }
    print(f"  Test B Finished: Flow disp={results['Test_B_Forward']['mean_displacement_px']:.2f} px, "
          f"Wheel vx={results['Test_B_Forward']['reference_wheel_vx']:.2f} m/s, "
          f"Unit t=[{results['Test_B_Forward']['mean_unit_tx']:+.2f}, "
          f"{results['Test_B_Forward']['mean_unit_ty']:+.2f}, "
          f"{results['Test_B_Forward']['mean_unit_tz']:+.2f}], "
          f"Scale={results['Test_B_Forward']['translation_scale_status']}")

    # Settle down
    suite.collect_samples(duration_sec=1.5, vx=0.0, wz=0.0)

    # -------------------------------------------------------------
    # TEST C: Pure Rotation (w_z = 0.8 rad/s)
    # -------------------------------------------------------------
    print("\n>>> Executing TEST C: Pure Rotation (v_x = 0.0, w_z = 0.8 rad/s) for 3.0s...")
    records_c = suite.collect_samples(duration_sec=3.0, vx=0.0, wz=0.8)
    if suite.latest_debug_image is not None:
        cv2.imwrite('/tmp/vo_turn.png', suite.latest_debug_image)

    med_dx_c = [r['median_dx_px'] for r in records_c if 'median_dx_px' in r]
    disp_c = [r['median_displacement_px'] for r in records_c if 'median_displacement_px' in r]
    inliers_c = [r['num_inliers'] for r in records_c if 'num_inliers' in r]
    odom_wz_c = [r['actual_odom_wz'] for r in records_c if 'actual_odom_wz' in r]
    geom_status_c = [r['geom_status'] for r in records_c if 'geom_status' in r]
    rot_yaw_c = [r['rot_yaw_deg'] for r in records_c if 'rot_yaw_deg' in r and r.get('geom_is_valid') == 'true']
    rot_angle_c = [r['rot_angle_deg'] for r in records_c if 'rot_angle_deg' in r and r.get('geom_is_valid') == 'true']

    results['Test_C_Rotation'] = {
        'num_samples': len(records_c),
        'mean_inliers': float(np.mean(inliers_c)) if inliers_c else 0,
        'mean_median_dx_px': float(np.mean(med_dx_c)) if med_dx_c else 0,
        'mean_displacement_px': float(np.mean(disp_c)) if disp_c else 0,
        'reference_wheel_wz': float(np.mean(odom_wz_c)) if odom_wz_c else 0,
        'dominant_geom_status': max(set(geom_status_c), key=geom_status_c.count) if geom_status_c else 'UNKNOWN',
        'mean_rot_yaw_deg': float(np.mean(rot_yaw_c)) if rot_yaw_c else 0.0,
        'mean_rot_angle_deg': float(np.mean(rot_angle_c)) if rot_angle_c else 0.0,
        'translation_scale_status': 'UNKNOWN',
    }
    print(f"  Test C Finished: Mean dx={results['Test_C_Rotation']['mean_median_dx_px']:.2f} px, "
          f"Wheel wz={results['Test_C_Rotation']['reference_wheel_wz']:.2f} rad/s, "
          f"Geom Yaw={results['Test_C_Rotation']['mean_rot_yaw_deg']:+.2f} deg, "
          f"Scale={results['Test_C_Rotation']['translation_scale_status']}")

    # -------------------------------------------------------------
    # TEST D: Deceleration & Stop
    # -------------------------------------------------------------
    print("\n>>> Executing TEST D: Deceleration & Stop (cmd_vel = 0) for 3.0s...")
    records_d = suite.collect_samples(duration_sec=3.0, vx=0.0, wz=0.0)
    if suite.latest_debug_image is not None:
        cv2.imwrite('/tmp/vo_stopped.png', suite.latest_debug_image)

    late_d = records_d[len(records_d)//2:]
    disp_d = [r['median_displacement_px'] for r in late_d if 'median_displacement_px' in r]
    inliers_d = [r['num_inliers'] for r in late_d if 'num_inliers' in r]
    geom_status_d = [r['geom_status'] for r in late_d if 'geom_status' in r]

    results['Test_D_Stop'] = {
        'num_samples': len(records_d),
        'settled_inliers': float(np.mean(inliers_d)) if inliers_d else 0,
        'settled_disp_px': float(np.mean(disp_d)) if disp_d else 0,
        'settled_geom_status': max(set(geom_status_d), key=geom_status_d.count) if geom_status_d else 'UNKNOWN',
        'translation_scale_status': 'UNKNOWN',
    }
    print(f"  Test D Finished: Settled inliers={results['Test_D_Stop']['settled_inliers']:.1f}, "
          f"Settled disp={results['Test_D_Stop']['settled_disp_px']:.3f} px, "
          f"Settled geom status={results['Test_D_Stop']['settled_geom_status']}")

    # Save to JSON
    out_path = '/tmp/vo_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nPhase 4 Geometric Motion Verification Tests completed successfully. Results saved to {out_path}")

    suite.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    run_suite()
