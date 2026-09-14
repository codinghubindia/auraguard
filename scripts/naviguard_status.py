#!/usr/bin/env python3
"""NAVIGUARD Live System Status Dashboard with Phase 9 Mission Navigation."""

import json
import os
import sys
import time
from typing import Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class StatusMonitor(Node):
    def __init__(self):
        super().__init__('naviguard_status_monitor')

        self.last_decision = None
        self.last_recovery = None
        self.last_nav = None
        self.last_slam_pose = None
        self.last_odom = None

        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.create_subscription(String, '/naviguard/decision', self._cb_decision, qos_reliable)
        self.create_subscription(String, '/recovery/state', self._cb_recovery, qos_reliable)
        self.create_subscription(String, '/navigation/state', self._cb_nav, qos_reliable)
        self.create_subscription(PoseStamped, '/slam/pose', self._cb_slam_pose, qos_reliable)
        self.create_subscription(Odometry, '/state_estimation/odom', self._cb_odom, qos_reliable)

    def _cb_decision(self, msg: String):
        try:
            self.last_decision = json.loads(msg.data)
        except Exception:
            self.last_decision = {'decision': msg.data}

    def _cb_recovery(self, msg: String):
        try:
            self.last_recovery = json.loads(msg.data)
        except Exception:
            self.last_recovery = {'recovery_state': msg.data}

    def _cb_nav(self, msg: String):
        try:
            self.last_nav = json.loads(msg.data)
        except Exception:
            self.last_nav = {'mission_state': msg.data}

    def _cb_slam_pose(self, msg: PoseStamped):
        p = msg.pose.position
        self.last_slam_pose = (p.x, p.y, p.z)

    def _cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.last_odom = ((p.x, p.y), v.x)


def print_dashboard():
    rclpy.init()
    monitor = StatusMonitor()

    start = time.time()
    while time.time() - start < 1.5:
        rclpy.spin_once(monitor, timeout_sec=0.1)

    raw_nodes = monitor.get_node_names()
    node_names = [n.lstrip('/') for n in raw_nodes]

    expected_nodes = [
        'robot_state_publisher',
        'ros_gz_bridge',
        'naviguard_perception_node',
        'visual_odometry_node',
        'sensor_sync_node',
        'state_estimation_node',
        'slam_node',
        'confidence_node',
        'recovery_node',
        'navigation_node',
    ]

    print("\n" + "=" * 68)
    print("           NAVIGUARD LIVE SYSTEM STATUS DASHBOARD")
    print("=" * 68)

    # 1. Pipeline Nodes Health
    print("\n[1] PIPELINE NODES:")
    print(f"  {'Node Name':<30} | {'Status':<10}")
    print("  " + "-" * 42)
    active_count = 0
    for node in expected_nodes:
        is_active = any(n == node or n.endswith(node) for n in node_names)
        status = "ONLINE" if is_active else "OFFLINE"
        if is_active:
            active_count += 1
        print(f"  {node:<30} | {status:<10}")
    print(f"  >> {active_count}/{len(expected_nodes)} Core Nodes Active")

    # 2. Phase 9 Mission & Navigation State
    print("\n[2] MISSION & NAVIGATION (PHASE 9):")
    if monitor.last_nav:
        mstate = monitor.last_nav.get('mission_state', 'UNKNOWN')
        dist = monitor.last_nav.get('dist_to_goal_m', 0.0)
        curr_wp = monitor.last_nav.get('current_waypoint_idx', 0)
        tot_wp = monitor.last_nav.get('total_waypoints', 0)
        replan = monitor.last_nav.get('replan_count', 0)
        vx = monitor.last_nav.get('cmd_vx', 0.0)
        wz = monitor.last_nav.get('cmd_wz', 0.0)
        own = monitor.last_nav.get('cmd_ownership', 'N/A')
        goal = monitor.last_nav.get('goal')
        goal_str = f"({goal['x']:.2f}, {goal['y']:.2f})" if goal else "None"
        print(f"  Mission State       : {mstate} [Ownership: {own}]")
        print(f"  Target Destination  : {goal_str} (Distance: {dist:.2f} m)")
        print(f"  Path Progress       : Waypoint {curr_wp}/{tot_wp} (Replans: {replan})")
        print(f"  Navigation Velocity : cmd_vx={vx:.2f} m/s, cmd_wz={wz:.2f} rad/s")
    else:
        print("  Mission State       : WAITING (/navigation/state)")

    # 3. Confidence & Recovery States
    print("\n[3] CONFIDENCE & RECOVERY (PHASES 7 & 8):")
    if monitor.last_decision:
        dec = monitor.last_decision.get('decision', 'UNKNOWN')
        conf = monitor.last_decision.get('composite_confidence', 0.0)
        print(f"  Confidence Decision : {dec} (Composite Score: {conf:.2f})")
    else:
        print("  Confidence Decision : WAITING (/naviguard/decision)")

    if monitor.last_recovery:
        rec_state = monitor.last_recovery.get('recovery_state', 'UNKNOWN')
        strat = monitor.last_recovery.get('active_strategy', 'None')
        dwell = monitor.last_recovery.get('dwell_sec', 0.0)
        print(f"  Recovery State      : {rec_state} (Strategy: {strat}, Dwell: {dwell:.1f}s)")
    else:
        print("  Recovery State      : WAITING (/recovery/state)")

    # 4. Pose & Odometry
    print("\n[4] LOCALIZATION & ESTIMATION:")
    if monitor.last_slam_pose:
        sp = monitor.last_slam_pose
        print(f"  SLAM Global Pose    : x={sp[0]:.2f}, y={sp[1]:.2f}, z={sp[2]:.2f} m")
    else:
        print("  SLAM Global Pose    : WAITING (/slam/pose)")

    if monitor.last_odom:
        pos, vx = monitor.last_odom
        print(f"  State Estimation    : pos=({pos[0]:.2f}, {pos[1]:.2f}) m, vx={vx:.2f} m/s")
    else:
        print("  State Estimation    : WAITING (/state_estimation/odom)")

    print("\n" + "=" * 68 + "\n")

    monitor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    print_dashboard()
