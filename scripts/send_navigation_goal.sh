#!/usr/bin/env bash
# Dispatches a target navigation goal to NAVIGUARD Phase 9 Navigation stack.

source /opt/ros/jazzy/setup.bash
source /home/maxx/naviguard_ws/install/setup.bash

if [ "$1" == "--cancel" ] || [ "$1" == "-c" ]; then
    echo ">>> Canceling active navigation mission..."
    ros2 service call /navigation/cancel_goal std_srvs/srv/Trigger "{}"
    exit 0
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <X_meters> <Y_meters> [YAW_radians]"
    echo "       $0 --cancel"
    echo ""
    echo "Example:"
    echo "  $0 2.5 0.5"
    echo "  $0 3.0 1.0 1.57"
    exit 1
fi

GOAL_X="$1"
GOAL_Y="$2"
GOAL_YAW="${3:-0.0}"

# Compute quaternion for yaw (rotation about Z)
python3 -c "
import math
import time
import rclpy
from geometry_msgs.msg import PoseStamped

rclpy.init()
node = rclpy.create_node('goal_sender')
pub = node.create_publisher(PoseStamped, '/goal_pose', 10)

msg = PoseStamped()
msg.header.stamp = node.get_clock().now().to_msg()
msg.header.frame_id = 'map'
msg.pose.position.x = float($GOAL_X)
msg.pose.position.y = float($GOAL_Y)
msg.pose.position.z = 0.0

yaw = float($GOAL_YAW)
msg.pose.orientation.x = 0.0
msg.pose.orientation.y = 0.0
msg.pose.orientation.z = math.sin(yaw * 0.5)
msg.pose.orientation.w = math.cos(yaw * 0.5)

# Wait for discovery
t0 = time.time()
while pub.get_subscription_count() == 0 and (time.time() - t0) < 3.0:
    time.sleep(0.1)

pub.publish(msg)
time.sleep(0.3)
print(f'>>> Navigation goal successfully published to /goal_pose: x={msg.pose.position.x:.2f}m, y={msg.pose.position.y:.2f}m, yaw={yaw:.2f}rad (frame: map)')
node.destroy_node()
rclpy.shutdown()
"
