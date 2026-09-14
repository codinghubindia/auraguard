#!/usr/bin/env bash
# Inspects and displays all core NAVIGUARD topics, their types, and active status.

source /opt/ros/jazzy/setup.bash
source /home/maxx/naviguard_ws/install/setup.bash

echo "================================================================="
echo " NAVIGUARD Topic Inventory & Interface Registry (Phases 1-9)"
echo "================================================================="
printf "%-32s | %-32s\n" "TOPIC" "MESSAGE TYPE"
echo "---------------------------------+---------------------------------"

CORE_TOPICS=(
    "/cmd_vel:geometry_msgs/msg/Twist"
    "/odom:nav_msgs/msg/Odometry"
    "/camera/image_raw:sensor_msgs/msg/Image"
    "/camera/camera_info:sensor_msgs/msg/CameraInfo"
    "/imu:sensor_msgs/msg/Imu"
    "/joint_states:sensor_msgs/msg/JointState"
    "/tf:tf2_msgs/msg/TFMessage"
    "/tf_static:tf2_msgs/msg/TFMessage"
    "/perception/debug_image:sensor_msgs/msg/Image"
    "/visual_odometry/telemetry:std_msgs/msg/String"
    "/visual_odometry/debug_image:sensor_msgs/msg/Image"
    "/state_estimation/odom:nav_msgs/msg/Odometry"
    "/slam/pose:geometry_msgs/msg/PoseStamped"
    "/slam/trajectory:nav_msgs/msg/Path"
    "/slam/map:nav_msgs/msg/OccupancyGrid"
    "/slam/landmarks:visualization_msgs/msg/MarkerArray"
    "/slam/diagnostics:diagnostic_msgs/msg/DiagnosticArray"
    "/naviguard/decision:std_msgs/msg/String"
    "/naviguard/decision_marker:visualization_msgs/msg/Marker"
    "/recovery/state:std_msgs/msg/String"
    "/recovery/visualization:visualization_msgs/msg/MarkerArray"
    "/goal_pose:geometry_msgs/msg/PoseStamped"
    "/navigation/state:std_msgs/msg/String"
    "/navigation/path:nav_msgs/msg/Path"
    "/navigation/waypoints:geometry_msgs/msg/PoseArray"
    "/navigation/markers:visualization_msgs/msg/MarkerArray"
    "/navigation/diagnostics:diagnostic_msgs/msg/DiagnosticArray"
)

ACTIVE_TOPICS=$(ros2 topic list 2>/dev/null || true)

for entry in "${CORE_TOPICS[@]}"; do
    topic="${entry%%:*}"
    type="${entry##*:}"
    if echo "$ACTIVE_TOPICS" | grep -q "^${topic}$"; then
        status_flag="[ACTIVE]"
    else
        status_flag="[INACTIVE]"
    fi
    printf "%-32s | %-32s %s\n" "$topic" "$type" "$status_flag"
done

echo "================================================================="
