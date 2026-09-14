#!/usr/bin/env bash
# NAVIGUARD Master Demonstration Launcher
# Starts Gazebo Harmonic simulation, all perception & estimation pipelines,
# Visual SLAM, Confidence Engine, Autonomous Recovery, and Demo Motion.

set -e

# Default settings
HEADLESS="false"
ENABLE_RVIZ="false"
START_MOTION="true"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --headless) HEADLESS="true" ;;
        --with-rviz) ENABLE_RVIZ="true" ;;
        --no-motion) START_MOTION="false" ;;
        --rviz-only)
            source /opt/ros/jazzy/setup.bash
            source /home/maxx/naviguard_ws/install/setup.bash
            RVIZ_CFG="/home/maxx/naviguard_ws/install/naviguard_description/share/naviguard_description/config/naviguard_demo.rviz"
            echo "Launching RViz2 with NAVIGUARD demo profile: $RVIZ_CFG"
            exec ros2 run rviz2 rviz2 -d "$RVIZ_CFG"
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --headless     Run Gazebo simulation headless (server-only, no GUI)"
            echo "  --with-rviz    Launch RViz2 with the preconfigured NAVIGUARD demo layout"
            echo "  --no-motion    Disable the demo motion generator (robot stays stationary)"
            echo "  --rviz-only    Launch only the RViz2 visualization interface"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Source ROS 2 environment
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
else
    echo "ERROR: /opt/ros/jazzy/setup.bash not found!"
    exit 1
fi

if [ -f "/home/maxx/naviguard_ws/install/setup.bash" ]; then
    source /home/maxx/naviguard_ws/install/setup.bash
else
    echo "ERROR: /home/maxx/naviguard_ws/install/setup.bash not found! Please build the workspace first."
    exit 1
fi

echo "================================================================="
echo " NAVIGUARD — Closed-Loop Demonstration & Integration Stack"
echo "================================================================="
echo " Mode:           $( [ "$HEADLESS" == "true" ] && echo 'Headless Gazebo' || echo 'Gazebo GUI' )"
echo " RViz2:          $ENABLE_RVIZ"
echo " Demo Motion:    $START_MOTION"
echo " Target Stack:   Simulation + Perception + VO + Sync + StateEst +"
echo "                 Visual SLAM + Confidence Engine + Autonomous Recovery"
echo "================================================================="
echo " Quick commands in other terminals:"
echo "   ./scripts/naviguard_status.sh     - Monitor live node & topic rates"
echo "   ./scripts/naviguard_topics.sh     - View full topic inventory"
echo "   ./scripts/inject_fault.sh         - Trigger controlled recovery"
echo "   ./scripts/view_visual_odometry.sh - Inspect live VO feature tracking"
echo "================================================================="

exec ros2 launch naviguard_description naviguard_demo.launch.py \
    headless:="$HEADLESS" \
    enable_rviz:="$ENABLE_RVIZ" \
    start_demo_motion:="$START_MOTION"
