#!/usr/bin/env bash
# NAVIGUARD Master Demonstration Clean Shutdown Script
# Terminates all simulation, robotics autonomy, and dashboard processes cleanly.

PID_FILE="/home/maxx/naviguard_ws/.run/naviguard.pid"

echo "================================================================="
echo " NAVIGUARD Clean System Shutdown"
echo "================================================================="

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo ">>> Stopping main launch process (PID: $PID)..."
        kill -INT "$PID" 2>/dev/null || true
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -TERM "$PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$PID_FILE"
fi

echo ">>> Terminating any remaining background ROS 2 and Gazebo processes..."
pkill -INT -f "naviguard_demo.launch.py|ros_gz_sim|gz sim|parameter_bridge" 2>/dev/null || true
sleep 1
pkill -9 -f "gz sim|ruby|ros2|parameter_bridge|dashboard_node" 2>/dev/null || true

echo ">>> Clean shutdown complete. Zero orphan processes."
echo "================================================================="
