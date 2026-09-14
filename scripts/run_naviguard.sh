#!/usr/bin/env bash
# =============================================================================
# NAVIGUARD Master Demonstration & Operator Dashboard Launcher
# SIH 2026 Problem Statement 26126
#
# Starts with ONE command:
#   1. Gazebo Harmonic with RELLIS-Derived Outdoor World
#   2. 4-Wheel Differential-Drive UGV Spawning
#   3. Robot State Publisher & TF Trees
#   4. ros_gz_bridge (Camera, IMU, Odometry, CmdVel)
#   5. Phase 2 Perception (Traversability & Edge Analysis)
#   6. Phase 3/4 Visual Odometry (Lucas-Kanade & Essential Matrix RANSAC)
#   7. Phase 5A Sensor Synchronization & Diagnostics
#   8. Phase 5B Multi-Rate Metric State Estimation EKF
#   9. Phase 6 Visual SLAM & Metric Occupancy Grid
#  10. Phase 7 Multi-Dimensional Confidence Engine
#  11. Phase 8 Autonomous Recovery State Machine
#  12. Phase 9 Mission Management, A* Planner & Path Follower
#  13. Phase 10 RELLIS Simulation Environment
#  14. Operator Web Dashboard on http://localhost:8080
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$WS_DIR/.run"
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/naviguard.pid"
LOG_FILE="$RUN_DIR/naviguard.log"

# Default configuration
HEADLESS="true"
ENABLE_RVIZ="false"
OPEN_BROWSER="true"
WORLD_FILE="$WS_DIR/install/naviguard_description/share/naviguard_description/worlds/rellis_outdoor_world.sdf"

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gui) HEADLESS="false" ;;
        --headless) HEADLESS="true" ;;
        --with-rviz|--rviz) ENABLE_RVIZ="true" ;;
        --no-browser) OPEN_BROWSER="false" ;;
        --world) WORLD_FILE="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --gui          Launch Gazebo with 3D GUI window (default: headless for WSLg efficiency)"
            echo "  --headless     Run Gazebo simulation headless (server-only)"
            echo "  --with-rviz    Launch RViz2 alongside operator dashboard"
            echo "  --no-browser   Do not automatically launch web browser"
            echo "  --world <sdf>  Custom SDF world file"
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

if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
else
    echo "ERROR: $WS_DIR/install/setup.bash not found! Run colcon build first."
    exit 1
fi

# Clean up any stale processes before starting
"$SCRIPT_DIR/stop_naviguard.sh" >/dev/null 2>&1 || true
sleep 1

echo "================================================================="
echo " NAVIGUARD — Closed-Loop Autonomous Vision Navigation Stack"
echo " SIH 2026 Problem Statement 26126"
echo "================================================================="
echo " Environment:      RELLIS-Derived Texas Outdoor Trail (Gazebo Harmonic)"
echo " Simulation Mode:  $( [ "$HEADLESS" == "true" ] && echo 'Headless Server' || echo 'Interactive Gazebo GUI' )"
echo " RViz2:            $ENABLE_RVIZ"
echo " Operator UI:      http://localhost:8080"
echo " Log output:       $LOG_FILE"
echo "================================================================="
echo ">>> Launching NAVIGUARD master demonstration stack..."

# Trap signals for graceful shutdown
cleanup() {
    echo ""
    echo ">>> Caught termination signal! Initiating clean shutdown..."
    "$SCRIPT_DIR/stop_naviguard.sh"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Launch master demo stack in background
ros2 launch naviguard_description naviguard_demo.launch.py \
    headless:="$HEADLESS" \
    world:="$WORLD_FILE" \
    start_demo_motion:=false \
    start_navigation:=true \
    start_dashboard:=true \
    enable_rviz:="$ENABLE_RVIZ" > "$LOG_FILE" 2>&1 &

LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$PID_FILE"

echo ">>> Stack launched (PID: $LAUNCH_PID). Running readiness verification..."

# Helper check function
wait_for_condition() {
    local label="$1"
    local cmd="$2"
    local max_wait=30
    local elapsed=0
    printf "  %-32s ... " "$label"
    while [ $elapsed -lt $max_wait ]; do
        if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
            echo -e "\033[31mFAILED\033[0m (Process died prematurely!)"
            echo "Check log file: $LOG_FILE"
            tail -n 25 "$LOG_FILE"
            exit 1
        fi
        if eval "$cmd" >/dev/null 2>&1; then
            echo -e "\033[32mREADY\033[0m"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo -e "\033[33mTIMEOUT (Proceeding with caution)\033[0m"
    return 0
}

# 10 Readiness Checks
wait_for_condition "[1/10] Gazebo Simulation" "ros2 topic list | grep -q /clock"
wait_for_condition "[2/10] Camera & IMU Sensors" "ros2 topic list | grep -q /camera/image_raw && ros2 topic list | grep -q /imu"
wait_for_condition "[3/10] Vision Perception" "ros2 topic list | grep -q /perception/debug_image"
wait_for_condition "[4/10] Visual Odometry" "ros2 topic list | grep -q /visual_odometry/telemetry"
wait_for_condition "[5/10] State Estimation EKF" "ros2 topic list | grep -q /state_estimation/odom"
wait_for_condition "[6/10] Visual SLAM & Map" "ros2 topic list | grep -q /slam/pose && ros2 topic list | grep -q /slam/map"
wait_for_condition "[7/10] Confidence Engine" "ros2 topic list | grep -q /naviguard/decision"
wait_for_condition "[8/10] Autonomous Recovery" "ros2 topic list | grep -q /recovery/state"
wait_for_condition "[9/10] Mission Navigation" "ros2 topic list | grep -q /navigation/state"
wait_for_condition "[10/10] Operator Dashboard" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8080 | grep -q 200"

echo ""
echo "================================================================="
echo -e " \033[32m✔ NAVIGUARD AUTONOMOUS STACK FULLY OPERATIONAL\033[0m"
echo "================================================================="
echo " Dashboard Web URL:   http://localhost:8080"
echo ""
echo " OPERATOR WORKFLOW:"
echo "   1. Open http://localhost:8080 in your browser."
echo "   2. Click 'Set Destination' button in the bottom panel."
echo "   3. Click anywhere on the map (e.g. forward on the trail)."
echo "   4. Confirm destination coordinates and safety validation."
echo "   5. Click 'Start Navigation' to dispatch the autonomous goal."
echo "   6. Observe real-time camera, VO vectors, A* path replanning,"
echo "      and confidence state transitions."
echo ""
echo " Press Ctrl+C at any time to stop the entire system cleanly."
echo "================================================================="

# Auto-open browser if enabled
if [ "$OPEN_BROWSER" == "true" ]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:8080 >/dev/null 2>&1 &
    elif command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe /c start http://localhost:8080 >/dev/null 2>&1 &
    fi
fi

# Keep script running and wait for background launch process
wait "$LAUNCH_PID"
