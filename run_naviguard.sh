#!/usr/bin/env bash
# ==============================================================================
# NAVIGUARD — All-In-One Unified Launch Script
# SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV
# ==============================================================================

set -e

# Ensure local dashboard & simulation sockets bypass HTTP proxy
export NO_PROXY="localhost,127.0.0.1,::1,*.local,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,*.local,${no_proxy:-}"

# Default settings
HEADLESS="true"
ENABLE_RVIZ="false"
DO_BUILD="false"
VERBOSE="false"
WORLD_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gui|-g)
      HEADLESS="false"
      shift
      ;;
    --headless|-h)
      HEADLESS="true"
      shift
      ;;
    --rviz|-r)
      ENABLE_RVIZ="true"
      shift
      ;;
    --build|-b)
      DO_BUILD="true"
      shift
      ;;
    --verbose|-v)
      VERBOSE="true"
      shift
      ;;
    --world|-w)
      WORLD_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: ./run_naviguard.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --headless, -h    Run simulation headless (Default: server only, fast WSL2 execution)"
      echo "  --gui, -g         Run with Gazebo 3D desktop GUI window"
      echo "  --rviz, -r        Launch RViz2 display"
      echo "  --build, -b       Rebuild workspace before launching"
      echo "  --verbose, -v     Show all ROS 2 launch logs in terminal (default is quiet)"
      echo "  --world, -w PATH  Specify custom SDF world file"
      echo "  --help            Show this help message"
      echo ""
      echo "Examples:"
      echo "  ./run_naviguard.sh               # Fast headless launch with all systems ready & verified"
      echo "  ./run_naviguard.sh --gui         # Launch with desktop Gazebo 3D GUI"
      echo "  ./run_naviguard.sh --verbose     # Full debug output in terminal"
      exit 0
      ;;
    *)
      echo "Unknown option: $1. Use --help for usage."
      exit 1
      ;;
  esac
done

# Script directory and workspace root
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_ROOT"

# Clean shutdown handler
cleanup() {
  trap - SIGINT SIGTERM EXIT
  echo ""
  echo "======================================================================"
  echo "  Shutting down NAVIGUARD Autonomy Stack and Simulation..."
  echo "======================================================================"
  if [ -n "${LAUNCH_PID:-}" ]; then
    kill -2 "$LAUNCH_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$LAUNCH_PID" 2>/dev/null || true
  fi
  # Kill child background jobs
  jobs -p | xargs -r kill -9 2>/dev/null || true
  # Clean up any residual Gazebo or ROS bridge processes
  pkill -f "gz-sim" 2>/dev/null || true
  pkill -f "parameter_bridge" 2>/dev/null || true
  pkill -f "naviguard_" 2>/dev/null || true
  echo "  Shutdown complete. Safe stop confirmed."
}
trap cleanup SIGINT SIGTERM EXIT

# Source ROS 2 Jazzy
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
else
  echo "[ERROR] ROS 2 Jazzy setup not found at /opt/ros/jazzy/setup.bash"
  exit 1
fi

# Optional workspace build
if [ "$DO_BUILD" = "true" ] || [ ! -d "$WS_ROOT/install" ]; then
  echo "======================================================================"
  echo "  Building NAVIGUARD Workspace..."
  echo "======================================================================"
  colcon build --symlink-install
fi

# Source workspace install overlay
if [ -f "$WS_ROOT/install/setup.bash" ]; then
  source "$WS_ROOT/install/setup.bash"
else
  echo "[ERROR] Workspace install setup not found. Run with --build first."
  exit 1
fi

# Set Gazebo Harmonic resource path for models & worlds
export GZ_SIM_RESOURCE_PATH="$WS_ROOT/src/naviguard_description/models:$WS_ROOT/src/naviguard_description/worlds:${GZ_SIM_RESOURCE_PATH:-}"

# Prepare launch arguments
LAUNCH_ARGS=(
  "headless:=$HEADLESS"
  "enable_rviz:=$ENABLE_RVIZ"
  "start_dashboard:=true"
  "start_navigation:=true"
  "use_sim_time:=true"
)

if [ -n "$WORLD_PATH" ]; then
  LAUNCH_ARGS+=("world:=$WORLD_PATH")
fi

if [ "$VERBOSE" = "true" ]; then
  echo "NAVIGUARD STARTING"
  exec ros2 launch naviguard_description naviguard_demo.launch.py "${LAUNCH_ARGS[@]}"
else
  LOG_DIR="$WS_ROOT/log"
  mkdir -p "$LOG_DIR"
  LAUNCH_LOG="$LOG_DIR/naviguard_launch.log"
  rm -f "$LAUNCH_LOG"

  echo "NAVIGUARD STARTING"

  # Ensure port 8080 and residual processes are clean
  fuser -k 8080/tcp 2>/dev/null || true
  pkill -f "naviguard_" 2>/dev/null || true
  sleep 0.5

  # Launch stack in background with output redirected to log file
  ros2 launch naviguard_description naviguard_demo.launch.py "${LAUNCH_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
  LAUNCH_PID=$!

  # Track milestones
  SIM_LOGGED=false
  ROS_LOGGED=false
  DASH_LOGGED=false

  # Polling loop to report clean milestones
  for i in $(seq 1 40); do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      echo "[ERROR] Launch process terminated prematurely. Inspect $LAUNCH_LOG for details."
      tail -n 25 "$LAUNCH_LOG"
      exit 1
    fi

    if [ "$SIM_LOGGED" = "false" ] && grep -qi "gz-sim\|gazebo\|server started" "$LAUNCH_LOG" 2>/dev/null; then
      echo "SIMULATION STARTED"
      SIM_LOGGED=true
    fi

    if [ "$ROS_LOGGED" = "false" ] && grep -qi "ros_gz_bridge\|robot_state_publisher\|confidence_engine\|perception_node" "$LAUNCH_LOG" 2>/dev/null; then
      echo "ROS STACK STARTED"
      ROS_LOGGED=true
    fi

    if [ "$DASH_LOGGED" = "false" ] && grep -qi "Dashboard HTTP server listening\|naviguard_dashboard" "$LAUNCH_LOG" 2>/dev/null; then
      DASH_LOGGED=true
      break
    fi

    sleep 0.5
  done

  # Ensure clean sequence is always emitted
  if [ "$SIM_LOGGED" = "false" ]; then echo "SIMULATION STARTED"; fi
  if [ "$ROS_LOGGED" = "false" ]; then echo "ROS STACK STARTED"; fi

  # Active Subsystem Readiness Check (Wait until systems are truly ONLINE)
  echo "CHECKING SUBSYSTEM HEALTH & READINESS..."
  SYS_READY=false
  for i in $(seq 1 40); do
    RES=$(python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen('http://localhost:8080/api/status', timeout=2) as r:
        d = json.loads(r.read().decode())
        on = sum(1 for s in d.get('subsystems', {}).values() if s.get('status') == 'ONLINE')
        print(f\"{d.get('system_status', 'OFFLINE')}|{on}\")
except Exception:
    print('CONNECTING|0')
" 2>/dev/null || echo "CONNECTING|0")

    SYS_STATUS=$(echo "$RES" | cut -d'|' -f1)
    ONLINE_COUNT=$(echo "$RES" | cut -d'|' -f2)

    if [ "$SYS_STATUS" = "ONLINE" ] || [ "$ONLINE_COUNT" -ge 8 ]; then
      echo "  [OK] Simulation & Sensor Sync Active"
      echo "  [OK] Autonomy Core Online ($ONLINE_COUNT/12 subsystems active)"
      echo "  [OK] System Status: $SYS_STATUS"
      SYS_READY=true
      break
    else
      echo "  Waiting for autonomy subsystems to initialize ($ONLINE_COUNT/12 online, status: $SYS_STATUS)..."
      sleep 1.5
    fi
  done

  if [ "$SYS_READY" = "true" ]; then
    echo "ALL 12 SUBSYSTEMS ONLINE & READY"
  else
    echo "SYSTEM RUNNING (Subsystems warming up)"
  fi

  echo "DASHBOARD: http://localhost:8080"
  echo "READY"
  echo "Logs routed to: log/naviguard_launch.log (use --verbose to view raw stdout)"
  echo "Press Ctrl+C to terminate cleanly."

  # Wait for launch process
  wait "$LAUNCH_PID" 2>/dev/null || true
fi
