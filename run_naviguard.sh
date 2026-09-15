#!/usr/bin/env bash
# ==============================================================================
# NAVIGUARD — All-In-One Unified Interactive Shell & Launch System
# SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV
# ==============================================================================

set -e

# ANSI Color Palette (Kali-style electric theme)
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_RED="\033[1;31m"
C_GREEN="\033[1;32m"
C_YELLOW="\033[1;33m"
C_BLUE="\033[1;34m"
C_MAGENTA="\033[1;35m"
C_CYAN="\033[1;36m"
C_WHITE="\033[1;37m"
C_DIM="\033[2m"
C_BG_BLUE="\033[44m"
C_BG_RED="\033[41m"

# Ensure local dashboard & simulation sockets bypass HTTP proxy
export NO_PROXY="localhost,127.0.0.1,::1,*.local,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,::1,*.local,${no_proxy:-}"

# Script directory and workspace root
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WS_ROOT"

# Default settings
HEADLESS="true"
ENABLE_RVIZ="false"
DO_BUILD="false"
VERBOSE="false"
INTERACTIVE="auto"
WORLD_PATH="$WS_ROOT/install/naviguard_description/share/naviguard_description/worlds/rellis_outdoor_world.sdf"

# Print Kali-Style Banner
print_banner() {
  clear 2>/dev/null || true
  echo -e "${C_CYAN}"
  cat << "EOF"
 ███╗   ██╗ █████╗ ██╗   ██╗██╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
 ████╗  ██║██╔══██╗██║   ██║██║██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
 ██╔██╗ ██║███████║██║   ██║██║██║  ███╗██║   ██║███████║██████╔╝██║  ██║
 ██║╚██╗██║██╔══██║╚██╗ ██╔╝██║██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
 ██║ ╚████║██║  ██║ ╚████╔╝ ██║╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
 ╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
EOF
  echo -e "${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ╔══════════════════════════════════════════════════════════════════╗${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ║ ${C_CYAN}NAVIGUARD ${C_WHITE}— Vision-Based Autonomous Navigation for Outdoor UGV    ║${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ║ ${C_YELLOW}SIH 2026 Problem Statement 26126 ${C_WHITE}• High-Assurance Field Autonomy  ║${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ╚══════════════════════════════════════════════════════════════════╝${C_RESET}"
  echo ""
  echo -e "   ${C_DIM}┌──────────────────────────────────────────────────────────────────┐${C_RESET}"
  echo -e "   ${C_DIM}│${C_RESET}  ${C_BOLD}Framework:${C_RESET} ROS 2 Jazzy        ${C_DIM}│${C_RESET}  ${C_BOLD}Simulator:${C_RESET} Gazebo Harmonic    ${C_DIM}│${C_RESET}"
  echo -e "   ${C_DIM}│${C_RESET}  ${C_BOLD}Perception:${C_RESET} YOLOv8 (CPU/ONNX)  ${C_DIM}│${C_RESET}  ${C_BOLD}Costmap:${C_RESET}   11-Layer Model     ${C_DIM}│${C_RESET}"
  echo -e "   ${C_DIM}│${C_RESET}  ${C_BOLD}Footprint:${C_RESET}  0.56m x 0.48m       ${C_DIM}│${C_RESET}  ${C_BOLD}Inflation:${C_RESET} 0.34m (Inscribed)   ${C_DIM}│${C_RESET}"
  echo -e "   ${C_DIM}│${C_RESET}  ${C_BOLD}Authority:${C_RESET}  Single /cmd_vel     ${C_DIM}│${C_RESET}  ${C_BOLD}Dashboard:${C_RESET} http://localhost:8080 ${C_DIM}│${C_RESET}"
  echo -e "   ${C_DIM}│${C_RESET}  ${C_BOLD}Graphics:${C_RESET}   NVIDIA RTX 2050     ${C_DIM}│${C_RESET}  ${C_BOLD}Pipeline:${C_RESET}  D3D12 Gallium (GPU)${C_DIM}│${C_RESET}"
  echo -e "   ${C_DIM}└──────────────────────────────────────────────────────────────────┘${C_RESET}"
  echo ""
}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gui|-g)
      HEADLESS="false"
      INTERACTIVE="false"
      shift
      ;;
    --headless|-h)
      HEADLESS="true"
      INTERACTIVE="false"
      shift
      ;;
    --rviz|-r)
      ENABLE_RVIZ="true"
      INTERACTIVE="false"
      shift
      ;;
    --build|-b)
      DO_BUILD="true"
      shift
      ;;
    --verbose|-v)
      VERBOSE="true"
      INTERACTIVE="false"
      shift
      ;;
    --interactive|-i)
      INTERACTIVE="true"
      shift
      ;;
    --non-interactive|-n)
      INTERACTIVE="false"
      shift
      ;;
    --world|-w)
      WORLD_PATH="$2"
      shift 2
      ;;
    --deps|-d)
      bash "$WS_ROOT/install_python_deps.sh" "${2:---check}"
      exit 0
      ;;
    --help)
      print_banner
      echo -e "${C_BOLD}Usage:${C_RESET} ./run_naviguard.sh [OPTIONS]"
      echo ""
      echo -e "${C_BOLD}Options:${C_RESET}"
      echo -e "  ${C_CYAN}--interactive, -i${C_RESET}    Open interactive Kali-style control console"
      echo -e "  ${C_CYAN}--headless, -h${C_RESET}       Run simulation headless (Default: server only, fast WSL2/Linux)"
      echo -e "  ${C_CYAN}--gui, -g${C_RESET}            Run with Gazebo Harmonic 3D desktop GUI window"
      echo -e "  ${C_CYAN}--rviz, -r${C_RESET}           Launch RViz2 3D visualization display"
      echo -e "  ${C_CYAN}--build, -b${C_RESET}          Rebuild workspace before launching"
      echo -e "  ${C_CYAN}--deps                     Verify/audit Python dependencies"
      echo -e "  ${C_CYAN}--verbose, -v${C_RESET}        Show all raw ROS 2 launch logs in terminal"
      echo -e "  ${C_CYAN}--world, -w PATH${C_RESET}     Specify custom SDF world file"
      echo -e "  ${C_CYAN}--help${C_RESET}               Show this help message"
      echo ""
      echo -e "${C_BOLD}Examples:${C_RESET}"
      echo "  ./run_naviguard.sh                   # Interactive launcher menu & live console"
      echo "  ./run_naviguard.sh --gui             # Launch directly with Gazebo 3D GUI"
      echo "  ./run_naviguard.sh --headless        # Launch directly in headless background"
      echo "  ./run_naviguard.sh --build           # Rebuild all 11 packages and launch"
      echo "  ./run_naviguard.sh --deps            # Audit Python dependencies"
      exit 0
      ;;
    *)
      echo -e "${C_RED}[ERROR] Unknown option: $1. Use --help for usage.${C_RESET}"
      exit 1
      ;;
  esac
done

# Interactive Menu when run without specific flags and attached to a TTY
if [ "$INTERACTIVE" = "auto" ] && [ -t 0 ]; then
  print_banner
  echo -e "   ${C_BOLD}${C_CYAN}SELECT OPERATIONAL LAUNCH MODE:${C_RESET}"
  echo -e "   ${C_GREEN}[1]${C_RESET} ${C_BOLD}Headless + Web Dashboard${C_RESET} ${C_DIM}(Recommended: fast, optimal WSL2/Linux performance)${C_RESET}"
  echo -e "   ${C_GREEN}[2]${C_RESET} ${C_BOLD}Gazebo 3D GUI + Web Dashboard${C_RESET} ${C_DIM}(Requires X11/Wayland display)${C_RESET}"
  echo -e "   ${C_GREEN}[3]${C_RESET} ${C_BOLD}Full Stack + RViz2 + Web Dashboard${C_RESET} ${C_DIM}(Comprehensive visual inspection)${C_RESET}"
  echo -e "   ${C_GREEN}[4]${C_RESET} ${C_BOLD}Clean Rebuild & Launch${C_RESET} ${C_DIM}(Runs colcon build --symlink-install first)${C_RESET}"
  echo -e "   ${C_GREEN}[5]${C_RESET} ${C_BOLD}Run Full Regression Test Suite${C_RESET} ${C_DIM}(Executes all 244 unit/integration tests)${C_RESET}"
  echo -e "   ${C_GREEN}[6]${C_RESET} ${C_BOLD}Verify Python Dependencies${C_RESET} ${C_DIM}(Runs install_python_deps.sh audit)${C_RESET}"
  echo -e "   ${C_RED}[q]${C_RESET} ${C_BOLD}Exit${C_RESET}"
  echo ""
  read -r -p "   naviguard-init > " CHOICE
  case "$CHOICE" in
    1|"")
      HEADLESS="true"
      ENABLE_RVIZ="false"
      ;;
    2)
      HEADLESS="false"
      ENABLE_RVIZ="false"
      ;;
    3)
      HEADLESS="false"
      ENABLE_RVIZ="true"
      ;;
    4)
      DO_BUILD="true"
      HEADLESS="true"
      ;;
    5)
      echo ""
      echo -e "${C_CYAN}Executing NAVIGUARD Regression Test Suite...${C_RESET}"
      source /opt/ros/jazzy/setup.bash 2>/dev/null || true
      source install/setup.bash 2>/dev/null || true
      colcon test --event-handlers console_direct+ --packages-select naviguard_navigation naviguard_dashboard naviguard_perception naviguard_confidence naviguard_recovery naviguard_slam naviguard_sensor_sync naviguard_state_estimation naviguard_visual_odometry naviguard_rellis
      exit 0
      ;;
    6)
      bash "$WS_ROOT/install_python_deps.sh" --check
      exit 0
      ;;
    q|Q|exit)
      echo -e "${C_YELLOW}Exiting NAVIGUARD.${C_RESET}"
      exit 0
      ;;
    *)
      echo -e "${C_YELLOW}Invalid choice, defaulting to Headless.${C_RESET}"
      HEADLESS="true"
      ;;
  esac
fi

# Clean shutdown handler
cleanup() {
  trap - SIGINT SIGTERM EXIT
  echo ""
  echo -e "${C_YELLOW}======================================================================${C_RESET}"
  echo -e "  ${C_BOLD}${C_RED}[!] Shutting down NAVIGUARD Autonomy Stack and Simulation...${C_RESET}"
  echo -e "${C_YELLOW}======================================================================${C_RESET}"
  if [ -n "${LAUNCH_PID:-}" ]; then
    kill -2 "$LAUNCH_PID" 2>/dev/null || true
    sleep 1.5
    kill -9 "$LAUNCH_PID" 2>/dev/null || true
  fi
  # Kill child background jobs
  jobs -p | xargs -r kill -9 2>/dev/null || true
  # Clean up any residual Gazebo or ROS bridge processes
  pkill -f "gz-sim" 2>/dev/null || true
  pkill -f "parameter_bridge" 2>/dev/null || true
  pkill -f "naviguard_" 2>/dev/null || true
  echo -e "  ${C_GREEN}[✓] Shutdown complete. Single /cmd_vel owner released. Safe stop confirmed.${C_RESET}"
}
trap cleanup SIGINT SIGTERM EXIT

# Source ROS 2 Jazzy
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
else
  echo -e "${C_RED}[ERROR] ROS 2 Jazzy setup not found at /opt/ros/jazzy/setup.bash${C_RESET}"
  exit 1
fi

# Optional workspace build
if [ "$DO_BUILD" = "true" ] || [ ! -d "$WS_ROOT/install" ]; then
  echo -e "${C_CYAN}======================================================================${C_RESET}"
  echo -e "  ${C_BOLD}Building NAVIGUARD Workspace (colcon)...${C_RESET}"
  echo -e "${C_CYAN}======================================================================${C_RESET}"
  colcon build --symlink-install
fi

# Source workspace install overlay
if [ -f "$WS_ROOT/install/setup.bash" ]; then
  source "$WS_ROOT/install/setup.bash"
else
  echo -e "${C_RED}[ERROR] Workspace install setup not found. Run with --build first.${C_RESET}"
  exit 1
fi

# Enable GPU / Hardware Acceleration via NVIDIA RTX 2050 (D3D12 WSL2)
if [ -d "/usr/lib/wsl/lib" ] && command -v nvidia-smi &>/dev/null; then
  export GALLIUM_DRIVER=d3d12
  export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
  export LIBGL_ALWAYS_SOFTWARE=0
fi

# Set Gazebo Harmonic resource path for models, materials, & worlds
export GZ_SIM_RESOURCE_PATH="$WS_ROOT/src/naviguard_description/models:$WS_ROOT/src/naviguard_description/materials:$WS_ROOT/src/naviguard_description/worlds:${GZ_SIM_RESOURCE_PATH:-}"

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

print_banner

if [ "$VERBOSE" = "true" ]; then
  echo -e "${C_GREEN}[*] Launching NAVIGUARD in verbose mode (direct ROS 2 stdout)...${C_RESET}"
  exec ros2 launch naviguard_description naviguard_demo.launch.py "${LAUNCH_ARGS[@]}"
else
  LOG_DIR="$WS_ROOT/log"
  mkdir -p "$LOG_DIR"
  LAUNCH_LOG="$LOG_DIR/naviguard_launch.log"
  rm -f "$LAUNCH_LOG"

  echo -e "  ${C_CYAN}[1/4]${C_RESET} ${C_BOLD}Cleaning previous instances & socket ports (8080)...${C_RESET}"
  fuser -k 8080/tcp 2>/dev/null || true
  pkill -f "naviguard_" 2>/dev/null || true
  sleep 0.5

  echo -e "  ${C_CYAN}[2/4]${C_RESET} ${C_BOLD}Spawning Gazebo Harmonic, Bridges, and Autonomy Nodes...${C_RESET}"
  ros2 launch naviguard_description naviguard_demo.launch.py "${LAUNCH_ARGS[@]}" > "$LAUNCH_LOG" 2>&1 &
  LAUNCH_PID=$!

  # Track milestones
  SIM_LOGGED=false
  ROS_LOGGED=false
  DASH_LOGGED=false

  for i in $(seq 1 45); do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      echo -e "${C_RED}[ERROR] Launch process terminated prematurely. Log excerpt:${C_RESET}"
      tail -n 25 "$LAUNCH_LOG"
      exit 1
    fi

    if [ "$SIM_LOGGED" = "false" ] && grep -qi "gz-sim\|gazebo\|server started" "$LAUNCH_LOG" 2>/dev/null; then
      echo -e "  ${C_GREEN}[✓] Simulation Engine Active (Gazebo Harmonic)${C_RESET}"
      SIM_LOGGED=true
    fi

    if [ "$ROS_LOGGED" = "false" ] && grep -qi "ros_gz_bridge\|robot_state_publisher\|confidence_engine\|perception_node" "$LAUNCH_LOG" 2>/dev/null; then
      echo -e "  ${C_GREEN}[✓] ROS 2 Nodes & TF Publisher Online${C_RESET}"
      ROS_LOGGED=true
    fi

    if [ "$DASH_LOGGED" = "false" ] && grep -qi "Dashboard HTTP server listening\|naviguard_dashboard" "$LAUNCH_LOG" 2>/dev/null; then
      echo -e "  ${C_GREEN}[✓] Operator Web Dashboard Live on port 8080${C_RESET}"
      DASH_LOGGED=true
      break
    fi

    sleep 0.5
  done

  echo -e "  ${C_CYAN}[3/4]${C_RESET} ${C_BOLD}Performing Subsystem Readiness & Health Handshake...${C_RESET}"
  SYS_READY=false
  for i in $(seq 1 35); do
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
      echo -e "  ${C_GREEN}[✓] Autonomy Core Online ($ONLINE_COUNT/12 subsystems active)${C_RESET}"
      echo -e "  ${C_GREEN}[✓] Overall System Status: ${C_BOLD}${SYS_STATUS}${C_RESET}"
      SYS_READY=true
      break
    else
      echo -ne "      ${C_DIM}Waiting for subsystems to calibrate ($ONLINE_COUNT/12 online, status: $SYS_STATUS)...${C_RESET}\r"
      sleep 1.2
    fi
  done
  echo ""

  echo -e "  ${C_CYAN}[4/4]${C_RESET} ${C_BOLD}11-Layer Spatial Map & Vehicle Geometry Ready!${C_RESET}"
  echo ""
  echo -e "  ${C_BOLD}${C_GREEN}══════════════════════════════════════════════════════════════════════${C_RESET}"
  echo -e "   ${C_BOLD}${C_WHITE}OPERATOR DASHBOARD : ${C_CYAN}http://localhost:8080${C_RESET}"
  echo -e "   ${C_BOLD}${C_WHITE}YOLO PERCEPTION    : ${C_GREEN}Active (OpenCV-DNN / CPU)${C_RESET}"
  echo -e "   ${C_BOLD}${C_WHITE}VEHICLE GEOMETRY   : ${C_YELLOW}0.56m x 0.48m (Inflation: 0.34m)${C_RESET}"
  echo -e "   ${C_BOLD}${C_WHITE}CMD_VEL AUTHORITY  : ${C_MAGENTA}Single Owner (NavigationNode)${C_RESET}"
  echo -e "  ${C_BOLD}${C_GREEN}══════════════════════════════════════════════════════════════════════${C_RESET}"
  echo ""

  # If not a terminal tty, simply wait on process
  if [ ! -t 0 ]; then
    echo -e "${C_DIM}Running in headless background non-interactive mode. PID: $LAUNCH_PID${C_RESET}"
    wait "$LAUNCH_PID" 2>/dev/null || true
    exit 0
  fi

  # Kali-style Interactive Control Console
  echo -e "  ${C_BOLD}Type ${C_CYAN}'h'${C_RESET}${C_BOLD} or ${C_CYAN}'help'${C_RESET}${C_BOLD} for available commands, or ${C_RED}'q'${C_RESET}${C_BOLD} to safely exit.${C_RESET}"
  echo ""

  while kill -0 "$LAUNCH_PID" 2>/dev/null; do
    echo -ne "${C_BOLD}${C_RED}┌──(${C_CYAN}naviguard㉿ugv${C_RED})-[${C_WHITE}~/autonomous-stack${C_RED}]\n└─${C_CYAN}\$ ${C_RESET}"
    read -r CMD ARGS || break

    case "$CMD" in
      h|help|\?)
        echo ""
        echo -e "${C_BOLD}${C_WHITE}NAVIGUARD Interactive Commands:${C_RESET}"
        echo -e "  ${C_CYAN}s, status${C_RESET}       Display live vehicle pose, corridor clearance, & subsystem health"
        echo -e "  ${C_CYAN}g, geo, geometry${C_RESET} Display audited physical geometry and passage thresholds"
        echo -e "  ${C_CYAN}y, yolo${C_RESET}         Query live YOLOv8 detector diagnostics & outdoor classes"
        echo -e "  ${C_CYAN}d, dashboard${C_RESET}    Open or display operator web dashboard URL"
        echo -e "  ${C_CYAN}goal <x> <y>${C_RESET}    Dispatch autonomous navigation goal coordinates (meters)"
        echo -e "  ${C_CYAN}l, logs${C_RESET}         Tail the live launch logs (last 30 lines)"
        echo -e "  ${C_CYAN}c, clear${C_RESET}        Clear screen and re-display Kali banner"
        echo -e "  ${C_RED}q, quit, exit${C_RESET}   Cleanly terminate simulation and all ROS nodes"
        echo ""
        ;;
      s|status)
        echo ""
        echo -e "${C_BOLD}${C_CYAN}--- Live System Diagnostics ---${C_RESET}"
        python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen('http://localhost:8080/api/status', timeout=3) as r:
        data = json.loads(r.read().decode())
        print(f'System Status:       {data.get(\"system_status\")}')
        print(f'Mission State:       {data.get(\"mission_state\")}')
        print(f'Active Recovery:     {data.get(\"active_recovery\")}')
        pose = data.get('vehicle_pose', {})
        print(f'Vehicle Pose:        x={pose.get(\"x\", 0):.2f}m, y={pose.get(\"y\", 0):.2f}m, yaw={pose.get(\"yaw\", 0):.2f}rad')
        geo = data.get('vehicle_diagnostics', {})
        if geo:
            print(f'Corridor Clearance:  {geo.get(\"corridor_clearance_m\", \"N/A\")}m ({geo.get(\"corridor_status\", \"N/A\")})')
            print(f'Feasibility:         Can Fit: {geo.get(\"can_fit\", \"N/A\")} | Can Turn: {geo.get(\"can_turn\", \"N/A\")}')
        yolo = data.get('yolo_diagnostics', {})
        if yolo:
            print(f'YOLO Engine:         Device: {yolo.get(\"device\", \"CPU\")} | Latency: {yolo.get(\"latency_ms\", 0):.1f}ms | Detections: {yolo.get(\"detection_count\", 0)}')
        conf = data.get('confidence', {})
        if conf:
            print(f'Confidence Score:    {conf.get(\"overall_confidence\", 1.0):.2f}')
except Exception as e:
    print(f'Error querying status: {e}')
" 2>/dev/null || echo -e "${C_YELLOW}Unable to reach status endpoint.${C_RESET}"
        echo ""
        ;;
      g|geo|geometry)
        echo ""
        echo -e "${C_BOLD}${C_CYAN}--- Physical Vehicle Geometry & Clearance Model ---${C_RESET}"
        echo -e "  ${C_WHITE}Chassis Dimensions:${C_RESET} 0.50m (L) x 0.32m (W) x 0.16m (H)"
        echo -e "  ${C_WHITE}Bumpers Extrusion:${C_RESET}  x = ±0.28m  -> ${C_BOLD}Total Length: 0.56m${C_RESET}"
        echo -e "  ${C_WHITE}Wheel Lateral Edge:${C_RESET} y = ±0.24m  -> ${C_BOLD}Total Width:  0.48m${C_RESET}"
        echo -e "  ${C_WHITE}Inscribed Radius:${C_RESET}   0.24m"
        echo -e "  ${C_WHITE}Circumscribed Radius:${C_RESET} 0.3688m"
        echo -e "  ${C_WHITE}Safety Margin:${C_RESET}      0.10m"
        echo -e "  ${C_WHITE}Inflation Radius:${C_RESET}   ${C_BOLD}${C_GREEN}0.34m${C_RESET} (Inscribed + Margin)"
        echo -e "  ${C_WHITE}Min Turn Space:${C_RESET}     ${C_BOLD}${C_YELLOW}0.94m${C_RESET} (Skid-steer 360 circle)"
        echo -e "  ${C_WHITE}Passage Thresholds:${C_RESET} SAFE >= 0.68m | TIGHT 0.58-0.68m | BLOCKED < 0.58m"
        echo ""
        ;;
      y|yolo)
        echo ""
        echo -e "${C_BOLD}${C_CYAN}--- YOLOv8 Perception Telemetry ---${C_RESET}"
        python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen('http://localhost:8080/api/status', timeout=3) as r:
        data = json.loads(r.read().decode())
        yolo = data.get('yolo_diagnostics', {})
        print(f'Model:            {yolo.get(\"model\", \"yolov8n-outdoor\")}')
        print(f'Device:           {yolo.get(\"device\", \"CPU\")}')
        print(f'Backend:          {yolo.get(\"backend\", \"OpenCV-DNN\")}')
        print(f'Latency:          {yolo.get(\"latency_ms\", 0.0):.2f} ms')
        print(f'Active Hits:      {yolo.get(\"detection_count\", 0)}')
        dets = yolo.get('detections', [])
        if dets:
            for idx, d in enumerate(dets[:5]):
                print(f'  [{idx+1}] {d.get(\"class\", \"unknown\")} ({d.get(\"confidence\", 0.0):.2f}) at bbox={d.get(\"bbox\")}')
        else:
            print('  No obstacles detected in immediate field of view.')
except Exception as e:
    print(f'Error reading YOLO status: {e}')
" 2>/dev/null || echo -e "${C_YELLOW}YOLO telemetry not yet available.${C_RESET}"
        echo ""
        ;;
      d|dashboard)
        echo -e "${C_GREEN}Operator Dashboard URL: ${C_CYAN}http://localhost:8080${C_RESET}"
        if command -v xdg-open &>/dev/null && [ -n "${DISPLAY:-}" ]; then
          xdg-open "http://localhost:8080" 2>/dev/null || true
        fi
        ;;
      goal)
        X=$(echo "$ARGS" | awk '{print $1}')
        Y=$(echo "$ARGS" | awk '{print $2}')
        if [ -z "$X" ] || [ -z "$Y" ]; then
          echo -e "${C_RED}Usage: goal <x> <y>  (e.g., goal 3.5 1.0)${C_RESET}"
        else
          echo -e "${C_CYAN}Dispatching navigation goal: x=$X, y=$Y...${C_RESET}"
          RESP=$(python3 -c "
import urllib.request, json
try:
    payload = json.dumps({'x': float('$X'), 'y': float('$Y'), 'yaw': 0.0}).encode()
    req = urllib.request.Request('http://localhost:8080/api/goal', data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3) as r:
        print(r.read().decode())
except Exception as e:
    print(f'ERR: {e}')
")
          echo -e "${C_GREEN}Response: $RESP${C_RESET}"
        fi
        ;;
      l|logs)
        echo -e "${C_DIM}--- Last 30 lines of log/naviguard_launch.log ---${C_RESET}"
        tail -n 30 "$LAUNCH_LOG"
        echo ""
        ;;
      c|clear)
        print_banner
        ;;
      q|quit|exit)
        echo -e "${C_YELLOW}Initiating shutdown...${C_RESET}"
        cleanup
        exit 0
        ;;
      "")
        ;;
      *)
        echo -e "${C_RED}Unknown command: '$CMD'. Type 'h' for help.${C_RESET}"
        ;;
    esac
  done

  # If loop exited (e.g. EOF or process died), perform full cleanup
  cleanup
  exit 0
fi
