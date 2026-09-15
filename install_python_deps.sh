#!/usr/bin/env bash
# ==============================================================================
# NAVIGUARD — Automated Python Dependency Installer
# SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV
# ==============================================================================

set -euo pipefail

# ANSI Color Palette
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_RED="\033[1;31m"
C_GREEN="\033[1;32m"
C_YELLOW="\033[1;33m"
C_BLUE="\033[1;34m"
C_CYAN="\033[1;36m"
C_WHITE="\033[1;37m"
C_DIM="\033[2m"

# Workspace root
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$WS_ROOT/requirements.txt"

# Default flags
MODE="auto"            # "auto", "venv", "system"
VENV_DIR="$WS_ROOT/.venv"
UPGRADE="false"
CHECK_ONLY="false"
AUTO_YES="false"
EXTRA_PIP_FLAGS=()

# Banner
print_banner() {
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
  echo -e "${C_BOLD}${C_WHITE}   ║ ${C_CYAN}NAVIGUARD ${C_WHITE}— Python Dependency Installation & Verification       ║${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ║ ${C_YELLOW}SIH 2026 Problem Statement 26126 ${C_WHITE}• Outdoor UGV Autonomy Stack    ║${C_RESET}"
  echo -e "${C_BOLD}${C_WHITE}   ╚══════════════════════════════════════════════════════════════════╝${C_RESET}"
  echo ""
}

show_help() {
  print_banner
  echo -e "${C_BOLD}Usage:${C_RESET} $0 [OPTIONS]"
  echo ""
  echo -e "${C_BOLD}Options:${C_RESET}"
  echo -e "  ${C_CYAN}-c, --check${C_RESET}              Verify installed packages vs requirements without installing"
  echo -e "  ${C_CYAN}-v, --venv [DIR]${C_RESET}         Install into a Python virtual environment (Default: .venv)"
  echo -e "                           Configured with ${C_YELLOW}--system-site-packages${C_RESET} to preserve ROS 2"
  echo -e "  ${C_CYAN}-s, --system${C_RESET}             Install packages into the system/user Python environment"
  echo -e "                           (Applies --break-system-packages on Debian/Ubuntu 24.04)"
  echo -e "  ${C_CYAN}-u, --upgrade${C_RESET}            Upgrade packages to the latest matching versions"
  echo -e "  ${C_CYAN}-y, --yes${C_RESET}                Non-interactive mode (automatically answer yes)"
  echo -e "  ${C_CYAN}-h, --help${C_RESET}               Show this help message"
  echo ""
  echo -e "${C_BOLD}Examples:${C_RESET}"
  echo "  $0 --check              # Audit environment and display dependency status"
  echo "  $0                      # Recommended auto-setup (creates .venv with ROS 2 links)"
  echo "  $0 --system             # Install directly into current system/user site-packages"
  echo "  $0 --venv my_env        # Create or use custom virtual environment"
  echo ""
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--check)
      CHECK_ONLY="true"
      shift
      ;;
    -v|--venv)
      MODE="venv"
      if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
        VENV_DIR="$2"
        shift 2
      else
        shift
      fi
      ;;
    -s|--system|--break-system-packages)
      MODE="system"
      shift
      ;;
    -u|--upgrade)
      UPGRADE="true"
      shift
      ;;
    -y|--yes)
      AUTO_YES="true"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo -e "${C_RED}[ERROR] Unknown option: $1. Use --help for usage.${C_RESET}"
      exit 1
      ;;
  esac
done

print_banner

# Step 1: Validate Python 3 existence
if ! command -v python3 &>/dev/null; then
  echo -e "${C_RED}[✗] python3 could not be found. Please install Python 3 (3.10+) first.${C_RESET}"
  exit 1
fi

PYTHON_BIN="python3"
PY_VERSION="$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")"
echo -e "  ${C_CYAN}[•] Host Python Version:${C_RESET} ${C_BOLD}$PY_VERSION${C_RESET}"

# Step 2: Validate requirements.txt exists
if [ ! -f "$REQ_FILE" ]; then
  echo -e "${C_RED}[✗] Requirements file not found: $REQ_FILE${C_RESET}"
  exit 1
fi
echo -e "  ${C_CYAN}[•] Requirements File:${C_RESET}   ${C_BOLD}$REQ_FILE${C_RESET}"

# Step 3: Check ROS 2 environment context
if [ -n "${ROS_DISTRO:-}" ]; then
  echo -e "  ${C_GREEN}[✓] Active ROS 2 Distro:${C_RESET} ${C_BOLD}$ROS_DISTRO${C_RESET}"
elif [ -f "/opt/ros/jazzy/setup.bash" ]; then
  echo -e "  ${C_GREEN}[✓] Detected ROS 2 Jazzy installation in /opt/ros/jazzy${C_RESET}"
else
  echo -e "  ${C_YELLOW}[!] No active ROS 2 environment sourced (standalone Python mode)${C_RESET}"
fi

# Step 4: Verification function (Audits packages)
verify_packages() {
  local target_python="$1"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" "$target_python" - << 'PY_EOF' 2>/dev/null
import sys
import warnings
warnings.filterwarnings("ignore")

modules = [
    ("numpy", "numpy", ">=1.24.0,<2.0.0"),
    ("scipy", "scipy", ">=1.10.0"),
    ("cv2 (opencv-python)", "cv2", ">=4.6.0"),
    ("Pillow", "PIL", ">=9.5.0"),
    ("PyYAML", "yaml", ">=6.0.1"),
    ("requests", "requests", ">=2.28.0"),
    ("psutil", "psutil", ">=5.9.0"),
    ("rich", "rich", ">=13.0.0"),
    ("matplotlib", "matplotlib", ">=3.7.0"),
    ("pytest", "pytest", ">=7.4.0"),
    ("pytest-cov", "pytest_cov", ">=4.1.0"),
    ("flake8", "flake8", ">=6.0.0"),
    ("setuptools", "setuptools", ">=65.0.0"),
    ("wheel", "wheel", ">=0.40.0"),
    ("onnx", "onnx", ">=1.14.0"),
    ("onnxruntime", "onnxruntime", ">=1.15.0"),
    ("ultralytics", "ultralytics", ">=8.0.0"),
]

ros_modules = [
    ("rclpy", "rclpy"),
    ("sensor_msgs", "sensor_msgs"),
    ("geometry_msgs", "geometry_msgs"),
    ("nav_msgs", "nav_msgs"),
    ("cv_bridge", "cv_bridge"),
]

for name, mod_name, req in modules:
    try:
        m = __import__(mod_name)
        ver = getattr(m, "__version__", "installed")
        print(f"DEP|{name}|{req}|OK|{ver}")
    except ImportError:
        print(f"DEP|{name}|{req}|MISSING|--")

for name, mod_name in ros_modules:
    try:
        m = __import__(mod_name)
        print(f"ROS|{name}|ROS 2 Jazzy|OK|available")
    except ImportError:
        print(f"ROS|{name}|ROS 2 Jazzy|WARN|not in sys.path")
PY_EOF
}

render_audit() {
  local target_python="$1"
  local output
  output=$(verify_packages "$target_python")

  echo ""
  echo -e "${C_BOLD}${C_WHITE}--- Python Dependency Audit & Verification ---${C_RESET}"
  printf "${C_DIM}%-25s %-15s %-12s %-20s${C_RESET}\n" "PACKAGE" "REQUIRED" "STATUS" "INSTALLED VERSION"
  echo -e "${C_DIM}----------------------------------------------------------------------${C_RESET}"

  echo "$output" | grep "^DEP|" | while IFS='|' read -r _ name req status ver; do
    if [ "$status" = "OK" ]; then
      printf "%-25s %-15s ${C_GREEN}%-12s${C_RESET} ${C_BOLD}%-20s${C_RESET}\n" "$name" "$req" "[✓] INSTALLED" "$ver"
    else
      printf "%-25s %-15s ${C_RED}%-12s${C_RESET} ${C_DIM}%-20s${C_RESET}\n" "$name" "$req" "[✗] MISSING" "--"
    fi
  done

  echo ""
  echo -e "${C_BOLD}${C_WHITE}--- ROS 2 Core Modules (Preserved via system-site-packages) ---${C_RESET}"
  printf "${C_DIM}%-25s %-15s %-12s %-20s${C_RESET}\n" "MODULE" "PROVIDER" "STATUS" "AVAILABILITY"
  echo -e "${C_DIM}----------------------------------------------------------------------${C_RESET}"

  echo "$output" | grep "^ROS|" | while IFS='|' read -r _ name req status ver; do
    if [ "$status" = "OK" ]; then
      printf "%-25s %-15s ${C_GREEN}%-12s${C_RESET} ${C_BOLD}%-20s${C_RESET}\n" "$name" "$req" "[✓] LINKED" "$ver"
    else
      printf "%-25s %-15s ${C_YELLOW}%-12s${C_RESET} ${C_DIM}%-20s${C_RESET}\n" "$name" "$req" "[!] UNLINKED" "$ver"
    fi
  done
  echo ""
}

# If user only wanted audit/check, run check and exit
if [ "$CHECK_ONLY" = "true" ]; then
  render_audit "$PYTHON_BIN"
  echo -e "${C_GREEN}[✓] Dependency audit complete.${C_RESET}"
  exit 0
fi

# Step 5: Determine Target Environment
TARGET_PIP=""
TARGET_PY=""

# Check if active virtualenv exists
if [ -n "${VIRTUAL_ENV:-}" ] && [ "$MODE" != "system" ]; then
  echo -e "  ${C_GREEN}[✓] Detected active virtual environment:${C_RESET} ${C_BOLD}$VIRTUAL_ENV${C_RESET}"
  TARGET_PIP="$VIRTUAL_ENV/bin/pip"
  TARGET_PY="$VIRTUAL_ENV/bin/python"
elif [ "$MODE" = "venv" ] || [ "$MODE" = "auto" ]; then
  # Check if Debian PEP 668 externally managed environment is present
  EXT_MANAGED=false
  if python3 -c "import sys; exit(0 if hasattr(sys, 'prefix') else 1)" 2>/dev/null; then
    if pip install --dry-run numpy 2>&1 | grep -q "externally-managed-environment"; then
      EXT_MANAGED=true
    fi
  fi

  if [ "$MODE" = "auto" ] && [ "$EXT_MANAGED" = "false" ]; then
    # Standard unrestricted environment, can install directly with pip
    TARGET_PIP="pip3"
    TARGET_PY="python3"
  else
    # Create or activate virtual environment with system site-packages
    echo -e "  ${C_CYAN}[•] Preparing isolated Python environment at:${C_RESET} ${C_BOLD}$VENV_DIR${C_RESET}"
    if [ ! -d "$VENV_DIR" ]; then
      echo -e "  ${C_CYAN}[•] Creating virtualenv with --system-site-packages (keeps ROS 2 Jazzy visible)...${C_RESET}"
      python3 -m venv --system-site-packages "$VENV_DIR"
    else
      echo -e "  ${C_GREEN}[✓] Existing virtual environment found at $VENV_DIR${C_RESET}"
    fi
    TARGET_PIP="$VENV_DIR/bin/pip"
    TARGET_PY="$VENV_DIR/bin/python"
  fi
else
  # System mode explicitly requested
  TARGET_PIP="pip3"
  TARGET_PY="python3"
  EXTRA_PIP_FLAGS+=("--break-system-packages")
fi

# Step 6: Upgrade pip and install requirements
echo ""
echo -e "${C_BOLD}${C_CYAN}Installing NAVIGUARD Python dependencies...${C_RESET}"
echo -e "  ${C_DIM}Target Python : $TARGET_PY${C_RESET}"
echo -e "  ${C_DIM}Target Pip    : $TARGET_PIP${C_RESET}"

INSTALL_ARGS=("-r" "$REQ_FILE")
if [ "$UPGRADE" = "true" ]; then
  INSTALL_ARGS+=("--upgrade")
fi
if [ ${#EXTRA_PIP_FLAGS[@]} -gt 0 ]; then
  INSTALL_ARGS+=("${EXTRA_PIP_FLAGS[@]}")
fi

# Run pip installation
"$TARGET_PIP" install "${INSTALL_ARGS[@]}"

echo ""
echo -e "${C_GREEN}${C_BOLD}[✓] Package installation process finished successfully!${C_RESET}"

# Step 7: Post-installation verification audit
render_audit "$TARGET_PY"

echo -e "${C_BOLD}${C_WHITE}══════════════════════════════════════════════════════════════════════${C_RESET}"
echo -e "  ${C_GREEN}[✓] NAVIGUARD Python environment is fully configured and ready.${C_RESET}"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
  echo ""
  echo -e "  ${C_BOLD}To activate this environment in your current shell:${C_RESET}"
  echo -e "    ${C_CYAN}source $VENV_DIR/bin/activate${C_RESET}"
  echo -e "    ${C_CYAN}source /opt/ros/jazzy/setup.bash${C_RESET}"
  echo -e "    ${C_CYAN}source install/setup.bash${C_RESET}"
fi
echo -e "${C_BOLD}${C_WHITE}══════════════════════════════════════════════════════════════════════${C_RESET}"
