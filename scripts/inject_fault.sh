#!/usr/bin/env bash
# NAVIGUARD Controlled Fault Injection Tool
# Allows the operator to inject controlled perception / localization failures
# and verify autonomous recovery behavior.

source /opt/ros/jazzy/setup.bash
source /home/maxx/naviguard_ws/install/setup.bash

ACTION="$1"

if [ -z "$ACTION" ]; then
    echo "================================================================="
    echo " NAVIGUARD Controlled Fault Injection & Recovery Control"
    echo "================================================================="
    echo " 1) Trigger Autonomous Recovery (SAFE_STOP -> RECOVER -> RESUME)"
    echo " 2) Reset Recovery Budget & Return to NORMAL"
    echo " 3) Query Current Recovery State"
    echo " 4) Exit"
    echo "================================================================="
    read -p "Select option [1-4]: " choice
    case $choice in
        1) ACTION="trigger" ;;
        2) ACTION="reset" ;;
        3) ACTION="status" ;;
        *) exit 0 ;;
    esac
fi

case $ACTION in
    trigger|--trigger|-t)
        echo ">>> Injecting fault: Triggering /recovery/trigger_manual_recovery service..."
        ros2 service call /recovery/trigger_manual_recovery std_srvs/srv/Trigger "{}"
        echo ">>> Checking recovery state transition:"
        ros2 topic echo /recovery/state --once 2>/dev/null || true
        ;;
    reset|--reset|-r)
        echo ">>> Resetting recovery budget via /recovery/reset_budget service..."
        ros2 service call /recovery/reset_budget std_srvs/srv/Trigger "{}"
        echo ">>> Checking recovery state transition:"
        ros2 topic echo /recovery/state --once 2>/dev/null || true
        ;;
    status|--status|-s)
        echo ">>> Current /recovery/state:"
        ros2 topic echo /recovery/state --once
        ;;
    *)
        echo "Usage: $0 [trigger|reset|status]"
        exit 1
        ;;
esac
