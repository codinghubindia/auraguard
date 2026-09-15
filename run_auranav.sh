#!/usr/bin/env bash
# ==============================================================================
# AURANAV — Master Interactive Launcher & Shell System
# SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV
# ==============================================================================
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$WS_ROOT/run_naviguard.sh" "$@"
