#!/usr/bin/env bash
# =============================================================================
# NAVIGUARD Master Demonstration & Operator Dashboard Launcher
# SIH 2026 Problem Statement 26126
#
# Unified wrapper delegating to root interactive launcher: run_naviguard.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$WS_ROOT/run_naviguard.sh" "$@"
