#!/usr/bin/env bash
# ==============================================================================
# AURANAV — Python Dependency Installer Wrapper
# ==============================================================================
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$WS_ROOT/install_python_deps.sh" "$@"
