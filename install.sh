#!/usr/bin/env bash
# ==============================================================================
# NAVIGUARD — Quick Installer Entrypoint
# ==============================================================================
WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$WS_ROOT/install_python_deps.sh" "$@"
