#!/bin/bash
# Serve the run logs. Usage: scripts/tensorboard.sh [logdir] [port]
set -euo pipefail
exec tensorboard --logdir "${1:-runs}" --host 0.0.0.0 --port "${2:-6006}" \
  --reload_multifile true --reload_interval 30
