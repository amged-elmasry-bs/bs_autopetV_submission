#!/bin/bash
# Train one fold. Usage: scripts/train_fold.sh <fold> <data-root> [extra bs-train args...]
set -euo pipefail
FOLD="${1:?usage: train_fold.sh <fold> <data-root> [args...]}"
DATA="${2:?usage: train_fold.sh <fold> <data-root> [args...]}"
shift 2
RUN="runs/fold${FOLD}"
exec bs-train \
  --fold "$FOLD" \
  --data-root "$DATA" \
  --checkpoint-save-path "$RUN/checkpoints" \
  --log-root-dir "$RUN/logs" \
  "$@"
