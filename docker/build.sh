#!/bin/bash
# Build the training image from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
docker build --platform=linux/amd64 -f "$ROOT/docker/Dockerfile" -t bs-submission:latest "$ROOT"
