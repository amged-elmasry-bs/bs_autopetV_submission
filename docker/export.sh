#!/bin/bash
# Build and save the image as a portable archive.
set -euo pipefail
"$(dirname "$0")/build.sh"
docker save bs-submission:latest | gzip -c > bs-submission.tar.gz
echo "wrote bs-submission.tar.gz"
