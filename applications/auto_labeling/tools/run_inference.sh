#!/usr/bin/env bash

set -euo pipefail
app_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$app_repo_root"
exec python -m applications.auto_labeling.inference "$@"
