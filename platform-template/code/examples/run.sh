#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/mish_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$ROOT/build/packages/vendors/mish_custom/op_api/lib:$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
cmake -S "$ROOT/examples" -B "$ROOT/examples/build"
cmake --build "$ROOT/examples/build" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"
python3 "$ROOT/samples/sample.py" generate --directory "$ROOT/samples/build"
"$ROOT/examples/build/sample" "$ROOT/samples/build" "${DEVICE_ID:-0}"
python3 "$ROOT/samples/sample.py" verify --directory "$ROOT/samples/build"
