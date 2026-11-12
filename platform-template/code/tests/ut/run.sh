#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="$ROOT/tests/ut/build"
cmake -S "$ROOT/tests/ut" -B "$BUILD" -DGTEST_SOURCE_DIR="${GTEST_SOURCE_DIR:-}" -DENABLE_INNER_KERNEL_UT="${ENABLE_INNER_KERNEL_UT:-OFF}"
cmake --build "$BUILD" --parallel "${CANNJUDGE_COMPILE_JOBS:-4}"
"$BUILD/op_host/mish_op_host_ut"
python3 "$ROOT/samples/sample.py" generate --directory "$ROOT/samples/build-cpu"
export SAMPLE_DIR="$ROOT/samples/build-cpu"
"$BUILD/op_kernel/mish_op_kernel_ut"
python3 "$ROOT/samples/sample.py" verify --directory "$SAMPLE_DIR"
if [ -x "$BUILD/op_api/mish_op_api_ut" ]; then
    export ASCEND_CUSTOM_OPP_PATH="$ROOT/build/packages/vendors/mish_custom${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    "$BUILD/op_api/mish_op_api_ut"
fi
