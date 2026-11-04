#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# mish ST 统一执行入口（复用 cann-bench kernel_eval 评测器，不自建比对框架）。
# workflow4 物化自 /workspace/算子挑战赛/mish/tests/run_st.sh（workflow2 创建、workflow3 沿用），
# 变更：环境脚本默认路径 → workflow4；容器内授权设备默认 0（host NPU 3，910B4）；
# --list 模式同样执行 --expect-cases 发现数核对；用例基线 52 → 61。
#
# 功能：
#   1) 发现核对：列出当前任务集发现的用例数（--list；基线 61 = 黑盒 50 + 白盒 11）
#   2) 全量黑盒精度：kernel_eval eval --no-perf（默认模式）
#   3) 规格外接口补充检查：pytest S1/S2（独立报告，不计入黑盒通过数）
#
# 白盒接入方式：白盒用例按 repo-test-develop 约定追加到
# tests/cannbench/tasks/level1/mish/cases.yaml（保留原 61 例 ID 与内容不变，
# 新增白盒自 case_id 62 起），本入口按发现数原样执行，无需修改；
# 追加后用 --expect-cases <新总数> 核对发现数，并重新派生 cases.csv（不得只改 CSV）。
#
# 用法：
#   tests/run_st.sh --reports-dir DIR --eval-code CODE [选项]
# 选项：
#   --source-dir DIR      候选源码构建快照目录；不传则评测已安装的 cann_bench 包
#   --device-id N         NPU 设备 ID（默认 0 = 容器内唯一 910B4；须属于授权资源；
#                         性能/精度执行前建议 npu-smi info 核查占用）
#   --case-id N           仅执行指定用例（开发期穿刺定位用；正式全量不得带此参数）
#   --expect-cases N      发现数核对：实际发现数不等于 N 时报错退出（全量验收须核对）
#   --list                仅列出任务发现（任务/用例清单），不执行评测
#   --skip-supplementary  跳过补充 pytest（如 cann_bench 尚未安装时仅跑精度评测）
#   --supplementary-only  仅运行补充 pytest S1/S2
#   --accuracy-only       等价于 --skip-supplementary（兼容书写习惯）
# 退出码：成功 0；任何子步骤失败按其原始退出码退出。
# 日志：本脚本输出到 stdout/stderr，由调用方重定向留痕；评测原始报告写 --reports-dir。

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DIR="${MISH_TASK_DIR:-$SCRIPT_DIR/cannbench/tasks}"
SUPPLEMENTARY_DIR="${MISH_SUPPLEMENTARY_DIR:-$SCRIPT_DIR/supplementary}"
ENV_SCRIPT="${MISH_ENV_SCRIPT:-/workspace/算子挑战赛/.cannbot/mish/workflow4/运行环境.sh}"

# 环境激活：优先任务环境脚本（含 CANN set_env、PYTHON、PYTHONPATH）；
# 脚本缺失时回退到环境记录文档化的默认值（环境信息.md v1.0 + workflow4 复用记录）。
if [ -f "$ENV_SCRIPT" ]; then
    # shellcheck disable=SC1090
    source "$ENV_SCRIPT"
fi
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="/workspace/.cannbot/dependencies/ops-direct-invoke/cann-bench/src:${PYTHONPATH:-}"

OPERATOR="mish"
SOURCE_DIR=""
DEVICE_ID="0"
CASE_ID=""
EXPECT_CASES=""
REPORTS_DIR=""
EVAL_CODE=""
MODE="eval"            # eval | list | supplementary_only
RUN_SUPPLEMENTARY="1"

fail() {
    echo "[run_st] ERROR: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --source-dir)       SOURCE_DIR="$2"; shift 2 ;;
        --device-id)        DEVICE_ID="$2"; shift 2 ;;
        --case-id)          CASE_ID="$2"; shift 2 ;;
        --expect-cases)     EXPECT_CASES="$2"; shift 2 ;;
        --reports-dir)      REPORTS_DIR="$2"; shift 2 ;;
        --eval-code)        EVAL_CODE="$2"; shift 2 ;;
        --list)             MODE="list"; shift ;;
        --supplementary-only) MODE="supplementary_only"; shift ;;
        --skip-supplementary|--accuracy-only) RUN_SUPPLEMENTARY="0"; shift ;;
        *) fail "未知参数: $1（用法见脚本头部注释）" ;;
    esac
done

echo "[run_st] PYTHON=$PYTHON"
echo "[run_st] TASK_DIR=$TASK_DIR"
echo "[run_st] mode=$MODE source_dir=${SOURCE_DIR:-<已安装cann_bench>} device_id=$DEVICE_ID"

# ---------- 1) 发现核对 ----------
list_cases() {
    echo "[run_st] 发现核对: kernel_eval list --cases"
    "$PYTHON" -m kernel_eval.cli list --bench-name cann \
        --task-dir "$TASK_DIR" --operator "$OPERATOR" --cases
    return $?
}

# 解析官方 list 输出的「共 N 个用例」行（kernel_eval 原生口径）
list_output=$(list_cases 2>&1)
list_rc=$?
discovered=$(printf '%s\n' "$list_output" | grep -oE '共 [0-9]+ 个用例' | grep -oE '[0-9]+' | head -1)
echo "[run_st] 发现数（官方 list 口径）=${discovered:-未知} (list_rc=$list_rc)"

if [ "$MODE" = "list" ]; then
    # --list 模式同样执行 --expect-cases 核对（发现数与预期一致是验收硬项，不因模式而豁免）
    if [ -n "$EXPECT_CASES" ]; then
        [ "$list_rc" -eq 0 ] || fail "发现核对命令失败 (list_rc=$list_rc)，无法核对期望用例数"
        [ -n "$discovered" ] || fail "无法解析发现数（官方 list 输出无「共 N 个用例」行）"
        if [ "$discovered" -ne "$EXPECT_CASES" ]; then
            fail "发现数核对失败: 期望 $EXPECT_CASES, 实计 $discovered（先核对任务集，再谈全量通过）"
        fi
        echo "[run_st] 发现数核对通过: $discovered"
    fi
    list_cases
    exit $?
fi

if [ -n "$EXPECT_CASES" ]; then
    [ "$list_rc" -eq 0 ] || fail "发现核对命令失败 (list_rc=$list_rc)，无法核对期望用例数"
    [ -n "$discovered" ] || fail "无法解析发现数（官方 list 输出无「共 N 个用例」行）"
    if [ "$discovered" -ne "$EXPECT_CASES" ]; then
        fail "发现数核对失败: 期望 $EXPECT_CASES, 实计 $discovered（先核对任务集，再谈全量通过）"
    fi
    echo "[run_st] 发现数核对通过: $discovered"
fi

# ---------- 2) 全量黑盒精度（--no-perf）----------
eval_exit=0
if [ "$MODE" != "supplementary_only" ]; then
    [ -n "$REPORTS_DIR" ] || fail "精度评测必须显式传 --reports-dir（每次使用新报告目录）"
    [ -n "$EVAL_CODE" ] || fail "精度评测必须显式传 --eval-code（报告留痕标识）"

    cmd=( "$PYTHON" -m kernel_eval.cli eval --bench-name cann
          --task-dir "$TASK_DIR" --operator "$OPERATOR"
          --device npu --device-id "$DEVICE_ID"
          --reports-dir "$REPORTS_DIR" --eval-code "$EVAL_CODE"
          --eval-seed 0 --no-perf )
    if [ -n "$SOURCE_DIR" ]; then
        cmd+=( --source-dir "$SOURCE_DIR" )
    fi
    if [ -n "$CASE_ID" ]; then
        echo "[run_st] 警告: 带 --case-id $CASE_ID 为开发期穿刺，不构成全量通过"
        cmd+=( --case-id "$CASE_ID" )
    fi

    echo "[run_st] 执行: ${cmd[*]}"
    "${cmd[@]}"
    eval_exit=$?
    echo "[run_st] kernel_eval 退出码: $eval_exit"
    if [ "$eval_exit" -ne 0 ]; then
        echo "[run_st] 精度评测失败，仍继续补充检查阶段（--supplementary-only 除外）" >&2
    fi
fi

# ---------- 3) 规格外接口补充检查 S1/S2 ----------
supp_exit=0
if [ "$RUN_SUPPLEMENTARY" = "1" ]; then
    echo "[run_st] 补充 pytest: $SUPPLEMENTARY_DIR（S1/S2 独立报告，不计入黑盒通过数）"
    "$PYTHON" -m pytest "$SUPPLEMENTARY_DIR" -v
    supp_exit=$?
    echo "[run_st] 补充 pytest 退出码: $supp_exit"
fi

# ---------- 汇总 ----------
echo "[run_st] 汇总: eval_exit=$eval_exit supplementary_exit=$supp_exit"
[ "$eval_exit" -eq 0 ] && [ "$supp_exit" -eq 0 ] || exit 1
exit 0
