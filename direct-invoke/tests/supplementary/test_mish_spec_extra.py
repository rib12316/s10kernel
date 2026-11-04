#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""mish 规格外接口补充检查（独立入口、独立报告，不计入 cases.yaml 61 例通过数）。

依据：$WORK_DIR/0.1-黑盒测试设计.md（workflow4）第 4.3 节 / 第 8.4 节定稿。
- S1（断言型）：不支持 dtype（int32）必须明确异常拒绝，不得静默输出错误数值。
- S2（信息型旁证）：rank-0 标量自 workflow4 起已入规格（R3「0 维标量按 1 元素」），
  主覆盖移入 cases.yaml 黑盒用例 C53~55；本测试仅保留为接口行为旁证：
  支持则结果必须与 golden 同口径一致；不支持则按 skip 记录报错信息留痕，
  缺陷归属由主用例 C53~55 判定（不因本旁证放宽或掩盖）。

继承自上轮资产（/workspace/算子挑战赛/mish/tests/supplementary/test_mish_spec_extra.py，
workflow3 执行记录 2 passed），本轮仅同步文档说明；断言逻辑与 checker 口径未变。
断言复用 kernel_eval.utils.compare.compare_tensors 同口径，不退回 allclose。

前置条件：cann_bench wheel 已安装到当前 Python 环境、NPU 可用。
模块级 import 与 examples/direct_launch_example/tests/add/test_add.py 同约定：
cann_bench 未安装时收集失败，大声暴露环境/安装问题，不静默跳过。

运行：
    pytest direct-invoke/tests/supplementary/test_mish_spec_extra.py -v
"""

import logging

import pytest
import torch
import torch_npu
import cann_bench

# 复用评测器同一 checker 口径做断言，不退回 allclose（PYTHONPATH 含 cann-bench/src）
from kernel_eval.utils.compare import compare_tensors

logger = logging.getLogger(__name__)


def _golden_fp64_cpu(x: torch.Tensor) -> torch.Tensor:
    """golden 参考：输入升 float64 在 CPU 上调用 torch.nn.functional.mish。

    与评测器默认 golden_precision=fp64_cpu 口径一致。
    """
    return torch.nn.functional.mish(x.detach().to("cpu", torch.float64))


def test_unsupported_dtype_rejected():
    """S1 断言型：int32 输入必须明确异常，不得静默输出错误数值。

    PyTorch F.mish 对整型输入抛 RuntimeError；mish 包装层对规格外 dtype
    的预期行为是明确拒绝（对齐 golden 行为），静默返回即失败。
    """
    x = torch.zeros(4, dtype=torch.int32, device="npu")
    with pytest.raises(Exception):
        cann_bench.mish(x)


def test_rank0_scalar_informative():
    """S2 信息型旁证：rank-0 接口行为留痕（规格内主覆盖见 cases.yaml C53~55）。

    支持则：shape 保持 rank-0，数值与 FP64 golden 截断口径按 relative_error
    checker 判定一致；不支持则：记录报错并 skip 留痕。workflow4 起 rank-0 已入
    规格（R3），若实现不支持属算子缺陷，由主用例 C53~55 失败判定并交修复，
    本旁证不替代主覆盖、不构成放宽。
    """
    x = torch.tensor(1.5, dtype=torch.float32, device="npu")
    try:
        y = cann_bench.mish(x)
    except Exception as e:  # noqa: BLE001 - 旁证按预期记录接口行为，缺陷归属由 C53~55 判定
        pytest.skip(f"rank-0 调用报错（缺陷归属由主用例 C53~55 判定，此处仅留痕）: {type(e).__name__}: {e}")

    assert y.shape == x.shape, f"rank-0 支持时 shape 须保持: y={y.shape}, x={x.shape}"
    assert y.dtype == x.dtype, f"rank-0 支持时 dtype 须保持: y={y.dtype}, x={x.dtype}"

    golden_fp64 = _golden_fp64_cpu(x)
    golden_truncated = golden_fp64.to(y.dtype).double()
    result = compare_tensors(
        y.detach().to("cpu").double(), golden_truncated,
        dtype=str(y.dtype).removeprefix("torch."),
    )
    logger.info(
        "S2 rank-0 标量对比: passed=%s mere=%.6g mare=%.6g threshold=%.6g",
        result.passed, result.mere, result.mare, result.threshold,
    )
    assert result.passed, f"rank-0 数值与 golden 不一致: {result.error_msg}"
