#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch

"""
Mish 算子 Torch Golden 参考实现

计算输入张量的 Mish 激活函数：mish(x) = x * tanh(softplus(x))，softplus(x) = ln(1 + e^x)
对齐 PyTorch torch.nn.functional.mish（torch 内部为数值稳定实现）。

精度说明：
- kernel_eval 默认 golden_precision=fp64_cpu：输入先被评测器升为 float64 并在 CPU 上
  调用本函数，因此实际以 FP64 精度计算，checker 再将结果截断回输出 dtype 比对。
- 直接以原始 dtype 调用（脱离评测器）时，FP16/BF16 升 FP32 计算、FP32/FP64 原精度计算，
  与 tasks/level1/exp/golden.py 范本写法一致。
- 特殊值语义（torch 2.10.0 CPU 实测）：mish(+Inf)=+Inf、mish(-Inf)=NaN（-Inf*0）、
  mish(NaN)=NaN、mish(±0)=±0。
"""


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    计算输入张量的 Mish 激活函数

    y = x * tanh(softplus(x))，softplus(x) = ln(1 + e^x)

    Args:
        x: 输入张量（float16 / bfloat16 / float32）

    Returns:
        Mish 激活结果，shape/dtype 与输入一致
    """
    # 检测输入 dtype
    input_dtype = x.dtype

    # FP16/BF16 输入需要升到 FP32 计算以保证精度
    # FP32/FP64 输入保持原样计算
    if input_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = input_dtype

    # 转换到计算精度
    x_compute = x.to(compute_dtype)
    y = torch.nn.functional.mish(x_compute)

    # 转回原始 dtype
    if input_dtype in (torch.float16, torch.bfloat16):
        return y.to(input_dtype)
    return y
