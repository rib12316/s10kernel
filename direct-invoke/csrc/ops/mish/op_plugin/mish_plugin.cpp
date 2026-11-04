/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mish_plugin.cpp
 * \brief Mish API layer - torch bindings (compiled with g++)
 *
 * Wrapper responsibilities kept to the allowed set: dtype/shape query, tiling
 * calculation, output allocation, current-device stream fetch, kernel launch.
 * No Torch/CANN built-in operator performs any part of the mish computation
 * and no data movement, cast or layout transform happens here - output is an
 * empty_like allocation filled by the submitted kernel. rank-0 tensors pass
 * through as numel()==1 (workflow4 R3: 0-dim scalar handled as 1 element);
 * empty tensors (numel()==0) launch one guarded no-op core.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "../op_kernel/mish_launch.h"

namespace cann_bench {

TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("mish(Tensor x) -> Tensor");
}

torch::Tensor mish_meta(const torch::Tensor &x)
{
    return torch::empty_like(x);
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("mish", mish_meta);
}

torch::Tensor mish_npu(const torch::Tensor &x)
{
    TORCH_CHECK(
        x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16 ||
            x.scalar_type() == torch::kBFloat16,
        "mish: unsupported dtype ", x.scalar_type(), ", expected float32/float16/bfloat16.");
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = mish_meta(x);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = x.numel();
    int64_t dtypeWidth = x.element_size();
    const bool isBfloat16 = (x.scalar_type() == torch::kBFloat16);
    int64_t numBlocks, blockLength, tileElems, granuleRem;
    std::tie(numBlocks, blockLength, tileElems, granuleRem) =
        calc_mish_tiling_params(totalLength, dtypeWidth, isBfloat16);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        if (x.scalar_type() == torch::kFloat32) {
            MISH_KERNEL_LAUNCH_FLOAT(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, granuleRem, stream);
        } else if (x.scalar_type() == torch::kFloat16) {
            MISH_KERNEL_LAUNCH_HALF(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, granuleRem, stream);
        } else if (x.scalar_type() == torch::kBFloat16) {
            MISH_KERNEL_LAUNCH_BFLOAT16(x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileElems, granuleRem, stream);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("Mish", acl_call);
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("mish", mish_npu);
}

} // namespace cann_bench
