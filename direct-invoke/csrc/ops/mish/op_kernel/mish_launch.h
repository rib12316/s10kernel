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
 * \file mish_launch.h
 * \brief Launch function declarations for g++
 */

#ifndef MISH_LAUNCH_H
#define MISH_LAUNCH_H

#include <cstdint>
#include <tuple>

#ifndef GM_ADDR
#define GM_ADDR void*
#endif

// Tiling function declaration: returns (numBlocks, blockLength, tileElems,
// granuleRem). tileElems is expressed in ELEMENTS of the input dtype; the
// per-dtype UB budget is derived from dtypeWidth (4 float, 2 half/bfloat16)
// plus isBfloat16 - half and bfloat16 share the same byte width but route to
// different compute chains (fp16 native E3 vs Cast->fp32), so their UB
// accounting differs (14 vs 24 B/element). granuleRem is the round-5 host
// computed remainder of the 512B-granule base+r multi-core split.
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_mish_tiling_params(
    int64_t totalLength, int64_t dtypeWidth, bool isBfloat16);

// Launch function declarations
extern "C" {
void launch_mish_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream);
void launch_mish_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream);
void launch_mish_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream);
}

// Convenience macros for API layer
#define MISH_KERNEL_LAUNCH_FLOAT(x, y, len, blocks, blkLen, tileElems, gRem, stream) \
    launch_mish_kernel_float(x, y, len, blocks, blkLen, tileElems, gRem, stream)
#define MISH_KERNEL_LAUNCH_HALF(x, y, len, blocks, blkLen, tileElems, gRem, stream) \
    launch_mish_kernel_half(x, y, len, blocks, blkLen, tileElems, gRem, stream)
#define MISH_KERNEL_LAUNCH_BFLOAT16(x, y, len, blocks, blkLen, tileElems, gRem, stream) \
    launch_mish_kernel_bfloat16(x, y, len, blocks, blkLen, tileElems, gRem, stream)

#endif // MISH_LAUNCH_H
