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
 * \file mish_kernel.cpp
 * \brief Mish kernel (bisheng, --npu-arch=dav-2201 -xasc)
 *
 * workflow4 direct-invoke development vehicle. Round-7 adopted changes
 * (workflow4 performance node, 2026-11-04; A/B attributed on device, msprof
 * 10-pair median protocol, vs the round-6 baseline below):
 *   direction-3b: the single-tile lightweight path holds x / y / e / p / q
 *     in ONE TBuf sized to the actual per-core length (each region 512B
 *     aligned) instead of 5-6 full-tile TBuf allocations (~138 KiB per core
 *     for a 4 KiB workload before) - one InitBuffer call, smaller TPipe
 *     bookkeeping; measured -0.2..-0.35us on 64x64, bf16 -0.1..-0.23us
 *     across the small zone, no large-shape effect (multi-tile path keeps
 *     the merged-scratch form).
 *   direction-5: fine core split with a cap - numBlocks =
 *     min(ceil(total/2048), max(ceil(total/4096), 16)): doubles the core
 *     count only for totals <= 65536 within the measured 16-core optimum
 *     (64x64 -> 2 cores -0.3..-0.4us, 128x128 -> 8 cores -0.2..-0.85us);
 *     for totals > 65536 the expression degenerates exactly to the base
 *     ceil(total/4096) policy, so large-shape tiling is bit-identical to
 *     the round-6 baseline.
 * Rejected this round (A/B evidence in 3-性能优化报告.md): chain interleave
 * (direction-2: +2.7..+13us on large shapes, reverted), aligned-path
 * DataCopy replacing DataCopyPad (direction-4: no gain), 16B tiling argument
 * packing (direction-6: +0.3..+5us, launch-arg narrowing hurts this form).
 *
 * Kernel computation is a faithful
 * port of the round-6 platform baseline in s10kernel/code/op_kernel/{mish.h,
 * mish.cpp} (6 rounds of on-device validation); only the tiling delivery
 * differs: the platform version reads a MishTilingData struct from GM
 * (GET_TILING_DATA_WITH_STRUCT), the direct-launch form receives the same four
 * fields as scalar kernel arguments (totalLength/blockLength/tileElems/
 * granuleRem == totalNum/blockFactor/ubFactor/granuleRem). The host-side tiling
 * calculation (calc_mish_tiling_params below) ports s10kernel/code/op_host/
 * mish_tiling.cpp one-to-one.
 *
 * Mathematical route (round-3 single-Exp algebraic form, round-4 unified):
 *   mish(x) = x * tanh(softplus(x)), softplus(x) = ln(1 + e^x)
 *   e = exp(x); a = e+2; p = e*a; p' = min(p, dtypeMax); w = p'+2;
 *   q = x/w; y = q*p'        (E3 chain: 7 ops, 1 Div, scalar broadcasts)
 * Edge semantics (IEEE-754; platform task §3.6, on-device verified):
 *   +Inf -> +Inf; -Inf -> NaN; NaN -> NaN; x=±0 -> ±0 (sign preserved).
 *
 * dtype routing (branch condition is dtype identity only - a runtime generic
 * quantity):
 *   - float32:  fp32 E3 chain (2 fp32 scratch e/p actually allocated; the
 *     tile-size accounting deliberately retains the round-7 28 B/element)
 *   - float16:  native fp16 E3 chain with p' = min(p, 65504) clamp (2 fp16
 *     scratch e/p allocated, round-7 14 B/element accounting; fp16 vector
 *     pipeline does not FTZ subnormals - the quotient-first ordering keeps
 *     |q| <= 1 so y never exceeds the dtype maximum)
 *   - bfloat16: no native bf16 vector arithmetic on dav_c220; Cast -> fp32 E3
 *     chain -> CAST_RINT back (3 fp32 scratch x32/e/p allocated, round-7
 *     24 B/element accounting)
 *
 * Structure (rounds 2..6, all validated on device):
 *   - single-tile cores (small tensors: every assigned element fits in one UB
 *     tile) take a lightweight path: TBuf buffers + explicit MTE2_V / V_MTE3
 *     pipe events, no queue state machine; tile counts derived by comparison
 *     only (round-6: the last per-launch 64-bit integer division eliminated
 *     from this path)
 *   - multi-tile cores use double-buffered VECIN/VECOUT queues with prefetch;
 *     this is the only path that keeps the integer division for tileNum
 *   - multi-core split: whole 512B transfer granules, uniform base extent plus
 *     one extra granule for the first r cores; r is computed on the host and
 *     handed down via the granuleRem argument (round-5), so the kernel prologue
 *     re-derives nothing
 *   - empty tensors (totalLength == 0) launch one core that returns via the
 *     tiles == 0 guard without touching memory
 *
 * Reciprocal is deliberately NOT used (documented accuracy on A2 does not meet
 * the precision budget for float); L2CacheHint bypass was measured and
 * reverted in round-2 (kept default residency).
 */

#include <tuple>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "mish_launch.h"

namespace NsMish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int64_t GRANULE_BYTES = 512;  // MTE transfer granularity (bytes), matches host tiling
// dtype maximum finite values: upper bound of the E3 chain Mins clamp (a dtype
// range-saturation guard, not any test-derived constant)
constexpr float HALF_MAX_FINITE = 65504.0f;
constexpr float FLT_MAX_FINITE = 3.402823466e+38f;

// Unified E3 compute chain (fp32 direct / fp16 native / bfloat16 via Cast):
//   e = exp(x); a = e+2; p = e*a; p' = min(p, dtypeMax); w = p'+2; q = x/w; y = q*p'
// 7 ops, single Div, all constants scalar broadcasts (no constant tile, no
// prologue fill). Range safety (+Inf/-Inf/NaN/±0/large-positive exp-overflow
// region/small-negative subnormal region) documented in the platform baseline.
// round-8: only TWO scratch tensors (e, p). The Div lands directly in the
// output tensor y (q is consumed immediately by y = q*p'), removing the third
// scratch tile (qT) and its UB/buffer bookkeeping. For bfloat16 the fp32
// chain's output tensor is the Cast staging buffer x32, so the same aliasing
// applies. Buffer roles: xT read-only input; eT evolves e->w; pT evolves
// a->p' (after clamp); y receives q then the final result.
template <typename T>
__aicore__ inline constexpr T MishDtypeMaxFin()
{
    if constexpr (IsSameType<T, half>::value) {
        return static_cast<half>(HALF_MAX_FINITE);
    } else {
        return static_cast<T>(FLT_MAX_FINITE);
    }
}

template <typename T>
__aicore__ inline void MishE3Chain(const LocalTensor<T> xT, LocalTensor<T> eT,
    LocalTensor<T> pT, LocalTensor<T> y, int32_t count)
{
    constexpr T kDtypeMax = MishDtypeMaxFin<T>();
    Exp<T>(eT, xT, count);                              // e = exp(x)
    Adds<T>(pT, eT, static_cast<T>(2.0), count);        // a = e + 2
    Mul<T>(pT, eT, pT, count);                          // p = e(e+2)
    Mins<T>(pT, pT, kDtypeMax, count);                  // p' = min(p, dtype maximum finite)
    Adds<T>(eT, pT, static_cast<T>(2.0), count);        // w = p' + 2 (reuse eT)
    Div<T>(y, xT, eT, count);                           // q = x / w  (y doubles as the q slot)
    Mul<T>(y, y, pT, count);                            // y = q * p'
}

// dtype routing (compile-time, keyed on dtype identity)
template <typename T>
__aicore__ inline constexpr bool MishIsHalf()
{
    return IsSameType<T, half>::value;
}

// round-8 single-tile raw layout helpers (bytes aligned to the 512B MTE
// granule; the LocalTensor descriptor length is rounded to 32B because the
// raw constructor requires a 32B-aligned byte length).
__aicore__ inline int64_t AlignGranule(int64_t bytes)
{
    return (bytes + GRANULE_BYTES - 1) / GRANULE_BYTES * GRANULE_BYTES;
}

template <typename T>
__aicore__ inline uint32_t RawTensorElems(int32_t count)
{
    constexpr int64_t kElemBytes = static_cast<int64_t>(sizeof(T));
    const int64_t bytes = (static_cast<int64_t>(count) * kElemBytes + 31) / 32 * 32;
    return static_cast<uint32_t>(bytes / kElemBytes);
}

template <typename T>
class Mish {
public:
    __aicore__ inline Mish(){};

    // Tiling scalars: totalLength/blockLength/tileElems/granuleRem mirror the
    // platform MishTilingData fields totalNum/blockFactor/ubFactor/granuleRem.
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, int64_t totalNum, int64_t blockFactor,
        int64_t ubFactor, int64_t granuleRem);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t tileIdx, int64_t tileElemNum);
    __aicore__ inline void SingleTileProcess();
    // round-7 direction-3: element-offset view into the shared scratch TBuf
    // (slot extent = one tile of the compute dtype). Multi-tile path only;
    // the single-tile path uses raw UB addresses and constructs its tensors
    // directly (round-8 TPipe-free layout).
    template <typename CT>
    __aicore__ inline AscendC::LocalTensor<CT> ScratchSlot(int32_t slot)
    {
        return bufScratch.template Get<CT>()[static_cast<uint32_t>(slot) * static_cast<uint32_t>(elemPerTile_)];
    }

private:
    // round-8: no TPipe member. The single-tile path is TPipe-free (raw UB
    // tensors + fixed event IDs); the multi-tile path constructs a local
    // TPipe in Process() and allocates only there.
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> bufScratch;               // one allocation, hand-sliced

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalLength_ = 0;
    int64_t blockLength_ = 0;
    int64_t elemPerTile_ = 0;
    int64_t currentBlockLength_ = 0;
    int64_t tileNum_ = 0;
    int64_t tailTileElementNum_ = 0;
    int64_t tiles_ = 0;
    bool singleTile_ = false;
};

template <typename T>
__aicore__ inline void Mish<T>::Init(GM_ADDR x, GM_ADDR y, int64_t totalNum, int64_t blockFactor,
    int64_t ubFactor, int64_t granuleRem)
{
    totalLength_ = totalNum;
    // blockFactor is the 512B-granule uniform base length (first r cores carry
    // one extra granule; r re-assembled below from granuleRem).
    blockLength_ = blockFactor;
    elemPerTile_ = ubFactor;

    // Per-core extent: 512B granule base+r split (round-5: r handed down by the
    // host via the granuleRem tiling field, always 0 <= r < numBlocks - no
    // 64-bit division and no GetBlockNum() dependency). The last granule is
    // clamped to the real end of the tensor.
    const int64_t blkIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t elemsPerGranule = GRANULE_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t r = granuleRem;
    int64_t blockStart = blkIdx * blockLength_ + (blkIdx < r ? blkIdx : r) * elemsPerGranule;
    currentBlockLength_ = blockLength_ + (blkIdx < r ? elemsPerGranule : 0);
    if (blockStart >= totalLength_) {
        blockStart = totalLength_;
        currentBlockLength_ = 0;
    } else if (blockStart + currentBlockLength_ > totalLength_) {
        currentBlockLength_ = totalLength_ - blockStart;
    }

    // round-6: single-tile cores decide by comparison (no 64-bit integer
    // division; this is the small-shape main path); only multi-tile cores keep
    // the division derivation. The tileNum/tail representation difference on
    // the single-tile branch is never consumed downstream (SingleTileProcess
    // only uses currentBlockLength_).
    singleTile_ = (currentBlockLength_ > 0) && (currentBlockLength_ <= elemPerTile_);
    if (singleTile_) {
        tileNum_ = 0;
        tailTileElementNum_ = currentBlockLength_;
        tiles_ = 1;
    } else if (currentBlockLength_ > 0) {
        tileNum_ = currentBlockLength_ / elemPerTile_;
        tailTileElementNum_ = currentBlockLength_ - tileNum_ * elemPerTile_;
        tiles_ = tileNum_ + (tailTileElementNum_ > 0 ? 1 : 0);
    } else {
        tileNum_ = 0;
        tailTileElementNum_ = 0;
        tiles_ = 0;  // empty-tensor guard (tiles == 0: Process returns immediately)
    }

    // round-8: no buffer allocation in Init. The single-tile path lays out
    // x/y/e/p as raw UB regions starting at address 0 (no TPipe/InitBuffer at
    // all); the multi-tile path allocates its queues + scratch in Process().

    inputGMX.SetGlobalBuffer((__gm__ T *)x + blockStart);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + blockStart);
}

// Submit one tile copy-in (DataCopyPad accepts any byte length, incl. misaligned tails)
template <typename T>
__aicore__ inline void Mish<T>::CopyInTile(int64_t tileIdx, int64_t tileElemNum)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileElemNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    auto xLocal = inputQueueX.AllocTensor<T>();
    DataCopyPad(xLocal, inputGMX[tileIdx * elemPerTile_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

// Single-tile TPipe-free path (round-8): raw UB layout at address 0
//   native  : [x][y][e][p]          (regions 512B-aligned)
//   bfloat16: [x][y][x32][e][p]     (scratch regions fp32-wide)
// No TPipe constructor, no InitBuffer, no TBuf bookkeeping, no queue state
// machine; fixed hardware event IDs (this path never touches the queue event
// pool). InitSocState() is called explicitly: TPipe's constructor is what
// normally performs it (set_atomic_none + set_mask_norm + vector-mask init).
template <typename T>
__aicore__ inline void Mish<T>::SingleTileProcess()
{
    constexpr bool kIsHalf = MishIsHalf<T>();
    constexpr bool kIsFloat = (sizeof(T) == static_cast<size_t>(FLOAT_BYTES));
    constexpr int64_t scratchWidth = kIsFloat ? FLOAT_BYTES : (kIsHalf ? 2 : FLOAT_BYTES);

    AscendC::InitSocState();  // TPipe normally does this in its constructor

    const int64_t count64 = currentBlockLength_;
    const int32_t count = static_cast<int32_t>(count64);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count64 * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // 512B-aligned region sizes: x/y are one T element per element; scratch
    // uses the compute dtype (T for fp32/fp16, fp32 for bfloat16).
    const int64_t region = AlignGranule(count64 * static_cast<int64_t>(sizeof(T)));
    const int64_t regionS = AlignGranule(count64 * scratchWidth);
    const uint32_t xAddr = 0;
    const uint32_t yAddr = static_cast<uint32_t>(region);
    const uint32_t sAddr = static_cast<uint32_t>(2 * region);

    auto xT = LocalTensor<T>(TPosition::VECCALC, xAddr, RawTensorElems<T>(count));
    auto yT = LocalTensor<T>(TPosition::VECCALC, yAddr, RawTensorElems<T>(count));
    DataCopyPad(xT, inputGMX[0], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(event_t::EVENT_ID0);

    if constexpr (kIsFloat || kIsHalf) {
        auto eT = LocalTensor<T>(TPosition::VECCALC, sAddr, RawTensorElems<T>(count));
        auto pT = LocalTensor<T>(TPosition::VECCALC,
            sAddr + static_cast<uint32_t>(region), RawTensorElems<T>(count));
        MishE3Chain<T>(xT, eT, pT, yT, count);
    } else {
        // bfloat16 -> fp32 compute -> round-to-nearest-even back.
        // x32 doubles as the chain output; e/p are fp32 scratch.
        auto x32 = LocalTensor<float>(TPosition::VECCALC, sAddr, RawTensorElems<float>(count));
        auto eT = LocalTensor<float>(TPosition::VECCALC,
            sAddr + static_cast<uint32_t>(regionS), RawTensorElems<float>(count));
        auto pT = LocalTensor<float>(TPosition::VECCALC,
            sAddr + static_cast<uint32_t>(2 * regionS), RawTensorElems<float>(count));
        Cast<float, T>(x32, xT, RoundMode::CAST_NONE, count);
        MishE3Chain<float>(x32, eT, pT, x32, count);
        Cast<T, float>(yT, x32, RoundMode::CAST_RINT, count);
    }

    SetFlag<HardEvent::V_MTE3>(event_t::EVENT_ID1);
    WaitFlag<HardEvent::V_MTE3>(event_t::EVENT_ID1);
    DataCopyPad(outputGMY[0], yT, copyParams);
}

template <typename T>
__aicore__ inline void Mish<T>::Process()
{
    constexpr bool kIsHalf = MishIsHalf<T>();
    constexpr bool kIsFloat = (sizeof(T) == static_cast<size_t>(FLOAT_BYTES));
    if (tiles_ <= 0) {
        return;
    }
    if (singleTile_) {
        SingleTileProcess();  // round-8: TPipe-free raw-UB path
        return;
    }

    // round-8: the multi-tile path is the only TPipe user; construct the pipe
    // here so single-tile launches never pay the constructor/pool-reset cost.
    // Scratch count dropped by one: y (or x32 for bfloat16) doubles as q.
    constexpr int64_t scratchWidth = kIsFloat ? FLOAT_BYTES : (kIsHalf ? 2 : FLOAT_BYTES);
    constexpr int64_t SCRATCH_COUNT = (kIsFloat || kIsHalf) ? 2 : 3;
    const int64_t tileBytesT = elemPerTile_ * static_cast<int64_t>(sizeof(T));
    const int64_t tileBytesS = elemPerTile_ * scratchWidth;
    TPipe pipe;
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(bufScratch, SCRATCH_COUNT * tileBytesS);

    // Prologue: issue tile 0 copy-in so the loop can prefetch tile i+1 before
    // consuming tile i.
    {
        const int64_t firstN = (tileNum_ > 0) ? elemPerTile_ : tailTileElementNum_;
        CopyInTile(0, firstN);
    }

    for (int64_t i = 0; i < tiles_; ++i) {
        // Prefetch tile i+1 so the MTE copy-in overlaps this tile's vector
        // compute and copy-out.
        if (i + 1 < tiles_) {
            const int64_t nextN = (i + 1 < tileNum_) ? elemPerTile_ : tailTileElementNum_;
            CopyInTile(i + 1, nextN);
        }

        const int32_t count = static_cast<int32_t>((i < tileNum_) ? elemPerTile_ : tailTileElementNum_);
        auto xLocal = inputQueueX.DeQue<T>();
        auto yLocal = outputQueueY.AllocTensor<T>();
        if constexpr (kIsFloat) {
            MishE3Chain<float>(xLocal.template ReinterpretCast<float>(), ScratchSlot<float>(0),
                ScratchSlot<float>(1), yLocal.template ReinterpretCast<float>(), count);
        } else if constexpr (kIsHalf) {
            MishE3Chain<half>(xLocal, ScratchSlot<half>(0), ScratchSlot<half>(1), yLocal, count);
        } else {
            // bfloat16 -> fp32 compute -> round-to-nearest-even back
            auto x32 = ScratchSlot<float>(0);
            Cast<float, T>(x32, xLocal, RoundMode::CAST_NONE, count);
            MishE3Chain<float>(x32, ScratchSlot<float>(1), ScratchSlot<float>(2), x32, count);
            Cast<T, float>(yLocal, x32, RoundMode::CAST_RINT, count);
        }
        outputQueueY.EnQue<T>(yLocal);
        inputQueueX.FreeTensor(xLocal);
        yLocal = outputQueueY.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(static_cast<int64_t>(count) * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
        DataCopyPad(outputGMY[i * elemPerTile_], yLocal, copyParams);
        outputQueueY.FreeTensor(yLocal);
    }
}

} // namespace NsMish

// Kernel entry (direct launch form). Scalar tiling arguments mirror the
// platform version's MishTilingData {totalNum, blockFactor, ubFactor,
// granuleRem}; the dtype template parameter replaces the platform tiling key.
template <typename T>
__global__ __aicore__ void mish_kernel(GM_ADDR x, GM_ADDR y, int64_t totalLength,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem)
{
    NsMish::Mish<T> op;
    op.Init(x, y, totalLength, blockLength, static_cast<int64_t>(tileElems), granuleRem);
    op.Process();
}

// Host tiling calculation: ports s10kernel/code/op_host/mish_tiling.cpp
// one-to-one. All parameters derive from runtime-queried generic quantities
// (AIV core count, UB size, dtype width/identity) - no shape specialization.
// Returns (numBlocks, blockLength, tileElems, granuleRem).
std::tuple<int64_t, int64_t, int64_t, int64_t> calc_mish_tiling_params(
    int64_t totalLength, int64_t dtypeWidth, bool isBfloat16)
{
    constexpr int64_t PIPELINE_DEPTH = 2;
    constexpr int64_t MIN_SPLIT_THRESHOLD = 4096;  // min elements per core (round-3: 1024 -> 4096)
    // round-8: the kernel actually allocates only 2 native / 3 bf16 scratch
    // (q is folded into y/x32), but the tile-size accounting deliberately
    // keeps the round-7 granularity (3 native / 4 bf16 scratch equivalent).
    // A/B on the mini case set (2026-11-12) showed that converting the freed
    // UB into larger tiles regressed the medium multi-tile points by
    // 1.0-3.4% (coarser double-buffer pipeline granularity), while small
    // shapes were unchanged; the extra UB remains headroom. The constants
    // derive from ubSize and dtype width only - no shape specialization.
    constexpr int64_t SCRATCH_NATIVE = 3;          // tile-size accounting: round-7 native granularity
    constexpr int64_t SCRATCH_BF16 = 4;            // tile-size accounting: round-7 bf16 granularity
    constexpr int64_t FLOAT_BYTES = 4;
    constexpr int64_t GRANULE_BYTES = 512;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    // Elements per tile from the conservative round-7 accounting, routed by
    // dtype identity (dtypeWidth alone cannot separate half from bfloat16):
    //   float32:  3 fp32-equivalent scratch -> 12 + 16 = 28 B/element -> 6912
    //   float16:  native fp16 chain, 3 fp16-equivalent scratch -> 6 + 8 = 14 B/element -> 13824
    //   bfloat16: Cast route, 4 fp32-equivalent scratch -> 16 + 8 = 24 B/element;
    //             the exact value claims 100.0% of UB, so 512 elements (~12 KB)
    //             are reserved for the system area
    // The kernel allocates one scratch fewer than accounted (round-8 q fold);
    // the difference is intentional headroom, see SCRATCH_NATIVE above.
    // Queue side costs PIPELINE_DEPTH * 2 (in/out) * dtypeWidth bytes/element.
    const bool nativeWidth = (dtypeWidth == FLOAT_BYTES) || (dtypeWidth == 2 && !isBfloat16);
    const int64_t scratchBytes = nativeWidth ? (SCRATCH_NATIVE * dtypeWidth) : (SCRATCH_BF16 * FLOAT_BYTES);
    const int64_t bytesPerElement = scratchBytes + PIPELINE_DEPTH * 2 * dtypeWidth;
    int64_t tileElems = static_cast<int64_t>(ubSize) / bytesPerElement;
    if (!nativeWidth) {
        tileElems -= 512;  // bf16 exact value claims 100.0% of UB; keep margin
    }
    // Floor to the 512B transfer granularity of the dtype (MTE efficiency),
    // never below one granule.
    const int64_t elemsPerGranule = GRANULE_BYTES / dtypeWidth;
    tileElems = tileElems / elemsPerGranule * elemsPerGranule;
    if (tileElems < elemsPerGranule) {
        tileElems = elemsPerGranule;
    }

    // Core count: small tensors capped at MIN_SPLIT_THRESHOLD elements per core
    // (fewer cores amortize block-dispatch fixed overhead); the branch keys on
    // total elements and core count - runtime generic quantities.
    // round-7 direction-5 rescan (A/B measured core-count curve, 2026-11-04):
    // halving the per-core minimum doubles the core count, which measured
    // faster for mid-size small tensors (4096-element 64x64: -0.35us at 2
    // cores vs 1; 16384-element 128x128: -0.85us fp32 at 8 cores vs 4) but
    // slower once the doubled count exceeds the measured optimum (65536-
    // element 256x256: +0.34us at 32 cores vs 16).  The adopted policy keeps
    // the doubled granularity ONLY while it stays within the measured
    // 16-core optimum of the latency-bound zone; for any total > 65536 the
    // expression degenerates EXACTLY to the base ceil(total/4096) policy
    // (capped by coreNum below), so large-shape tiling is bit-identical to
    // the round-6 baseline.  Branch inputs are runtime generic quantities
    // (total element count, core count) - no shape specialization.
    constexpr int64_t SPLIT_THRESHOLD_FINE = 2048;    // doubled per-core granularity
    constexpr int64_t SMALL_ZONE_CORE_CAP = 16;       // measured optimum at the zone edge
    int64_t numBlocksFine = (totalLength + SPLIT_THRESHOLD_FINE - 1) / SPLIT_THRESHOLD_FINE;
    int64_t numBlocksCoarse = (totalLength + MIN_SPLIT_THRESHOLD - 1) / MIN_SPLIT_THRESHOLD;
    int64_t capOrCoarse = (numBlocksCoarse > SMALL_ZONE_CORE_CAP) ? numBlocksCoarse : SMALL_ZONE_CORE_CAP;
    int64_t numBlocks = (numBlocksFine < capOrCoarse) ? numBlocksFine : capOrCoarse;
    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }
    if (numBlocks <= 0) {
        numBlocks = 1;  // totalLength == 0 still launches one no-op core (guarded)
    }
    // 512B-granule uniform split: blockLength carries the uniform base extent;
    // the first r cores carry one extra granule (r = unitTotal - base*numBlocks,
    // always 0 <= r < numBlocks). Every core starts 512B-aligned; inter-core
    // skew is at most one granule.
    const int64_t unitTotal = (totalLength + elemsPerGranule - 1) / elemsPerGranule;
    const int64_t baseUnits = unitTotal / numBlocks;
    const int64_t blockLength = baseUnits * elemsPerGranule;
    const int64_t granuleRem = unitTotal - baseUnits * numBlocks;
    return std::make_tuple(numBlocks, blockLength, tileElems, granuleRem);
}

// Launch wrappers - regular C functions callable from g++
extern "C" {

void launch_mish_kernel_float(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream)
{
    mish_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems, granuleRem);
}

void launch_mish_kernel_half(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream)
{
    mish_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems, granuleRem);
}

void launch_mish_kernel_bfloat16(GM_ADDR x, GM_ADDR y, int64_t totalLength, int64_t numBlocks,
    int64_t blockLength, uint32_t tileElems, int64_t granuleRem, void* stream)
{
    mish_kernel<bfloat16_t><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileElems, granuleRem);
}

}
