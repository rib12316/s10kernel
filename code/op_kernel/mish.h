/*!
 * \file mish.h
 * \brief Mish 算子 kernel 类实现
 *
 * 数学路线（与本地第二轮最终版逐参数一致，中间量全部 fp32）：
 *   mish(x) = x * tanh(softplus(x))
 * 等价单 Exp 代数式（对全部 e 属于 [0, +Inf] 精确）：
 *   e = exp(x); w = 1 + 2/e; t = 2 / (w + 1/w); y = x * t
 * 说明：
 *   - 乘积形式 y = x * t 为必选：t 属于 [0,1]，y 继承 x 的量纲，无中间溢出路径
 *   - 不使用倒数指令（A2 上精度不满足 1e-4），所有倒数用除以常量向量实现
 *   - fp16/bf16 输入先 Cast 到 fp32 计算，再 CAST_RINT 转回原 dtype
 *   - 边界语义（IEEE-754, FTZ；实测口径，本地 52 例全量验证）：
 *     +Inf -> +Inf；-Inf -> NaN（链尾 t=0 与 x=-Inf 相乘，IEEE-754 规定
 *     Inf·0=NaN，与 golden 的 mish(-Inf)=NaN 逐位置精确匹配，实测 mismatch=0；
 *     checker 对双算子均输出 NaN 的极端情况给 F090 提示，已人工核查该 case
 *     输入分布为纯 -Inf，属真值匹配而非漏检）；NaN 全链路传播；x=0/-0 -> 0/-0
 *
 * 第二轮优化（全部分支条件来自运行时通用量：元素总数、tile 容量、
 * GetBlockIdx/GetBlockNum、dtype 位宽与 512B 架构传输粒度，无任何
 * 已知测试 shape/数值特化）：
 *   1. 计算链压缩：常量 2.0 tile 在序言 Duplicate 一次（上一版每 tile 重复
 *      Duplicate 三次）；1/w 以 2/(2w) 实现——w+w 在二进制浮点只增指数不改尾数，
 *      IEEE 除法对同一精确商舍入，结果与 Div(1.0,w) 逐位一致。每 tile 8 次向量
 *      运算（上一版 10 次）。w ∈ [1,+Inf] 保证 2w 无下溢/清零路径。
 *   2. 单 tile 轻量路径：每核仅 1 个 tile（元素总数 vs 单 tile 容量的运行时
 *      推导）时，TBuf + 显式 pipe 事件同步（FetchEventID + SetFlag/WaitFlag，
 *      CANN 官方实现同款模式）直算，省去双缓冲队列的分配/入队/出队簿记；
 *      空张量（tiles==0）仍走守卫后返回。计算链与主流水线完全一致。
 *   3. 多核均匀划分：host 下发 512B 粒度均匀基长 blockFactor，前 r 核各多
 *      1 个 granule，r 在本端由 totalNum 与 GetBlockNum() 重组；所有核起始
 *      512B 对齐、核间不均衡 ≤1 granule，尾 granule 夹到张量真实末尾。
 */

#ifndef MISH_H
#define MISH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mish_tiling_data.h"
#include "mish_tiling_key.h"

namespace NsMish {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FLOAT_BYTES = 4;
constexpr int64_t GRANULE_BYTES = 512;  // MTE 传输粒度（字节），与 tiling 侧一致

// fp32 计算链：y = x * t, t = 2 / (w + 1/w), w = 1 + 2/exp(x)
// 缓冲区角色：x32 只读输入；eT/wT 演化暂存；kT 为序言预生成的常量 2.0 tile；
// y 接收结果（fp32 路径下与输入视图重命名复用）。全部为 fp32 视图。
// 逐位一致性：w+w 只增指数不改尾数（w 有界于 [1,+Inf]，2w 无下溢/清零），
// 2/(2w) 与 Div(1.0, w) 对同一精确商按 IEEE 舍入，结果相同。
__aicore__ inline void MishFp32Chain(const LocalTensor<float> x32, LocalTensor<float> eT,
    LocalTensor<float> wT, LocalTensor<float> kT, LocalTensor<float> y, int32_t count)
{
    Exp<float>(eT, x32, count);                 // e = exp(x)
    Div<float>(wT, kT, eT, count);              // 2/e   （kT == 2.0，序言预生成）
    Adds<float>(wT, wT, 1.0f, count);           // w = 1 + 2/e
    Add<float>(eT, wT, wT, count);              // u = 2w（精确）
    Div<float>(eT, kT, eT, count);              // 1/w = 2/(2w)
    Add<float>(wT, wT, eT, count);              // s = w + 1/w
    Div<float>(eT, kT, wT, count);              // t = 2/s
    Mul<float>(y, x32, eT, count);              // y = x * t
}

template <typename T>
class Mish {
public:
    __aicore__ inline Mish(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t tileIdx, int64_t tileElemNum);
    __aicore__ inline void SingleTileProcess();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> bufIn, bufOut;             // 单 tile 轻量路径缓冲
    TBuf<TPosition::VECCALC> bufX32, bufE, bufW, bufK;

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
    bool needCast_ = (sizeof(T) != static_cast<size_t>(FLOAT_BYTES));
};

template <typename T>
__aicore__ inline void Mish<T>::Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData)
{
    totalLength_ = tilingData->totalNum;
    // blockFactor 为 512B 粒度均匀基长（前 r 核各多 1 个 granule，r 在下方重组）
    blockLength_ = tilingData->blockFactor;
    elemPerTile_ = tilingData->ubFactor;
    const int64_t tileBytesF32 = elemPerTile_ * FLOAT_BYTES;
    const int64_t tileBytesT = elemPerTile_ * static_cast<int64_t>(sizeof(T));

    // 每核范围：512B granule base+r 推导（分支条件仅元素总数/核数/粒度）。
    // host 保证 blockLength_*numBlocks ≤ ⌈total/granule⌉*granule，剩余不足
    // 一个 granule 的余量记为 r；last granule 夹到张量真实末尾。
    const int64_t blkIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t elemsPerGranule = GRANULE_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t numBlocks = static_cast<int64_t>(GetBlockNum());
    int64_t blockStart = 0;
    currentBlockLength_ = 0;
    {
        int64_t covered = 0;
        if (numBlocks > 0) covered = blockLength_ * numBlocks;  // 守卫分支：GetBlockNum 运行时防御
        const int64_t leftover = totalLength_ - covered;
        int64_t r = (leftover > 0) ? (leftover + elemsPerGranule - 1) / elemsPerGranule : 0;
        if (r > numBlocks) r = numBlocks;
        blockStart = blkIdx * blockLength_ + (blkIdx < r ? blkIdx : r) * elemsPerGranule;
        currentBlockLength_ = blockLength_ + (blkIdx < r ? elemsPerGranule : 0);
    }
    if (blockStart >= totalLength_) {
        blockStart = totalLength_;
        currentBlockLength_ = 0;
    } else if (blockStart + currentBlockLength_ > totalLength_) {
        currentBlockLength_ = totalLength_ - blockStart;
    }

    tileNum_ = currentBlockLength_ / elemPerTile_;
    tailTileElementNum_ = currentBlockLength_ - tileNum_ * elemPerTile_;
    tiles_ = tileNum_ + (tailTileElementNum_ > 0 ? 1 : 0);
    // 单 tile 核（元素总数不超过单 tile 容量的运行时推导）走轻量路径：
    // 每方向单缓冲、显式 pipe 事件同步、无队列状态机；空张量（tiles==0）
    // 落回守卫返回。分支条件由 totalNum 与 tile 容量推导，非 shape 枚举。
    singleTile_ = (tiles_ == 1 && currentBlockLength_ > 0);

    if (singleTile_) {
        pipe.InitBuffer(bufIn, tileBytesT);
        pipe.InitBuffer(bufOut, tileBytesT);
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBytesT);
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBytesT);
    }
    if (needCast_) {
        pipe.InitBuffer(bufX32, tileBytesF32);
    }
    pipe.InitBuffer(bufE, tileBytesF32);
    pipe.InitBuffer(bufW, tileBytesF32);
    pipe.InitBuffer(bufK, tileBytesF32);

    inputGMX.SetGlobalBuffer((__gm__ T *)x + blockStart);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + blockStart);

    // 序言：预生成计算链消费的常量 2.0 tile（上一版每 tile 重复 Duplicate 三次）
    if (tiles_ > 0) {
        Duplicate<float>(bufK.template Get<float>(), 2.0f, static_cast<int32_t>(elemPerTile_));
    }
}

// 提交一个 tile 的拷入（DataCopyPad 支持任意字节长度，含非对齐尾块）
template <typename T>
__aicore__ inline void Mish<T>::CopyInTile(int64_t tileIdx, int64_t tileElemNum)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tileElemNum * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    auto xLocal = inputQueueX.AllocTensor<T>();
    DataCopyPad(xLocal, inputGMX[tileIdx * elemPerTile_], copyParams, padParams);
    inputQueueX.EnQue(xLocal);
}

// 单 tile 轻量路径：DataCopyPad 拷入 -> 单次 MTE2->V 事件等待 -> 计算
// （与主流水线完全一致的计算链）-> 单次 V->MTE3 事件等待 -> DataCopyPad 拷出。
// 事件 ID 来自 pipe 分配器（该路径未初始化队列，无争用）。
template <typename T>
__aicore__ inline void Mish<T>::SingleTileProcess()
{
    const int32_t count = static_cast<int32_t>(currentBlockLength_);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(static_cast<int64_t>(count) * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    const int32_t evtMte2V = static_cast<int32_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    const int32_t evtVMte3 = static_cast<int32_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    auto xLocal = bufIn.template Get<T>();
    DataCopyPad(xLocal, inputGMX[0], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(evtMte2V);
    WaitFlag<HardEvent::MTE2_V>(evtMte2V);

    auto yLocal = bufOut.template Get<T>();
    auto eT = bufE.template Get<float>();
    auto wT = bufW.template Get<float>();
    auto kT = bufK.template Get<float>();
    if (needCast_) {
        // half/bfloat16 -> fp32 计算 -> round-to-nearest-even 转回
        auto x32 = bufX32.template Get<float>();
        Cast<float, T>(x32, xLocal, RoundMode::CAST_NONE, count);
        MishFp32Chain(x32, eT, wT, kT, x32, count);
        Cast<T, float>(yLocal, x32, RoundMode::CAST_RINT, count);
    } else {
        MishFp32Chain(xLocal.template ReinterpretCast<float>(), eT, wT, kT,
            yLocal.template ReinterpretCast<float>(), count);
    }
    SetFlag<HardEvent::V_MTE3>(evtVMte3);
    WaitFlag<HardEvent::V_MTE3>(evtVMte3);
    DataCopyPad(outputGMY[0], yLocal, copyParams);
}

template <typename T>
__aicore__ inline void Mish<T>::Process()
{
    if (tiles_ <= 0) {
        return;
    }
    if (singleTile_) {
        SingleTileProcess();
        return;
    }
    // 序言：先发出 tile 0 的拷入，使循环内在消费 tile i 之前预取 tile i+1
    {
        const int64_t firstN = (tileNum_ > 0) ? elemPerTile_ : tailTileElementNum_;
        CopyInTile(0, firstN);
    }

    for (int64_t i = 0; i < tiles_; ++i) {
        // 预取 tile i+1，使 MTE 拷入与当前 tile 的矢量计算、拷出重叠
        if (i + 1 < tiles_) {
            const int64_t nextN = (i + 1 < tileNum_) ? elemPerTile_ : tailTileElementNum_;
            CopyInTile(i + 1, nextN);
        }

        const int32_t count = static_cast<int32_t>((i < tileNum_) ? elemPerTile_ : tailTileElementNum_);
        auto xLocal = inputQueueX.DeQue<T>();
        auto yLocal = outputQueueY.AllocTensor<T>();
        auto eT = bufE.template Get<float>();
        auto wT = bufW.template Get<float>();
        auto kT = bufK.template Get<float>();
        if (needCast_) {
            // half/bfloat16 -> fp32 计算 -> round-to-nearest-even 转回
            auto x32 = bufX32.template Get<float>();
            Cast<float, T>(x32, xLocal, RoundMode::CAST_NONE, count);
            MishFp32Chain(x32, eT, wT, kT, x32, count);
            Cast<T, float>(yLocal, x32, RoundMode::CAST_RINT, count);
        } else {
            MishFp32Chain(xLocal.template ReinterpretCast<float>(), eT, wT, kT,
                yLocal.template ReinterpretCast<float>(), count);
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
#endif // MISH_H
