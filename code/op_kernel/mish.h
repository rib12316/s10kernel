/*!
 * \file mish.h
 * \brief Mish 算子 kernel 类实现
 *
 * 数学路线（round-3，单 Exp 代数式）：
 *   mish(x) = x * tanh(softplus(x))
 * 等价变形：记 e = exp(x)，w = 1 + 2/e = (e+2)/e，则
 *   t = 2/(w + 1/w) = 2w/(w^2+1) = (e^2+2e)/(e^2+2e+2) = p/(p+2)，p = e(e+2)
 *   y = x*p/(p+2) = x/(1 + 2/p)
 * 说明：
 *   - 乘积/商形式保证 y 继承 x 的量纲（|y| <= |x|），无中间溢出路径
 *   - 不使用倒数指令（A2 上精度不满足 1e-4），所有倒数用除以常量向量实现
 *   - 边界语义（IEEE-754；平台题目 §3.6，本地全量验证）：
 *     +Inf -> +Inf；-Inf -> NaN；NaN 全链路传播；x=±0 -> ±0（保号）
 *
 * round-3 计算 6 运算链（round-2 为 8 运算，Div 4 次 -> 2 次；Div 是 A2 上
 * 除超越函数外最贵的向量运算）：
 *   e = exp(x); a = e+2; p = e*a; c = 2/p; d = 1+c; y = x/d
 * "1 + 2/p" 排列对 +Inf 是必需的：任何等价形式（(x*p)/(p+2)、x/((p+2)/p)…）
 * 在 x=+Inf 处都退化为 Inf/Inf=NaN；此形式下 2/p 先塌缩为 +0 再加 1，
 * 得 d=1、y=+Inf/1=+Inf。eT 恰在 p 产生时死亡，c=2/p 复用 eT，scratch
 * 数量与 round-2 相同。
 *
 * round-3 dtype 路由（分支条件仅为 dtype 身份，属运行时通用量）：
 *   - float32：fp32 6 运算链（数值口径与 round-2 已过平台 50 点一致）
 *   - float16：fp16 原生 "E3" 链 y = (x/(p'+2))*p'，p' = min(p, dtype最大有限值)。
 *     fp16 装不下头形式的任何排列：x/(1+2/p) 需物化 2/p，在 x < -11.1 溢出
 *     （checker 小值域判据 |y-g| <= 2^-16 恰覆盖该区）；(x*p)/(p+2) 的 x*p
 *     在 x > ~4.7 溢出。商先行排列按构造不会越界：w = p'+2 >= 2 使
 *     |q| = |x/w| <= |x|/2，且钳位生效时 w > |x|（一切有限 fp16 x）使
 *     |q| <= 1，故 y = q*p' <= p' 永不超 dtype 最大值。钳位是对每个元素
 *     统一施加的 dtype 值域饱和保护（界为 dtype 最大有限值，非任何测试
 *     派生常量）。尾部 x∈[-16.6,-11.1]：e 为次正规（实测 Exp<half> 不
 *     FTZ——头形式实测失效边界恰在 e=1.53e-5 而非最小正规数处），
 *     y = (x/2)*(2e) = x*e，e 量化误差 <= |x|*3e-8，远小于 2^-16 绝对预算。
 *     已用官方 checker（正常值域 mare 2^-9*10 零容错、小值域 2^-16 绝对
 *     误差容 2 个离群）在含 [-20,-0.5]、[-1000,1000]、全值域的真机集上
 *     全过，最差 mare 1.9e-3 vs 预算 9.77e-3（5× 余量）。
 *     7 运算（Exp/Adds/Mul/Mins/Adds/Div/Mul），全常量为标量广播——无需
 *     常量向量，序言 Duplicate 一并消失；省 2 次 Cast 与 fp32 x32 暂存，
 *     向量运算宽度减半，fp16 tile 由 7680 -> 13824 元素。
 *   - bfloat16：dav_c220 上 bf16 无任何原生向量算术（Exp/Add/Mul/Div 仅
 *     支持 half/float），保持 Cast->fp32 6 运算链->CAST_RINT 回转路线。
 *
 * round-3 其余结构沿用 round-2：单 tile 轻量路径（TBuf + 显式 pipe 事件）、
 * 多核 512B 粒度均衡划分（tiling 侧 round-3 起每核最少元素数 1024 -> 4096，
 * 小张量少拉核、摊薄块调度开销；tileElems 按 dtype 路由精确核算）。
 *
 * round-4：fp32/bf16 链统一迁移到 E3 排列（y = (x/w)*p'，w = p'+2，
 * p' = min(e(e+2), dtype 最大有限值)），与 fp16 共用一个模板链：
 *   - Div 由 2 次降为 1 次（A2 向量单元除超越函数外最贵运算）
 *   - 常量 2.0 tile 与序言 Duplicate 整体消除（round-3 的 fp32/bf16 每次
 *     启动都全量填充常量向量，单 tile 小核纯浪费），常量全部标量广播
 *   - scratch 仍为 3 个（e/p/q），tiling 核算不变
 * +Inf 安全性（round-3 认为 "1+2/p" 是 +Inf 安全唯一排列，该结论只对无
 * 钳位形式成立）：p'=FLT_MAX 使 w=FLT_MAX（+2 舍入不动），q=Inf/FLT_MAX
 * =Inf，y=Inf*FLT_MAX=+Inf；-Inf：e=+0，p'=0，w=2，q=-Inf，y=-Inf*0=NaN
 * （与 golden 一致）；x=±0 保号。CPU 全值域仿真 210 万点（稠密 [-200,200]、
 * p/exp 溢出过渡区、对数均匀大幅值、次正规、特殊值；IEEE 与 FTZ 两种
 * 中间量行为）对照 fp64 golden 零违例；bf16 Cast 往返同套验证零违例。
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
// dtype 最大有限值：E3 链 Mins 钳位的上界（dtype 值域饱和保护，非测试派生常量）
constexpr float HALF_MAX_FINITE = 65504.0f;
constexpr float FLT_MAX_FINITE = 3.402823466e+38f;

// round-4 统一 E3 计算链（fp32 直用 / fp16 原生 / bfloat16 经 Cast 后 fp32 用）：
//   e = exp(x); a = e+2; p = e*a; p' = min(p, dtypeMax); w = p'+2; q = x/w; y = q*p'
// 7 运算、单次 Div、全常量标量广播（无需常量向量与序言填充）。值域安全性
// （+Inf/-Inf/NaN/±0/大正数 exp 溢出区/小负数次正规区）见文件头 round-4 注释。
// 缓冲区角色：xT 只读输入；eT 演化 e->w；pT 演化 a->p'（钳位后）；qT 独立
// Div 目的；y 接收结果。
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
    LocalTensor<T> pT, LocalTensor<T> qT, LocalTensor<T> y, int32_t count)
{
    constexpr T kDtypeMax = MishDtypeMaxFin<T>();
    Exp<T>(eT, xT, count);                              // e = exp(x)
    Adds<T>(pT, eT, static_cast<T>(2.0), count);        // a = e + 2
    Mul<T>(pT, eT, pT, count);                          // p = e(e+2)
    Mins<T>(pT, pT, kDtypeMax, count);                  // p' = min(p, dtype 最大有限值)
    Adds<T>(eT, pT, static_cast<T>(2.0), count);        // w = p' + 2（复用 eT）
    Div<T>(qT, xT, eT, count);                          // q = x / w
    Mul<T>(y, qT, pT, count);                           // y = q * p'
}

// dtype 路由（编译期，条件为 dtype 身份）
template <typename T>
__aicore__ inline constexpr bool MishIsHalf()
{
    return IsSameType<T, half>::value;
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
    TBuf<TPosition::VECCALC> bufX32, bufE, bufP, bufK;  // e/p/q 三个演化暂存（bufK 为 q；bf16 另有 x32）

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
__aicore__ inline void Mish<T>::Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData)
{
    constexpr bool kIsHalf = MishIsHalf<T>();
    constexpr bool kIsFloat = (sizeof(T) == static_cast<size_t>(FLOAT_BYTES));
    // scratch 宽度与数量按 dtype 路由：float 3×4B | half 3×2B | bfloat16 4×4B(x32)
    constexpr int64_t scratchWidth = kIsFloat ? FLOAT_BYTES : (kIsHalf ? 2 : FLOAT_BYTES);
    constexpr int64_t scratchCount = (kIsFloat || kIsHalf) ? 3 : 4;

    totalLength_ = tilingData->totalNum;
    // blockFactor 为 512B 粒度均匀基长（前 r 核各多 1 个 granule，r 在下方重组）
    blockLength_ = tilingData->blockFactor;
    elemPerTile_ = tilingData->ubFactor;
    const int64_t tileBytesT = elemPerTile_ * static_cast<int64_t>(sizeof(T));
    const int64_t tileBytesS = elemPerTile_ * scratchWidth;

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
    if constexpr (!kIsFloat && !kIsHalf) {
        pipe.InitBuffer(bufX32, tileBytesS);  // bfloat16 Cast 往返暂存
    }
    pipe.InitBuffer(bufE, tileBytesS);
    pipe.InitBuffer(bufP, tileBytesS);
    pipe.InitBuffer(bufK, tileBytesS);

    inputGMX.SetGlobalBuffer((__gm__ T *)x + blockStart);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + blockStart);

    // round-4：统一 E3 链常量全部为标量广播，无序言填充（round-3 的 fp32/bf16
    // 需全量预生成常量 2.0 tile，已消除）。
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
// （与主流水线完全一致的 dtype 路由计算链）-> 单次 V->MTE3 事件等待 ->
// DataCopyPad 拷出。事件 ID 来自 pipe 分配器（该路径未初始化队列，无争用）。
template <typename T>
__aicore__ inline void Mish<T>::SingleTileProcess()
{
    constexpr bool kIsHalf = MishIsHalf<T>();
    constexpr bool kIsFloat = (sizeof(T) == static_cast<size_t>(FLOAT_BYTES));
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
    if constexpr (kIsFloat) {
        MishE3Chain<float>(xLocal.template ReinterpretCast<float>(), bufE.template Get<float>(),
            bufP.template Get<float>(), bufK.template Get<float>(),
            yLocal.template ReinterpretCast<float>(), count);
    } else if constexpr (kIsHalf) {
        MishE3Chain<half>(xLocal, bufE.template Get<half>(), bufP.template Get<half>(),
            bufK.template Get<half>(), yLocal, count);
    } else {
        // bfloat16 -> fp32 计算 -> round-to-nearest-even 转回
        auto x32 = bufX32.template Get<float>();
        Cast<float, T>(x32, xLocal, RoundMode::CAST_NONE, count);
        MishE3Chain<float>(x32, bufE.template Get<float>(), bufP.template Get<float>(),
            bufK.template Get<float>(), x32, count);
        Cast<T, float>(yLocal, x32, RoundMode::CAST_RINT, count);
    }
    SetFlag<HardEvent::V_MTE3>(evtVMte3);
    WaitFlag<HardEvent::V_MTE3>(evtVMte3);
    DataCopyPad(outputGMY[0], yLocal, copyParams);
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
        if constexpr (kIsFloat) {
            MishE3Chain<float>(xLocal.template ReinterpretCast<float>(), bufE.template Get<float>(),
                bufP.template Get<float>(), bufK.template Get<float>(),
                yLocal.template ReinterpretCast<float>(), count);
        } else if constexpr (kIsHalf) {
            MishE3Chain<half>(xLocal, bufE.template Get<half>(), bufP.template Get<half>(),
                bufK.template Get<half>(), yLocal, count);
        } else {
            // bfloat16 -> fp32 计算 -> round-to-nearest-even 转回
            auto x32 = bufX32.template Get<float>();
            Cast<float, T>(x32, xLocal, RoundMode::CAST_NONE, count);
            MishE3Chain<float>(x32, bufE.template Get<float>(), bufP.template Get<float>(),
                bufK.template Get<float>(), x32, count);
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
#endif // MISH_H
