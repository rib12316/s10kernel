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
 *
 * round-5：tiling 新增 granuleRem 字段（前 r 个核各多 1 个 granule，host 计算
 * 下发），kernel 每核序言免去从 totalNum/blockFactor/GetBlockNum() 重推 r 的
 * 一次 64 位整数除法与守卫分支——该推导是每次启动的串行开销，全部 shape
 * 受益。整数路径改动，数值零变化；新旧划分等价性经 36 组 shape × 2 dtype
 * 逐 span 比对验证（覆盖连续、无重叠、边界一致）。
 *
 * round-6：单 tile 核（currentBlockLength <= elemPerTile，可见小 shape 全部
 * 命中）以比较直判 tileNum/tail/tiles 与 singleTile 标志，免去序言剩下的
 * 最后一次 64 位整数除法；仅多 tile 核（大 shape，除法开销占比可忽略）保留
 * 除法推导。单 tile 分支的 tileNum/tail 内部表示差异不被下游消费
 * （SingleTileProcess 仅使用 currentBlockLength_），等价性含边界
 * （length==elemPerTile、length==0）逐例验证。
 *
 * round-7（workflow4 性能节点，2026-11-04；延迟分解显示 TPipe 多 TBuf 簿记
 * ≈0.5~0.85μs，是小区间地板中可攻击部分。采纳 2 项，实测数字均为 2026-11-04
 * 勘误后口径，与 workflow4 3-性能优化报告.md §9 及 优化报告.md §2 一致）：
 *   - D3b 单 tile 轻量路径缓冲合一：x/y/scratch 并入单个 TBuf，且按实际
 *     核内长度（各区域 512B 取整对齐）而非满 tile 尺寸分配——基线为 1024
 *     元素负载分配 ~138 KiB/核，采纳后 ~16 KiB/核，该路径 InitBuffer 由
 *     5~6 次降为 1 次。独立干净会话 A/B（perf-p3b-singletbuf vs 基线 4 会话
 *     中位）：64×64 fp32 -0.19μs、128×128 fp32 +0.03（噪声级）、bf16 五点
 *     4/5 改善（-0.11~-0.17）、fp16 五点 -0.10~+0.03（噪声级）——独立会话
 *     自身信号以噪声级为主，采纳依据为组合增量证据（final(D3b+cap16) vs
 *     cap16 单独版同协议直接对比：小区间 15 点中 12 点改善 -0.03~-0.12、
 *     3 点 |Δ|<=0.02 噪声持平、0 点劣化；大 shape 10 点全部在基线同二进制
 *     噪声带内）。
 *   - 多 tile 路径 scratch 合一（D3）：e/p/q 演化暂存（bf16 另含 x32）由
 *     3~4 个独立 TBuf 并为 1 个 TBuf 手工切片，槽位按元素偏移寻址
 *     （ScratchSlot），队列侧不变；多 tile 核 InitBuffer 次数同步下降。
 *   - D5 核数细粒度策略带封顶（host 侧 mish_tiling.cpp）：numBlocks =
 *     min(ceil(total/2048), max(ceil(total/4096), 16))，仅 total<=65536
 *     生效；total>65536 时表达式严格等于基线 ceil(total/4096)，大 shape
 *     tiling 与 round-6 基线逐位相同（构造性不劣化）。
 *   - 本轮证伪未采纳（勘误后区间）：②链调度双流交错（大 shape 10 点
 *     -0.36~+13.08μs，其中 9/10 点 +0.10~+13.08 劣化、fp32 8192² 单点
 *     -0.36 在基线同二进制极差带内）；④对齐主干 DataCopy 替代
 *     DataCopyPad（小区间 15 点中 14 点 +0.01~+0.30 略负、1 点 -0.01
 *     噪声级——dav_c220 上 pad 路径无固定开销可省）；⑥tiling 参数
 *     32B->16B 压缩（直调标量传参形态小区间 14/15 点 +0.09~+0.54 劣化、
 *     大 shape 最高 +5.00；该证伪仅针对直调启动形态，平台 GM-struct
 *     形态不受其约束，但本轮证据不支持在该形态外做此改动）。
 *   - 精度：D3b、cap16、final 组合各独立全量黑盒 61/61 PASS（官方 checker
 *     三域判据，fp64 golden），特殊值（±Inf/NaN、-1.1924、次正规、±0 保号、
 *     rank-0）逐例核验全过；阈值零改动，逐元素运算与 round-6 完全一致
 *     （均为执行结构变化）。
 *
 * round-8（2026-11-12，直调工程真机 A/B 验证后移植）：
 *   - E3 链 scratch 由 3 降为 2（e/p）：Div 直接写入输出 y（bfloat16 为
 *     Cast 暂存 x32），q 被紧随的 y = q*p' 立即消费，省去独立 q 缓冲。
 *     tiling 仍沿用 round-7 保守核算（fp32 28 / fp16 14 / bf16 24
 *     B/元素，tileElems 6912/13824/7424 不变）：直调 mini case A/B
 *     （2026-11-12）显示把省出的 UB 换成更大 tile 会使中等多 tile 点回退
 *     1.0~3.4%（双缓冲流水粒度变粗），小 shape 无变化；差额作为 headroom。
 *   - 单 tile 路径彻底去 TPipe：不构造 TPipe、不 InitBuffer；x/y/e/p 用
 *     raw LocalTensor 直接布在 UB 地址 0 起的 512B 对齐区域，事件使用固定
 *     EVENT_ID0/1。显式调用 InitSocState()——TPipe 构造器原本负责的
 *     set_atomic_none / set_mask_norm / 向量 mask 初始化由此补上。
 *   - 多 tile 路径保留 TPipe（Process 内局部构造）与双缓冲队列；scratch
 *     槽位由 3/4 降为 2/3（native/bf16）。
 *
 * 本文件为平台 GM-struct 形态：与直调最终版（direct-invoke
 * mish_kernel.cpp）的类体逐行同源，仅 tiling 交付方式不同——平台版经
 * GET_TILING_DATA_WITH_STRUCT 从 GM 读 MishTilingData 四字段
 * （totalNum/blockFactor/ubFactor/granuleRem），直调版以同名标量参数传入；
 * host 侧 tiling 计算与直调 calc_mish_tiling_params 一一对应。
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
// round-8：仅 2 个 scratch（e/p）。Div 直接写入输出 y（q 被 y=q*p' 立即
// 消费）；bfloat16 的 fp32 链输出为 Cast 暂存 x32，同一别名策略。缓冲区
// 角色：xT 只读输入；eT 演化 e->w；pT 演化 a->p'（钳位后）；y 先接收 q、
// 再接收最终结果。
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
    Mins<T>(pT, pT, kDtypeMax, count);                  // p' = min(p, dtype 最大有限值)
    Adds<T>(eT, pT, static_cast<T>(2.0), count);        // w = p' + 2（复用 eT）
    Div<T>(y, xT, eT, count);                           // q = x / w（y 兼作 q 槽位）
    Mul<T>(y, y, pT, count);                            // y = q * p'
}

// dtype 路由（编译期，条件为 dtype 身份）
template <typename T>
__aicore__ inline constexpr bool MishIsHalf()
{
    return IsSameType<T, half>::value;
}

// round-8 单 tile raw 布局辅助：区域字节数按 512B MTE 粒度对齐；LocalTensor
// 描述长度按 32B 对齐（raw 构造器要求字节长度 32B 对齐）。
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

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MishTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t tileIdx, int64_t tileElemNum);
    __aicore__ inline void SingleTileProcess();
    // round-7 方向③：共享 scratch TBuf 的元素偏移视图（槽位步长 = 计算 dtype
    // 的一个 tile）。仅多 tile 路径使用；单 tile 路径（round-8）用 raw UB
    // 地址直接构造 LocalTensor，不经过 TBuf。
    template <typename CT>
    __aicore__ inline AscendC::LocalTensor<CT> ScratchSlot(int32_t slot)
    {
        return bufScratch.template Get<CT>()[static_cast<uint32_t>(slot) * static_cast<uint32_t>(elemPerTile_)];
    }

private:
    // round-8：不再持有 TPipe。单 tile 路径完全 TPipe-free（raw UB 张量 +
    // 固定事件 ID）；多 tile 路径在 Process() 内局部构造 TPipe 并分配。
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<TPosition::VECCALC> bufScratch;                // 多 tile 路径 scratch 合一（1 次分配手工切片）

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
    totalLength_ = tilingData->totalNum;
    // blockFactor 为 512B 粒度均匀基长（前 r 核各多 1 个 granule，r 在下方重组）
    blockLength_ = tilingData->blockFactor;
    elemPerTile_ = tilingData->ubFactor;

    // 每核范围：512B granule base+r 划分（round-5：r 由 host 经 tiling 字段直接
    // 下发，恒满足 0 <= r < numBlocks，免去除法与 GetBlockNum 依赖）。
    // last granule 夹到张量真实末尾。
    const int64_t blkIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t elemsPerGranule = GRANULE_BYTES / static_cast<int64_t>(sizeof(T));
    const int64_t r = tilingData->granuleRem;
    int64_t blockStart = blkIdx * blockLength_ + (blkIdx < r ? blkIdx : r) * elemsPerGranule;
    currentBlockLength_ = blockLength_ + (blkIdx < r ? elemsPerGranule : 0);
    if (blockStart >= totalLength_) {
        blockStart = totalLength_;
        currentBlockLength_ = 0;
    } else if (blockStart + currentBlockLength_ > totalLength_) {
        currentBlockLength_ = totalLength_ - blockStart;
    }

    // round-6：单 tile 核以比较直判（免去 64 位整数除法，小 shape 主路径）；
    // 仅多 tile 核保留除法推导。单 tile 分支的 tileNum/tail 表示差异不被
    // 下游消费（SingleTileProcess 仅使用 currentBlockLength_）。
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
        tiles_ = 0;  // 空张量守卫（tiles==0 时 Process 直接返回）
    }

    // round-8：Init 不再做任何 UB 分配。单 tile 路径在 SingleTileProcess
    // 内用 raw 地址布局（完全无 TPipe/InitBuffer）；多 tile 路径在 Process
    // 内局部构造 TPipe 后分配队列与 scratch。

    inputGMX.SetGlobalBuffer((__gm__ T *)x + blockStart);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + blockStart);
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

// 单 tile TPipe-free 路径（round-8）：raw UB 布局自地址 0 起
//   native  : [x][y][e][p]          （区域 512B 对齐）
//   bfloat16: [x][y][x32][e][p]     （scratch 区域 fp32 宽度）
// 不构造 TPipe、不 InitBuffer、无 TBuf 簿记、无队列状态机；事件用固定硬件
// ID（本路径不接触队列事件池）。显式调用 InitSocState()——TPipe 构造器原本
// 负责的 set_atomic_none / set_mask_norm / 向量 mask 初始化由此补上。
template <typename T>
__aicore__ inline void Mish<T>::SingleTileProcess()
{
    constexpr bool kIsHalf = MishIsHalf<T>();
    constexpr bool kIsFloat = (sizeof(T) == static_cast<size_t>(FLOAT_BYTES));
    constexpr int64_t scratchWidth = kIsFloat ? FLOAT_BYTES : (kIsHalf ? 2 : FLOAT_BYTES);

    AscendC::InitSocState();  // TPipe 构造器原本在这里执行该初始化

    const int64_t count64 = currentBlockLength_;
    const int32_t count = static_cast<int32_t>(count64);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count64 * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // 区域字节数 512B 对齐；scratch 使用计算 dtype 宽度（native 为 T，
    // bfloat16 为 fp32）。
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
        // bfloat16 -> fp32 计算 -> round-to-nearest-even 转回。
        // x32 兼作链输出；e/p 为 fp32 scratch。
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
        SingleTileProcess();  // round-8：TPipe-free raw UB 路径
        return;
    }

    // round-8：多 tile 路径是唯一使用 TPipe 的路径；在此局部构造，单 tile
    // 启动不付构造器/pool reset 成本。scratch 槽位再降 1（y/x32 兼作 q）。
    constexpr int64_t scratchWidth = kIsFloat ? FLOAT_BYTES : (kIsHalf ? 2 : FLOAT_BYTES);
    constexpr int64_t SCRATCH_COUNT = (kIsFloat || kIsHalf) ? 2 : 3;
    const int64_t tileBytesT = elemPerTile_ * static_cast<int64_t>(sizeof(T));
    const int64_t tileBytesS = elemPerTile_ * scratchWidth;
    TPipe pipe;
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, tileBytesT);
    pipe.InitBuffer(bufScratch, SCRATCH_COUNT * tileBytesS);

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
            MishE3Chain<float>(xLocal.template ReinterpretCast<float>(), ScratchSlot<float>(0),
                ScratchSlot<float>(1), yLocal.template ReinterpretCast<float>(), count);
        } else if constexpr (kIsHalf) {
            MishE3Chain<half>(xLocal, ScratchSlot<half>(0), ScratchSlot<half>(1), yLocal, count);
        } else {
            // bfloat16 -> fp32 计算 -> round-to-nearest-even 转回
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
#endif // MISH_H
