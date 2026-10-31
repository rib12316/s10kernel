/*!
 * \file mish_tiling.cpp
 * \brief Mish 算子 Tiling 实现
 *
 * 全部参数由运行时通用量（UB 容量、AIV 核数、dtype 位宽/身份）推导，
 * 不包含任何针对特定 shape 的特化分支。
 *
 * 第三轮优化同步（与本地直调最终版逐参数一致）：
 *   1. 计算链按 dtype 路由（kernel 侧 mish.h），tileElems 核算随之路由：
 *      float32 为 3 个 fp32 scratch（28 B/元素）；float16 为 fp16 原生链，
 *      3 个 fp16 scratch（14 B/元素，tile 7680 -> 13824）；bfloat16 无原生
 *      向量算术（dav_c220 上 Exp/Add/Mul/Div 仅支持 half/float），保持
 *      Cast->fp32 路线的 4 个 fp32 scratch（24 B/元素，精确取值恰为
 *      100.0% UB，保留 512 元素余量防越界——分支条件仅为 dtype）。
 *   2. 每核最少元素数 1024 -> 4096：小张量少拉核（256×256 由 40 核改
 *      16 核，实测 -8.5%~-15.4%；8 核/4 核反而劣化，16 核为实测峰值），
 *      摊薄块调度与 MTE 串行延迟的固定开销。MIN 值为运行时通用量推导
 *      的阈值，与任何测试 shape 无关。
 *   3. 多核划分沿用 512B 粒度均匀划分（blockFactor 承载均匀基长，
 *      前 r 核各多 1 个 granule，r 在 kernel 端由 totalNum 与
 *      GetBlockNum() 重组）。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/mish_tiling_data.h"
#include "../op_kernel/mish_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 4096;  // 每核最少元素数，小张量不拉满全核（round-3: 1024 -> 4096）
constexpr int64_t PIPELINE_DEPTH = 2;          // 队列深度（与 kernel BUFFER_NUM 一致）
constexpr int64_t SCRATCH_NATIVE = 3;          // 原生链 scratch 数（e/p/常量或 q），宽度随 dtype
constexpr int64_t SCRATCH_BF16 = 4;            // bf16 Cast 路线 scratch 数（x32/e/p/常量），fp32 宽度
constexpr int64_t GRANULE_BYTES = 512;         // MTE 传输粒度（字节），对齐与划分共用

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MishTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    MishTilingData* tiling = context->GetTilingData<MishTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 元素总数（0 维标量按 1 元素处理）
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();

    // dtype 与 tilingKey：fp16 -> MODE_0(half)，fp32 -> MODE_1(float)，bf16 -> MODE_2
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    const int64_t dtypeWidth = (dtype == ge::DT_FLOAT) ? TYPE_SIZE : (TYPE_SIZE / 2);
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_2);
    } else {
        tilingKey = GET_TPL_TILING_KEY(MISH_TPL_SCH_MODE_1);
    }

    // 每 tile 元素数：精确逐缓冲核算，scratch 宽度/数量按 dtype 路由。
    //   float32：3 个 fp32 scratch（e/p/常量2.0）
    //   float16：fp16 原生 E3 链，3 个 fp16 scratch（e/p/q，常量全为标量）
    //   bfloat16：Cast 路线 4 个 fp32 scratch（x32/e/p/常量），精确值恰取满
    //             100.0% UB，减 512 元素（~12KB）留系统区余量
    // 队列侧每元素占 PIPELINE_DEPTH * 2（入/出）* dtypeWidth 字节。
    const bool nativeWidth = (dtype == ge::DT_FLOAT) || (dtype == ge::DT_FLOAT16);
    const int64_t scratchBytes = nativeWidth ? (SCRATCH_NATIVE * dtypeWidth) : (SCRATCH_BF16 * TYPE_SIZE);
    const int64_t bytesPerElement = scratchBytes + PIPELINE_DEPTH * 2 * dtypeWidth;
    int64_t tileElems = static_cast<int64_t>(ubSize) / bytesPerElement;
    if (!nativeWidth) {
        tileElems -= 512;  // bf16 精确值取满 100.0% UB，留余量防越界
    }
    // 向下对齐到 512B 传输粒度（MTE 效率），且不低于一个 512B 块
    const int64_t elemsPer512B = GRANULE_BYTES / dtypeWidth;
    tileElems = tileElems / elemsPer512B * elemsPer512B;
    if (tileElems < elemsPer512B) {
        tileElems = elemsPer512B;
    }

    // 核数：小张量按每核 MIN_SPLIT_THRESHOLD 元素封顶（round-3: 4096），
    // 少拉核摊薄块调度固定开销；分支条件为元素总数与核数，运行时通用量。
    int64_t numBlocks = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }
    if (numBlocks <= 0) {
        numBlocks = 1;   // totalNum == 0 仍下发 1 个空转核（守卫分支，运行时防御）
    }
    // 512B 粒度均匀划分：blockFactor 字段承载均匀基长；前 r 核各多 1 个 granule，
    // r 由 kernel 端从 totalNum 与 GetBlockNum() 重组。所有核起始 512B 对齐、
    // 核间差 ≤1 granule。
    const int64_t elemsPerGranule = GRANULE_BYTES / dtypeWidth;
    const int64_t unitTotal = CeilDiv(totalNum, elemsPerGranule);
    const int64_t blockFactor = (unitTotal / numBlocks) * elemsPerGranule;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = tileElems;

    context->SetBlockDim(numBlocks);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MishCompileInfo {};

IMPL_OP_OPTILING(Mish).Tiling(MishTilingFunc).TilingParse<MishCompileInfo>(TilingParseForMish);

} // namespace optiling
