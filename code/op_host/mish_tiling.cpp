/*!
 * \file mish_tiling.cpp
 * \brief Mish 算子 Tiling 实现
 *
 * 全部参数由运行时通用量（UB 容量、AIV 核数、dtype 位宽）推导，
 * 不包含任何针对特定 shape 的特化分支。
 *
 * 第二轮优化同步（与本地最终版 mish_kernel.cpp 逐参数一致）：
 *   1. tileElems 由"UB×9/10 保守预算"改为精确逐缓冲核算：2×2 队列 tile（dtypeWidth）
 *      + fp32 暂存 tile（fp32 路径 3 个、fp16/bf16 路径 4 个）。窄 dtype 在精确值上
 *      再减 512 元素（约 12KB）余量：其精确值恰好取满 100.0% UB，而 UB 系统区边界
 *      无公开文档，留头空间防越界；分支条件仅为 dtype 位宽（运行时通用量）。
 *   2. 多核划分由 ceil 均分改为 512B 粒度均匀划分：blockFactor 字段承载均匀基长
 *      ⌊⌈total/g⌉/numBlocks⌋×g（g=512B 的 dtype 元素数），前 r 核各多 1 个 granule
 *      （r 在 kernel 端由 totalNum 与 GetBlockNum() 重组）。所有核起始 512B 对齐、
 *      核间不均衡 ≤1 granule；MishTilingData 三字段结构不变。
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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;   // 每核最少元素数，小张量不拆核
constexpr int64_t PIPELINE_DEPTH = 2;           // 队列深度（与 kernel BUFFER_NUM 一致）
constexpr int64_t SCRATCH_TENSORS = 4;          // fp32 暂存上限（x32/e/w/常量），fp32 路径用 3 个
constexpr int64_t GRANULE_BYTES = 512;          // MTE 传输粒度（字节），对齐与划分共用

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

    // 每 tile 元素数：精确逐缓冲核算（替代上一版 9/10 保守预算）。
    // fp32 路径需要 3 个 fp32 暂存（e/w/k）；fp16/bf16 路径额外需要 x32 转换缓冲（共 4 个）。
    // 队列侧每元素占 PIPELINE_DEPTH * 2（入/出）* dtypeWidth 字节。
    const int64_t scratchTensors = (dtypeWidth == TYPE_SIZE) ? (SCRATCH_TENSORS - 1) : SCRATCH_TENSORS;
    const int64_t bytesPerElement = scratchTensors * TYPE_SIZE + PIPELINE_DEPTH * 2 * dtypeWidth;
    int64_t tileElems = static_cast<int64_t>(ubSize) / bytesPerElement;
    if (dtypeWidth != TYPE_SIZE) {
        tileElems -= 512;  // 窄 dtype 精确值取满 100% UB，减 512 元素（~12KB）留系统区余量
    }
    // 向下对齐到 512B 传输粒度（MTE 效率），且不低于一个 512B 块
    const int64_t elemsPer512B = GRANULE_BYTES / dtypeWidth;
    tileElems = tileElems / elemsPer512B * elemsPer512B;
    if (tileElems < elemsPer512B) {
        tileElems = elemsPer512B;
    }

    // 核数：小张量按每核 MIN_SPLIT_THRESHOLD 元素封顶，避免小输入也拉满全核
    int64_t numBlocks = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (numBlocks > coreNum) {
        numBlocks = coreNum;
    }
    if (numBlocks <= 0) {
        numBlocks = 1;   // totalNum == 0 仍下发 1 个空转核（守卫分支，运行时防御）
    }
    // 512B 粒度均匀划分：blockFactor 字段承载均匀基长；前 r 核各多 1 个 granule，
    // r 由 kernel 端从 totalNum 与 GetBlockNum() 重组（上一版 ceil 划分的核起始仅按
    // dtype 宽度对齐、尾核长度非 512B 倍数；均匀划分保证全部核起始 512B 对齐、
    // 核间差 ≤1 granule）。
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
