/*!
 * \file mish.cpp
 * \brief Mish 算子 kernel 入口
 */

#include "mish.h"

enum class MishTilingKey : uint32_t
{
    TILING_KEY_MISH_MODE_0 = 0,
    TILING_KEY_MISH_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void mish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MishTilingData);
    GET_TILING_DATA_WITH_STRUCT(MishTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(MishTilingKey::TILING_KEY_MISH_MODE_0)) {
        NsMish::Mish<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(MishTilingKey::TILING_KEY_MISH_MODE_1)) {
        NsMish::Mish<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
