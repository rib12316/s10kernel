/*!
 * \file test_mish.cpp
 * \brief Mish 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "mish_tiling.h"
#include "../../../op_kernel/mish.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class MishKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MishKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MishKernelTest TearDown" << endl;
    }
};

TEST_F(MishKernelTest, test_kernel_run)
{
    constexpr size_t size = 1;
    constexpr size_t tilingDataSize = sizeof(MishTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 1 * 2;
    constexpr size_t yByteSize = 1 * 2;
    std::vector<uint8_t> xHost(xByteSize);
    std::vector<uint8_t> yHost(yByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/x.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(x), 2); ASSERT_EQ(f.gcount(), 2); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/mish_tiling_data.h 中的字段定义
    //         参考 op_host/mish_tiling.cpp 中的 tiling 计算逻辑
    MishTilingData* tilingData = reinterpret_cast<MishTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/mish_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((mish<0>), numBlocks, x, y, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs((root + "/output/y.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
