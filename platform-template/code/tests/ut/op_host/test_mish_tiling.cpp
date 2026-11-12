#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "mish_tiling_data.h"

namespace MishUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "Mish";

struct MishTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/mish_tiling_key.h 和 op_host/mish_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/mish_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/mish_tiling.cpp 中 GetWorkspaceSize 的逻辑
static MishTestParam testCases[] = {
    {"mish_0", {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class MishTilingTest : public testing::TestWithParam<MishTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MishTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MishTilingTest TearDown." << std::endl;
    }
};

struct MishCompileInfo {} compileInfo;

static void TestOneParamCase(const MishTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(MishTilingTest, tiling_test)
{
    const MishTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    MishTilingTests,
    MishTilingTest,
    testing::ValuesIn(testCases));

}
