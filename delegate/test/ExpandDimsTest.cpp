//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

namespace armnnDelegate
{

void ExpandDimsSimpleTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };
    std::vector<int32_t> axis { 0 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_EXPAND_DIMS,
                        ::tflite::TensorType_FLOAT32,
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        axis);
}

void ExpandDimsWithNegativeAxisTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };
    std::vector<int32_t> axis { -1 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_EXPAND_DIMS,
                        ::tflite::TensorType_FLOAT32,
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        axis);
}

TEST_SUITE("ExpandDims_GpuAccTests")
{

TEST_CASE ("ExpandDims_Simple_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ExpandDimsSimpleTest(backends);
}

TEST_CASE ("ExpandDims_With_Negative_Axis_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ExpandDimsWithNegativeAxisTest(backends);
}

} // TEST_SUITE("ExpandDims_GpuAccTests")

TEST_SUITE("ExpandDims_CpuAccTests")
{

TEST_CASE ("ExpandDims_Simple_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ExpandDimsSimpleTest(backends);
}

TEST_CASE ("ExpandDims_With_Negative_Axis_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ExpandDimsWithNegativeAxisTest(backends);
}

} // TEST_SUITE("ExpandDims_CpuAccTests")

TEST_SUITE("ExpandDims_CpuRefTests")
{

TEST_CASE ("ExpandDims_Simple_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ExpandDimsSimpleTest(backends);
}

TEST_CASE ("ExpandDims_With_Negative_Axis_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ExpandDimsWithNegativeAxisTest(backends);
}

} // TEST_SUITE("ExpandDims_CpuRefTests")

} // namespace armnnDelegate