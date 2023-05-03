//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

namespace armnnDelegate
{

void SqueezeSimpleTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 2, 2 };
    std::vector<int32_t> squeezeDims { };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_SQUEEZE,
                        ::tflite::TensorType_FLOAT32,
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        squeezeDims);
}

void SqueezeWithDimsTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2 };
    std::vector<int32_t> squeezeDims { -1 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_SQUEEZE,
                        ::tflite::TensorType_FLOAT32,
                        backends,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        squeezeDims);
}

TEST_SUITE("Squeeze_GpuAccTests")
{

TEST_CASE ("Squeeze_Simple_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SqueezeSimpleTest(backends);
}

TEST_CASE ("Squeeze_With_Dims_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SqueezeWithDimsTest(backends);
}

} // TEST_SUITE("Squeeze_GpuAccTests")

TEST_SUITE("Squeeze_CpuAccTests")
{

TEST_CASE ("Squeeze_Simple_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SqueezeSimpleTest(backends);
}

TEST_CASE ("Squeeze_With_Dims_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SqueezeWithDimsTest(backends);
}

} // TEST_SUITE("Squeeze_CpuAccTests")

TEST_SUITE("Squeeze_CpuRefTests")
{

TEST_CASE ("Squeeze_Simple_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SqueezeSimpleTest(backends);
}

TEST_CASE ("Squeeze_With_Dims_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SqueezeWithDimsTest(backends);
}

} // TEST_SUITE("Squeeze_CpuRefTests")

} // namespace armnnDelegate