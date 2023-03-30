//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreluTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate {

void PreluFloatSimpleTest(std::vector <armnn::BackendId>& backends, bool isAlphaConst, bool isDynamicOutput = false)
{
    std::vector<int32_t> inputShape { 1, 2, 3 };
    std::vector<int32_t> alphaShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 3 };

    if (isDynamicOutput)
    {
        outputShape.clear();
    }

    std::vector<float> inputData = { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f };
    std::vector<float> alphaData = { 0.5f };
    std::vector<float> expectedOutput = { -7.f, 2.f, 0.f, 1.f, -2.5f, 14.f };

    PreluTest(tflite::BuiltinOperator_PRELU,
              ::tflite::TensorType_FLOAT32,
              backends,
              inputShape,
              alphaShape,
              outputShape,
              inputData,
              alphaData,
              expectedOutput,
              isAlphaConst);
}

TEST_SUITE("Prelu_CpuRefTests")
{

TEST_CASE ("PreluFp32SimpleConstTest_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PreluFloatSimpleTest(backends, true);
}

TEST_CASE ("PreluFp32SimpleTest_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PreluFloatSimpleTest(backends, false);
}

TEST_CASE ("PreluFp32SimpleConstDynamicTest_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PreluFloatSimpleTest(backends, true, true);
}

TEST_CASE ("PreluFp32SimpleDynamicTest_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PreluFloatSimpleTest(backends, false, true);
}

} // TEST_SUITE("Prelu_CpuRefTests")

TEST_SUITE("Prelu_CpuAccTests")
{

TEST_CASE ("PreluFp32SimpleConstTest_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PreluFloatSimpleTest(backends, true);
}

TEST_CASE ("PreluFp32SimpleTest_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PreluFloatSimpleTest(backends, false);
}

TEST_CASE ("PreluFp32SimpleConstDynamicTest_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PreluFloatSimpleTest(backends, true, true);
}

TEST_CASE ("PreluFp32SimpleDynamicTest_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PreluFloatSimpleTest(backends, false, true);
}

} // TEST_SUITE("Prelu_CpuAccTests")

TEST_SUITE("Prelu_GpuAccTests")
{

TEST_CASE ("PreluFp32SimpleConstTest_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PreluFloatSimpleTest(backends, true);
}

TEST_CASE ("PreluFp32SimpleTest_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PreluFloatSimpleTest(backends, false);
}

TEST_CASE ("PreluFp32SimpleConstDynamicTest_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PreluFloatSimpleTest(backends, true, true);
}

TEST_CASE ("PreluFp32SimpleDynamicTest_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PreluFloatSimpleTest(backends, false, true);
}

} // TEST_SUITE("Prelu_GpuAccTests")

}