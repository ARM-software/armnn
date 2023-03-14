//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArgMinMaxTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ArgMaxFP32Test(std::vector<armnn::BackendId>& backends, int axisValue)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 2, 4 };
    std::vector<int32_t> outputShape { 1, 3, 4 };
    std::vector<int32_t> axisShape { 1 };

    std::vector<float> inputValues = { 1.0f,   2.0f,   3.0f,   4.0f,
                                       5.0f,   6.0f,   7.0f,   8.0f,

                                       10.0f,  20.0f,  30.0f,  40.0f,
                                       50.0f,  60.0f,  70.0f,  80.0f,

                                       100.0f, 200.0f, 300.0f, 400.0f,
                                       500.0f, 600.0f, 700.0f, 800.0f };

    std::vector<int32_t> expectedOutputValues = { 1, 1, 1, 1,
                                                  1, 1, 1, 1,
                                                  1, 1, 1, 1 };

    ArgMinMaxTest<float, int32_t>(tflite::BuiltinOperator_ARG_MAX,
                                  ::tflite::TensorType_FLOAT32,
                                  backends,
                                  inputShape,
                                  axisShape,
                                  outputShape,
                                  inputValues,
                                  expectedOutputValues,
                                  axisValue,
                                  ::tflite::TensorType_INT32);
}

void ArgMinFP32Test(std::vector<armnn::BackendId>& backends, int axisValue)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 2, 4 };
    std::vector<int32_t> outputShape { 1, 3, 2 };
    std::vector<int32_t> axisShape { 1 };

    std::vector<float> inputValues = { 1.0f,   2.0f,   3.0f,   4.0f,
                                       5.0f,   6.0f,   7.0f,   8.0f,

                                       10.0f,  20.0f,  30.0f,  40.0f,
                                       50.0f,  60.0f,  70.0f,  80.0f,

                                       100.0f, 200.0f, 300.0f, 400.0f,
                                       500.0f, 600.0f, 700.0f, 800.0f };

    std::vector<int32_t> expectedOutputValues = { 0, 0,
                                                  0, 0,
                                                  0, 0 };

    ArgMinMaxTest<float, int32_t>(tflite::BuiltinOperator_ARG_MIN,
                                  ::tflite::TensorType_FLOAT32,
                                  backends,
                                  inputShape,
                                  axisShape,
                                  outputShape,
                                  inputValues,
                                  expectedOutputValues,
                                  axisValue,
                                  ::tflite::TensorType_INT32);
}

void ArgMaxUint8Test(std::vector<armnn::BackendId>& backends, int axisValue)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 1, 1, 5 };
    std::vector<int32_t> outputShape { 1, 1, 1 };
    std::vector<int32_t> axisShape { 1 };

    std::vector<uint8_t> inputValues = { 5, 2, 8, 10, 9 };

    std::vector<int32_t> expectedOutputValues = { 3 };

    ArgMinMaxTest<uint8_t, int32_t>(tflite::BuiltinOperator_ARG_MAX,
                                    ::tflite::TensorType_UINT8,
                                    backends,
                                    inputShape,
                                    axisShape,
                                    outputShape,
                                    inputValues,
                                    expectedOutputValues,
                                    axisValue,
                                    ::tflite::TensorType_INT32);
}

TEST_SUITE("ArgMinMax_CpuRefTests")
{

TEST_CASE ("ArgMaxFP32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ArgMaxFP32Test(backends, 2);
}

TEST_CASE ("ArgMinFP32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ArgMinFP32Test(backends, 3);
}

TEST_CASE ("ArgMaxUint8Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ArgMaxUint8Test(backends, -1);
}

} // TEST_SUITE("ArgMinMax_CpuRefTests")

TEST_SUITE("ArgMinMax_CpuAccTests")
{

TEST_CASE ("ArgMaxFP32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ArgMaxFP32Test(backends, 2);
}

TEST_CASE ("ArgMinFP32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ArgMinFP32Test(backends, 3);
}

TEST_CASE ("ArgMaxUint8Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ArgMaxUint8Test(backends, -1);
}

} // TEST_SUITE("ArgMinMax_CpuAccTests")

TEST_SUITE("ArgMinMax_GpuAccTests")
{

TEST_CASE ("ArgMaxFP32Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ArgMaxFP32Test(backends, 2);
}

TEST_CASE ("ArgMinFP32Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ArgMinFP32Test(backends, 3);
}

TEST_CASE ("ArgMaxUint8Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ArgMaxUint8Test(backends, -1);
}

} // TEST_SUITE("ArgMinMax_GpuAccTests")

} // namespace armnnDelegate