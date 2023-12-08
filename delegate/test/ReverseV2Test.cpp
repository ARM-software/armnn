//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2TestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ReverseV2Float32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f
    };

    // The output data
    std::vector<float> expectedOutputValues =
    {
        3.0f, 2.0f, 1.0f,
        6.0f, 5.0f, 4.0f,
        9.0f, 8.0f, 7.0f,

        13.0f, 12.0f, 11.0f,
        16.0f, 15.0f, 14.0f,
        19.0f, 18.0f, 17.0f,

        23.0f, 22.0f, 21.0f,
        26.0f, 25.0f, 24.0f,
        29.0f, 28.0f, 27.0f
    };

    // The axis to reverse
    const std::vector<int32_t> axisValues = {2};

    // Shapes
    const std::vector<int32_t> inputShape = {3, 3, 3};
    const std::vector<int32_t> axisShapeDims = {1};
    const std::vector<int32_t> expectedOutputShape = {3, 3, 3};

    ReverseV2FP32TestImpl(tflite::BuiltinOperator_REVERSE_V2,
                          backends,
                          inputValues,
                          inputShape,
                          axisValues,
                          axisShapeDims,
                          expectedOutputValues,
                          expectedOutputShape);
}

void ReverseV2NegativeAxisFloat32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f
    };

    // The output data
    std::vector<float> expectedOutputValues =
    {
        7.0f, 8.0f, 9.0f,
        4.0f, 5.0f, 6.0f,
        1.0f, 2.0f, 3.0f,

        17.0f, 18.0f, 19.0f,
        14.0f, 15.0f, 16.0f,
        11.0f, 12.0f, 13.0f,

        27.0f, 28.0f, 29.0f,
        24.0f, 25.0f, 26.0f,
        21.0f, 22.0f, 23.0f
    };

    // The axis to reverse
    const std::vector<int32_t> axisValues = {-2};

    // Shapes
    const std::vector<int32_t> inputShape = {3, 3, 3};
    const std::vector<int32_t> axisShapeDims = {1};
    const std::vector<int32_t> expectedOutputShape = {3, 3, 3};

    ReverseV2FP32TestImpl(tflite::BuiltinOperator_REVERSE_V2,
                          backends,
                          inputValues,
                          inputShape,
                          axisValues,
                          axisShapeDims,
                          expectedOutputValues,
                          expectedOutputShape);
}

TEST_SUITE("ReverseV2Tests_GpuAccTests")
{

    TEST_CASE ("ReverseV2_Float32_GpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
        ReverseV2Float32Test(backends);
    }

    TEST_CASE ("ReverseV2_NegativeAxis_Float32_GpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
        ReverseV2NegativeAxisFloat32Test(backends);
    }

} // TEST_SUITE("ReverseV2Tests_GpuAccTests")

TEST_SUITE("ReverseV2Tests_CpuAccTests")
{

    TEST_CASE ("ReverseV2_Float32_CpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
        ReverseV2Float32Test(backends);
    }

    TEST_CASE ("ReverseV2_NegativeAxis_Float32_CpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
        ReverseV2NegativeAxisFloat32Test(backends);
    }

} // TEST_SUITE("ReverseV2Tests_CpuAccTests")

TEST_SUITE("ReverseV2Tests_CpuRefTests")
{

    TEST_CASE ("ReverseV2_Float32_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        ReverseV2Float32Test(backends);
    }

    TEST_CASE ("ReverseV2_NegativeAxis_Float32_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        ReverseV2NegativeAxisFloat32Test(backends);
    }

} // TEST_SUITE("ReverseV2Tests_CpuRefTests")

} // namespace armnnDelegate
