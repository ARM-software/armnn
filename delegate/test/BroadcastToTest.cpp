//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BroadcastToTestHelper.hpp"

#include <armnn_delegate.hpp>
#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/version.h>
#include <doctest/doctest.h>

namespace armnnDelegate
{
template<typename T>
void BroadcastToTest(std::vector<armnn::BackendId> &backends, tflite::TensorType inputTensorType)
{
    // Set input data
    std::vector<T> inputValues = {
                                      0, 1, 2, 3
                                  };
    // Set output data
    std::vector<T> expectedOutputValues = {
                                               0, 1, 2, 3,
                                               0, 1, 2, 3,
                                               0, 1, 2, 3
                                           };

    // The shape data
    const std::vector<int32_t> shapeData = {3, 4};

    // Set shapes
    const std::vector<int32_t> inputShape = {1, 4};
    const std::vector<int32_t> shapeShape = {2};
    const std::vector<int32_t> expectedOutputShape = {3, 4};

    BroadcastToTestImpl<T>(inputTensorType,
                           tflite::BuiltinOperator_BROADCAST_TO,
                           backends,
                           inputValues,
                           inputShape,
                           shapeShape,
                           shapeData,
                           expectedOutputValues,
                           expectedOutputShape);
}

TEST_SUITE("BroadcastToTests_CpuRefTests")
{

    TEST_CASE ("BroadcastTo_int_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
        BroadcastToTest<int32_t>(backends, ::tflite::TensorType::TensorType_INT32);
    }

    TEST_CASE ("BroadcastTo_Float32_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
        BroadcastToTest<float>(backends, ::tflite::TensorType::TensorType_FLOAT32);
    }

    TEST_CASE ("BroadcastTo_Uint8_t_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
        BroadcastToTest<uint8_t>(backends, ::tflite::TensorType::TensorType_UINT8);
    }

    TEST_CASE ("BroadcastTo_Int8_t_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
        BroadcastToTest<int8_t>(backends, ::tflite::TensorType::TensorType_INT8);
    }

} // TEST_SUITE("BroadcastToTests_CpuRefTests")
}