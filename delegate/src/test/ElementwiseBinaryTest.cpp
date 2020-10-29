//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("ElementwiseBinaryTest")
{

TEST_CASE ("Add_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    // Set input data
    std::vector<int32_t> input0Shape { 2, 2, 2, 3 };
    std::vector<int32_t> input1Shape { 2, 2, 2, 3 };
    std::vector<int32_t> outputShape { 2, 2, 2, 3 };

    std::vector<float> input0Values =
    {
        0.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        1.0f, 2.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

        0.0f, 2.0f, 1.0f,
        4.2f, 1.0f, 2.0f,

        0.0f, 0.0f, 1.0f,
        0.2f, 1.0f, 2.0f,

    };

    std::vector<float> input1Values =
    {
        1.0f, 2.0f,  1.0f,
        0.0f, 1.0f,  2.0f,

        1.0f, 2.0f, -2.0f,
        0.2f, 1.0f,  2.0f,

        0.0f, 2.0f,  1.0f,
        4.2f, 0.0f, -3.0f,

        0.0f, 0.0f,  1.0f,
        0.7f, 1.0f,  5.0f,
    };

    std::vector<float> expectedOutputValues =
    {
        1.0f, 4.0f,  2.0f,
        0.2f, 2.0f,  4.0f,

        2.0f, 4.0f, -1.0f,
        0.4f, 2.0f,  4.0f,

        0.0f, 4.0f,  2.0f,
        8.4f, 1.0f, -1.0f,

        0.0f, 0.0f,  2.0f,
        0.9f, 2.0f,  7.0f,
    };


    ElementwiseBinaryFP32Test(tflite::BuiltinOperator_ADD,
                              tflite::ActivationFunctionType_NONE,
                              backends,
                              input0Shape,
                              input1Shape,
                              outputShape,
                              input0Values,
                              input1Values,
                              expectedOutputValues);
}

TEST_CASE ("Add_Broadcast_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    // Set input data
    std::vector<int32_t> input0Shape { 1, 3, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 1, 2, 3 };
    std::vector<int32_t> outputShape { 1, 3, 2, 3 };

    std::vector<float> input0Values
    {
        0.0f,
        1.0f,

        2.0f,
        3.0f,

        4.0f,
        5.0f,
    };
    std::vector<float> input1Values
    {
        0.5f, 1.5f, 2.5f,
        3.5f, 4.5f, 5.5f,
    };
    // Set output data
    std::vector<float> expectedOutputValues
    {
        0.5f, 1.5f, 2.5f,
        4.5f, 5.5f, 6.5f,

        2.5f, 3.5f, 4.5f,
        6.5f, 7.5f, 8.5f,

        4.5f, 5.5f, 6.5f,
        8.5f, 9.5f, 10.5f,
    };
    ElementwiseBinaryFP32Test(tflite::BuiltinOperator_ADD,
                              tflite::ActivationFunctionType_NONE,
                              backends,
                              input0Shape,
                              input1Shape,
                              outputShape,
                              input0Values,
                              input1Values,
                              expectedOutputValues);
}

TEST_CASE ("Add_ActivationRELU_Float32_GpuAcc_Test")
{
    // Create the ArmNN Delegate
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc,
                                               armnn::Compute::CpuRef };
    // Set input data
    std::vector<int32_t> input0Shape { 1, 2, 2, 1 };
    std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    std::vector<float> input0Values { 4.0f, 0.8f, 0.7f, -0.8f };
    std::vector<float> input1Values { 0.7f, -1.2f, 0.8f, 0.5f };
    // Set output data
    std::vector<float> expectedOutputValues { 4.7f, 0.0f, 1.5f, 0.0f };
    ElementwiseBinaryFP32Test(tflite::BuiltinOperator_ADD,
                              tflite::ActivationFunctionType_RELU,
                              backends,
                              input0Shape,
                              input1Shape,
                              outputShape,
                              input0Values,
                              input1Values,
                              expectedOutputValues);
}

}

} // namespace armnnDelegate