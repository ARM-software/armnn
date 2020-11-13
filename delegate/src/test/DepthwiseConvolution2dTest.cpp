//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvolutionTestHelper.hpp"

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

void DepthwiseConv2dValidReluFp32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 2, 2 };
    std::vector<int32_t> filterShape { 1, 2, 2, 4 };
    std::vector<int32_t> biasShape { 4 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    static std::vector<float> inputValues =
        {
            1, 2,  7,  8,
            3, 4,  9, 10,
            5, 6, 11, 12
        };

    std::vector<float> filterValues =
        {
            1,    2,   3,   4,
           -9,   10, -11,  12,
            5,    6,   7,   8,
            13,  -14,  15, -16
        };

    std::vector<float> biasValues = { 1, 2, 3, 4 };

    std::vector<float> expectedOutputValues =
        {
            71, 0,  99, 0,
            91, 0, 127, 0
        };

    tflite::Padding padding = tflite::Padding_VALID;
    int32_t depth_multiplier = 2;

    ConvolutionTest<float>(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                           ::tflite::TensorType_FLOAT32,
                           1, // strideX
                           1, // strideY
                           1, // dilationX
                           1, // dilationY
                           padding,
                           tflite::ActivationFunctionType_RELU,
                           backends,
                           inputShape,
                           filterShape,
                           outputShape,
                           inputValues,
                           filterValues,
                           expectedOutputValues,
                           biasShape,
                           biasValues,
                           1.0f, // filterScale
                           0,    // filterOffset
                           2.0f, // outputQuantScale
                           0,    // outputQuantOffset
                           1.0f, // quantScale
                           0,    // quantOffset
                           depth_multiplier);
}

void DepthwiseConv2dSameUint8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 3, 3, 1 };
    std::vector<int32_t> filterShape { 1, 3, 3, 1 };
    std::vector<int32_t> biasShape { 1 } ;
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    static std::vector<uint8_t> inputValues =
        {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8
        };

    std::vector<uint8_t> filterValues = { 9, 8, 7,  6, 5, 4,  3, 2, 1 };

    std::vector<int32_t> biasValues = { 10 };

    std::vector<uint8_t> expectedOutputValues =
        {
            12,  23, 24, // ( 14+10)/2, ( 35+10)/2, ( 38+10)/2,
            34,  65, 61, // ( 57+10)/2, (120+10)/2, (111+10)/2,
            60, 104, 84  // (110+10)/2, (197+10)/2, (158+10)/2
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<uint8_t, int32_t>(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                      ::tflite::TensorType_UINT8,
                                      1, // strideX
                                      1, // strideY
                                      1, // dilationX
                                      1, // dilationY
                                      padding,
                                      tflite::ActivationFunctionType_NONE,
                                      backends,
                                      inputShape,
                                      filterShape,
                                      outputShape,
                                      inputValues,
                                      filterValues,
                                      expectedOutputValues,
                                      biasShape,
                                      biasValues);
}

TEST_SUITE("DepthwiseConv2d_CpuRef_Tests")
{

TEST_CASE ("DepthwiseConv2d_Valid_Relu_Fp32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    DepthwiseConv2dValidReluFp32Test(backends);
}

TEST_CASE ("DepthwiseConv2d_Same_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    DepthwiseConv2dSameUint8Test(backends);
}

}//End of TEST_SUITE("DepthwiseConv2d_CpuRef_Tests")

TEST_SUITE("DepthwiseConv2d_CpuAcc_Tests")
{

TEST_CASE ("DepthwiseConv2d_Valid_Relu_Fp32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    DepthwiseConv2dValidReluFp32Test(backends);
}

TEST_CASE ("DepthwiseConv2d_Same_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    DepthwiseConv2dSameUint8Test(backends);
}

}//End of TEST_SUITE("DepthwiseConv2d_CpuAcc_Tests")

TEST_SUITE("DepthwiseConv2d_GpuAcc_Tests")
{

TEST_CASE ("DepthwiseConv2d_Valid_Relu_Fp32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    DepthwiseConv2dValidReluFp32Test(backends);
}

TEST_CASE ("DepthwiseConv2d_Same_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    DepthwiseConv2dSameUint8Test(backends);
}

}//End of TEST_SUITE("DepthwiseConv2d_GpuAcc_Tests")

} // namespace armnnDelegate