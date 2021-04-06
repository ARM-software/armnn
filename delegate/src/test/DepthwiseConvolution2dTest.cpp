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
                           {1.0f}, // biasScale
                           {0},    // biasOffset
                           {1.0f}, // filterScale
                           {0},    // filterOffsets
                           2.0f,   // outputQuantScale
                           0,      // outputQuantOffset
                           1.0f,   // quantScale
                           0,      // quantOffset
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

void DepthwiseConv2dSameInt8PerChannelTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 4, 4, 4 };
    std::vector<int32_t> filterShape { 1, 2, 2, 16 };
    std::vector<int32_t> biasShape {16} ;
    std::vector<int32_t> outputShape { 1, 4, 4, 16 };

    static std::vector<int8_t> inputValues =
        {
            3,3,3,4, 4,4,0,0, 0,3,4,3, 0,2,2,3,
            3,0,3,0, 0,3,2,1, 4,1,2,2, 0,0,0,4,
            3,2,2,2, 2,1,0,4, 4,3,2,4, 3,2,0,0,
            4,1,4,4, 1,0,4,3, 3,2,0,3, 1,1,0,2
        };

    std::vector<int8_t> filterValues = { 12,20,10, 3, 2,24, 9,10, 5,16,30,12, 3,10, 4,32,
                                           8, 0,30, 3, 0,16,12,15,20,12, 0, 3, 9,20, 8, 8,
                                          12,15,20, 0, 0, 0, 3,15,15, 8,40,12, 9, 5, 2,24,
                                           4, 0, 0, 6, 6, 0, 3, 5,20, 8,20, 3, 6,15, 4, 0 };
    std::vector<float> filterScales = {         0.25,   0.2,        0.1, 0.3333333333,
                                                 0.5, 0.125, 0.33333333,          0.2,
                                                 0.2,  0.25,        0.1,  0.333333333,
                                        0.3333333333,   0.2,        0.5,        0.125 };

    int32_t filterQuantizationDim = 3;

    int32_t depth_multiplier = 4;

    std::vector<int32_t> biasValues = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    float inputScale = 1.0f;
    std::vector<float> biasScales {};
    std::vector<int64_t> biasOffsets {};
    std::vector<int64_t> filterOffsets {};
    for (const auto& filterScale: filterScales)
    {
        biasScales.push_back(inputScale * filterScale);
        // filter and bias offset always needs to be zero for per channel. We don't support anything else
        biasOffsets.push_back(0);
        filterOffsets.push_back(0);
    }

    std::vector<int8_t> expectedOutputValues =
        {
            26,21,21, 7,12,17,28,21,20,22,25,26, 6,11,10,16,
            16,16, 4,12, 7,18,28,27,30,20,12,14,16,19,17, 6,
            12,12, 8, 0, 3,13,18,15,18,26,20,26,26,32,28,21,
            0, 0, 0, 0, 2, 6, 6, 4, 2, 8, 6, 8,15,10,10,24,
            20,21, 9, 7, 3, 6,15,16,17,22,17,22,17,18,14, 7,
            18, 6,16,12,12,11,17,15,18,18,10,12,27,26,22,18,
            27,28,12,10, 7, 3, 8,13, 8,12,14,16,26,24,24,24,
            9, 9, 6, 0, 0, 0, 2, 6, 0, 0, 0, 0, 4, 8, 8,16,
            26,24,17, 7, 2, 8,11,10,30,24,30,28,32,33,30,24,
            20,11,16,12, 7, 9,17,13,20,14,16,18,31,36,33,29,
            28,25,19, 9, 6,13,20,19, 2, 8, 6, 8,17,17,15,25,
            12,15, 5, 3, 2, 6, 7, 7, 0, 0, 0, 0, 6, 2, 2, 6,
            14,16, 7, 5, 1, 3, 3, 2,20,28,12,20,13,20,20,19,
            9, 4,10, 4, 0, 4, 8, 6, 4,16,12,16,12,18,18,15,
            11,12, 6, 4, 2, 8,10, 7, 0, 0, 0, 0, 9,14,14,14,
            3, 4, 1, 1, 1, 3, 3, 2, 0, 0, 0, 0, 2, 4, 4, 8
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<int8_t, int32_t>(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                                      ::tflite::TensorType_INT8,
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
                                      biasValues,
                                      biasScales,
                                      biasOffsets,
                                      filterScales,
                                      filterOffsets,
                                      1.0f,
                                      0,
                                      inputScale,
                                      0,
                                      depth_multiplier,
                                      filterQuantizationDim);
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

TEST_CASE ("DepthwiseConv2d_Same_Int8_PerChannelQuantization_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    DepthwiseConv2dSameInt8PerChannelTest(backends);
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