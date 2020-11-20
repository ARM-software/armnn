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

void Conv2DWithBiasesFp32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 5, 5, 1 };
    std::vector<int32_t> filterShape { 1, 3, 3, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    static std::vector<float> inputValues =
        {
            1, 5, 2, 3, 5,
            8, 7, 3, 6, 3,
            3, 3, 9, 1, 9,
            4, 1, 8, 1, 3,
            6, 8, 1, 9, 2
        };

    std::vector<float> filterValues =
        {
            4, 5, 6,
            0, 0, 0,
            3, 2, 1
        };

    std::vector<float> biasValues = { 0 };

    std::vector<float> expectedOutputValues =
        {
            23, 33, 24,
            91, 99, 48,
            26, 50, 19
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<float>(tflite::BuiltinOperator_CONV_2D,
                                 ::tflite::TensorType_FLOAT32,
                                 2, // strideX
                                 2, // strideY
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

void Conv2DWithBiasesInt8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    static std::vector<int8_t> inputValues = { 1, 2, 3, 4 };

    std::vector<int8_t> filterValues = { 2, 1, 0, 6 };

    std::vector<int32_t> biasValues = { 10 };

    std::vector<int8_t> expectedOutputValues =
        {
            (1 * 2 + 2 * 1 + 3 * 0 + 4 * 6 + 10) / 2, // 19
            (2 * 2 + 0 * 1 + 4 * 0 + 0 * 6 + 10) / 2, // 7
            (3 * 2 + 4 * 1 + 0 * 0 + 0 * 6 + 10) / 2, // 10
            (4 * 2 + 0 * 1 + 0 * 0 + 0 * 6 + 10) / 2,  // 9
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<int8_t, int32_t>(tflite::BuiltinOperator_CONV_2D,
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
                                            biasValues);
}

void Conv2DWithBiasesReluUint8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    static std::vector<uint8_t> inputValues = { 1, 2, 4, 8 };

    std::vector<uint8_t> filterValues = { 2, 1, 0, 6 };

    std::vector<int32_t> biasValues = { 16 };

    // factors to consider:
    // - the filter zero point is non zero, hence the (x-fz)
    // - the output scale is 2 hence the /2
    // - output zero point is non zero, hence the +outZero
    // - RELU cuts negative values and then we add the output zero point
    uint8_t bias = 16;
    uint8_t outZero = 20;
    uint8_t fz = 4; // filter zero point

    std::vector<uint8_t> expectedOutputValues =
        {
            std::max(outZero, static_cast<uint8_t>((1*(2-fz) + 2*(1-fz) + 4*(0-fz) + 8*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((2*(2-fz) + 0*(1-fz) + 8*(0-fz) + 0*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((4*(2-fz) + 8*(1-fz) + 0*(0-fz) + 0*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((8*(2-fz) + 0*(1-fz) + 0*(0-fz) + 0*(6-fz) + bias)/2 + outZero))
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<uint8_t, int32_t>(tflite::BuiltinOperator_CONV_2D,
                                            ::tflite::TensorType_UINT8,
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
                                            1, // filter scale
                                            4, // filter offset
                                            2, // output scale
                                            20); // output offset
}

void Conv2DWithBiasesRelu6Uint8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };

    static std::vector<uint8_t> inputValues = { 1, 2, 4, 1 };

    std::vector<uint8_t> filterValues = { 2, 1, 0, 6 };

    std::vector<int32_t> biasValues = { 0 };

    // factors to consider:
    // - the output scale is 2 hence the /2
    // - RELU6 cuts output values at +6
    uint8_t relu6Min = 6 / 2; // divide by output scale

    std::vector<uint8_t> expectedOutputValues =
        {
            std::min(relu6Min, static_cast<uint8_t>((1 * 2 + 2 * 1 + 4 * 0 + 1 * 6) / 2)),
            std::min(relu6Min, static_cast<uint8_t>((2 * 2 + 0 * 1 + 1 * 0 + 0 * 6) / 2)),
            std::min(relu6Min, static_cast<uint8_t>((4 * 2 + 1 * 1 + 0 * 0 + 0 * 6) / 2)),
            std::min(relu6Min, static_cast<uint8_t>((1 * 2 + 0 * 1 + 0 * 0 + 0 * 6) / 2))
        };

    tflite::Padding padding = tflite::Padding_SAME;

    ConvolutionTest<uint8_t, int32_t>(tflite::BuiltinOperator_CONV_2D,
                                            ::tflite::TensorType_UINT8,
                                            1, // strideX
                                            1, // strideY
                                            1, // dilationX
                                            1, // dilationY
                                            padding,
                                            tflite::ActivationFunctionType_RELU6,
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

TEST_SUITE("Convolution2dTest_CpuRefTests")
{

TEST_CASE ("Conv2DWithBiases_Fp32_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv2DWithBiasesFp32Test(backends);
}

TEST_CASE ("Conv2DWithBiases_Int8_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv2DWithBiasesInt8Test(backends);
}

} //End of TEST_SUITE("Convolution2dTest_CpuRef")

TEST_SUITE("Convolution2dTest_CpuAccTests")
{

TEST_CASE ("Conv2DWithBiases_Fp32_CpuAcc_Test")
{
std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
Conv2DWithBiasesFp32Test(backends);
}

TEST_CASE ("Conv2DWithBiases_Int8_CpuAcc_Test")
{
std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
Conv2DWithBiasesInt8Test(backends);
}

} //End of TEST_SUITE("Convolution2dTest_CpuAcc")

TEST_SUITE("Convolution2dTest_GpuAccTests")
{

TEST_CASE ("Conv2DWithBiases_Fp32_GpuAcc_Test")
{
std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
Conv2DWithBiasesFp32Test(backends);
}

TEST_CASE ("Conv2DWithBiases_Int8_GpuAcc_Test")
{
std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
Conv2DWithBiasesInt8Test(backends);
}

} //End of TEST_SUITE("Convolution2dTest_GpuAcc")

void TransposeConvInt8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> transposeTensorShape { 4 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    std::vector<int32_t> transposeData = { 1, 3, 3, 1 };
    static std::vector<int8_t> inputValues = { 1, 2, 3, 4 };
    std::vector<int8_t> filterValues = { 0, 1, 2, 4 };
    std::vector<int8_t> expectedOutputValues =
        {
            0, 1,  2,
            2, 11, 12,
            6, 20, 16
        };

    tflite::Padding padding = tflite::Padding_VALID;
    TransposeConvTest<int8_t>(backends,
                             ::tflite::TensorType_INT8,
                             1, // strideX
                             1, // strideY
                             padding,
                             transposeTensorShape,
                             filterShape,
                             inputShape,
                             outputShape,
                             transposeData,
                             filterValues,
                             inputValues,
                             expectedOutputValues);
}

void TransposeConvFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> transposeTensorShape { 4 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    std::vector<int32_t> transposeData = { 1, 3, 3, 1 };
    static std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> filterValues = { 0, 1, 2, 4 };
    std::vector<float> expectedOutputValues =
        {
            0, 1,  2,
            2, 11, 12,
            6, 20, 16
        };

    tflite::Padding padding = tflite::Padding_VALID;
    TransposeConvTest<float>(backends,
                             ::tflite::TensorType_FLOAT32,
                             1, // strideX
                             1, // strideY
                             padding,
                             transposeTensorShape,
                             filterShape,
                             inputShape,
                             outputShape,
                             transposeData,
                             filterValues,
                             inputValues,
                             expectedOutputValues);
}

TEST_SUITE("TransposeConv_CpuRef_Test")
{

TEST_CASE ("TransposeConv_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_CpuRef_Test)

TEST_SUITE("TransposeConv_CpuAcc_Test")
{

TEST_CASE ("TransposeConv_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_CpuAcc_Test)

TEST_SUITE("TransposeConv_GpuAcc_Test")
{

TEST_CASE ("TransposeConv_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_GpuAcc_Test)

} // namespace armnnDelegate