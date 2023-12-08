//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvolutionTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

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

TEST_CASE ("TransposeConv_CpuRef_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_CpuRef_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_CpuRef_Test)

TEST_SUITE("TransposeConv_CpuAcc_Test")
{

TEST_CASE ("TransposeConv_CpuAcc_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_CpuAcc_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_CpuAcc_Test)

TEST_SUITE("TransposeConv_GpuAcc_Test")
{

TEST_CASE ("TransposeConv_GpuAcc_Fp32_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    TransposeConvFp32Test(backends);
}

TEST_CASE ("TransposeConv_GpuAcc_Int8_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    TransposeConvInt8Test(backends);
}

} // End of  TEST_SUITE(TransposeConv_GpuAcc_Test)

} // namespace armnnDelegate