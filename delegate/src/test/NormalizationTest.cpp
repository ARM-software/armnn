//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NormalizationTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void L2NormalizationTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 1, 1, 10 };
    std::vector<int32_t> outputShape { 1, 1, 1, 10 };

    std::vector<float> inputValues
    {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f,
        9.0f,
        10.0f
    };

    const float approxInvL2Norm = 0.050964719f;
    std::vector<float> expectedOutputValues
    {
        1.0f  * approxInvL2Norm,
        2.0f  * approxInvL2Norm,
        3.0f  * approxInvL2Norm,
        4.0f  * approxInvL2Norm,
        5.0f  * approxInvL2Norm,
        6.0f  * approxInvL2Norm,
        7.0f  * approxInvL2Norm,
        8.0f  * approxInvL2Norm,
        9.0f  * approxInvL2Norm,
        10.0f * approxInvL2Norm
    };

    NormalizationTest<float>(tflite::BuiltinOperator_L2_NORMALIZATION,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             outputShape,
                             inputValues,
                             expectedOutputValues);
}

void LocalResponseNormalizationTest(std::vector<armnn::BackendId>& backends,
                                    int32_t radius,
                                    float bias,
                                    float alpha,
                                    float beta)
{
    // Set input data
    std::vector<int32_t> inputShape  { 2, 2, 2, 1 };
    std::vector<int32_t> outputShape { 2, 2, 2, 1 };

    std::vector<float> inputValues
    {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> expectedOutputValues
    {
        0.5f, 0.400000006f, 0.300000012f, 0.235294119f,
        0.192307696f, 0.16216217f, 0.140000001f, 0.123076923f
    };

    NormalizationTest<float>(tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION,
                             ::tflite::TensorType_FLOAT32,
                             backends,
                             inputShape,
                             outputShape,
                             inputValues,
                             expectedOutputValues,
                             radius,
                             bias,
                             alpha,
                             beta);
}


TEST_SUITE("L2Normalization_CpuRefTests")
{

TEST_CASE ("L2NormalizationFp32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    L2NormalizationTest(backends);
}

} // TEST_SUITE("L2Normalization_CpuRefTests")

TEST_SUITE("L2Normalization_CpuAccTests")
{

TEST_CASE ("L2NormalizationFp32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    L2NormalizationTest(backends);
}

} // TEST_SUITE("L2NormalizationFp32Test_CpuAcc_Test")

TEST_SUITE("L2Normalization_GpuAccTests")
{

TEST_CASE ("L2NormalizationFp32Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    L2NormalizationTest(backends);
}

} // TEST_SUITE("L2Normalization_GpuAccTests")

TEST_SUITE("LocalResponseNormalization_CpuRefTests")
{

TEST_CASE ("LocalResponseNormalizationTest_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    LocalResponseNormalizationTest(backends, 3, 1.f, 1.f, 1.f);
}

} // TEST_SUITE("LocalResponseNormalization_CpuRefTests")

TEST_SUITE("LocalResponseNormalization_CpuAccTests")
{

TEST_CASE ("LocalResponseNormalizationTest_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    LocalResponseNormalizationTest(backends, 3, 1.f, 1.f, 1.f);
}

} // TEST_SUITE("LocalResponseNormalization_CpuAccTests")

TEST_SUITE("LocalResponseNormalization_GpuAccTests")
{

TEST_CASE ("LocalResponseNormalizationTest_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    LocalResponseNormalizationTest(backends, 3, 1.f, 1.f, 1.f);
}

} // TEST_SUITE("LocalResponseNormalization_GpuAccTests")

} // namespace armnnDelegate