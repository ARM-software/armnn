//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SoftmaxTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

/// Convenience function to run softmax and log-softmax test cases
/// \param operatorCode tflite::BuiltinOperator_SOFTMAX or tflite::BuiltinOperator_LOG_SOFTMAX
/// \param backends armnn backends to target
/// \param beta multiplicative parameter to the softmax function
/// \param expectedOutput to be checked against transformed input
void SoftmaxTestCase(tflite::BuiltinOperator operatorCode,
                     std::vector<armnn::BackendId> backends, float beta, std::vector<float> expectedOutput) {
    std::vector<float> input = {
        1.0, 2.5, 3.0, 4.5, 5.0,
        -1.0, -2.5, -3.0, -4.5, -5.0};
    std::vector<int32_t> shape = {2, 5};

    SoftmaxTest(operatorCode,
                tflite::TensorType_FLOAT32,
                backends,
                shape,
                input,
                expectedOutput,
                beta);
}

TEST_SUITE ("Softmax_GpuAccTests")
{

TEST_CASE ("Softmax_Standard_Beta_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    std::vector<float> expectedOutput = {0.00994190481, 0.0445565246, 0.0734612942, 0.329230666, 0.542809606,
                                         0.710742831, 0.158588171, 0.0961885825, 0.0214625746, 0.0130177103};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 1, expectedOutput);
}

TEST_CASE ("Softmax_Different_Beta_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    std::vector<float> expectedOutput = {0.0946234912, 0.148399189, 0.172415257, 0.270400971, 0.314161092, 0.352414012,
                                         0.224709094, 0.193408906, 0.123322964, 0.106145054};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 0.3, expectedOutput);

}

TEST_CASE ("Log_Softmax_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    std::vector<float> expectedOutput =
        {-4.61099672, -3.11099672, -2.61099672, -1.11099672, -0.610996664,
         -0.341444582, -1.84144461, -2.34144449, -3.84144449, -4.34144449};
    SoftmaxTestCase(tflite::BuiltinOperator_LOG_SOFTMAX, backends, 0, expectedOutput);
}
} // TEST_SUITE ("Softmax_GpuAccTests")

TEST_SUITE ("Softmax_CpuAccTests")
{

TEST_CASE ("Softmax_Standard_Beta_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput = {0.00994190481, 0.0445565246, 0.0734612942, 0.329230666, 0.542809606,
                                         0.710742831, 0.158588171, 0.0961885825, 0.0214625746, 0.0130177103};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 1, expectedOutput);
}

TEST_CASE ("Softmax_Different_Beta_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput = {
        0.0946234912, 0.148399189, 0.172415257, 0.270400971, 0.314161092,
        0.352414012, 0.224709094, 0.193408906, 0.123322964, 0.106145054};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 0.3, expectedOutput);
}

TEST_CASE ("Log_Softmax_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput =
        {-4.61099672, -3.11099672, -2.61099672, -1.11099672, -0.610996664,
         -0.341444582, -1.84144461, -2.34144449, -3.84144449, -4.34144449};
    SoftmaxTestCase(tflite::BuiltinOperator_LOG_SOFTMAX, backends, 0, expectedOutput);
}
} // TEST_SUITE ("Softmax_CpuAccTests")

TEST_SUITE ("Softmax_CpuRefTests")
{

TEST_CASE ("Softmax_Standard_Beta_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<float> expectedOutput = {
        0.00994190481, 0.0445565246, 0.0734612942, 0.329230666, 0.542809606,
        0.710742831, 0.158588171, 0.0961885825, 0.0214625746, 0.0130177103};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 1, expectedOutput);
}

TEST_CASE ("Softmax_Different_Beta_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<float> expectedOutput = {
        0.0946234912, 0.148399189, 0.172415257, 0.270400971, 0.314161092,
        0.352414012, 0.224709094, 0.193408906, 0.123322964, 0.106145054};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, backends, 0.3, expectedOutput);
}

TEST_CASE ("Log_Softmax_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    std::vector<float> expectedOutput =
        {-4.61099672, -3.11099672, -2.61099672, -1.11099672, -0.610996664,
         -0.341444582, -1.84144461, -2.34144449, -3.84144449, -4.34144449};
    SoftmaxTestCase(tflite::BuiltinOperator_LOG_SOFTMAX, backends, 0, expectedOutput);
}
} // TEST_SUITE ("Softmax_CpuRefTests")
} // namespace armnnDelegate
