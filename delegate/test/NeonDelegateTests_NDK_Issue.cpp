//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NormalizationTestHelper.hpp"
#include "SoftmaxTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{
// There's a known Android NDK bug which causes this subset of Neon Tests to
// fail. We'll exclude these tests in if we're doing
// a debug build and NDK is less than r21.
// The exclusion takes place in test/CMakeLists.txt
// https://github.com/android/ndk/issues/1135

TEST_SUITE ("Softmax_CpuAccTests")
{

TEST_CASE ("Softmax_Standard_Beta_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput = {0.00994190481, 0.0445565246, 0.0734612942, 0.329230666, 0.542809606,
                                         0.710742831, 0.158588171, 0.0961885825, 0.0214625746, 0.0130177103};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, 1, expectedOutput, backends);
}

TEST_CASE ("Softmax_Different_Beta_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput = {
        0.0946234912, 0.148399189, 0.172415257, 0.270400971, 0.314161092,
        0.352414012, 0.224709094, 0.193408906, 0.123322964, 0.106145054};
    SoftmaxTestCase(tflite::BuiltinOperator_SOFTMAX, 0.3, expectedOutput, backends);
}

TEST_CASE ("Log_Softmax_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::CpuAcc };
    std::vector<float> expectedOutput =
        {-4.61099672, -3.11099672, -2.61099672, -1.11099672, -0.610996664,
         -0.341444582, -1.84144461, -2.34144449, -3.84144449, -4.34144449};
    SoftmaxTestCase(tflite::BuiltinOperator_LOG_SOFTMAX, 0, expectedOutput, backends);
}
} // TEST_SUITE ("Softmax_CpuAccTests")

TEST_SUITE("L2Normalization_CpuAccTests")
{

TEST_CASE ("L2NormalizationFp32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::CpuAcc };
    L2NormalizationTest(backends);
}
} // TEST_SUITE("L2NormalizationFp32Test_CpuAcc_Test")
}