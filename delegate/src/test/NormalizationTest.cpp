//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NormalizationTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("L2Normalization_CpuRefTests")
{

TEST_CASE ("L2NormalizationFp32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    L2NormalizationTest(backends);
}

} // TEST_SUITE("L2Normalization_CpuRefTests")

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