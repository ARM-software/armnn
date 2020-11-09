//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <doctest/doctest.h>
#include <flatbuffers/flatbuffers.h>

namespace armnnDelegate
{

TEST_SUITE ("Transpose_GpuAccTests")
{

TEST_CASE ("Transpose_Float32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    TransposeFP32Test(backends);
}

}

TEST_SUITE ("Transpose_CpuAccTests")
{

TEST_CASE ("Transpose_Float32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    TransposeFP32Test(backends);
}

}

TEST_SUITE ("Transpose_CpuRefTests")
{
TEST_CASE ("Transpose_Float32_CpuRef_Test")
{
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        TransposeFP32Test(backends);
}
}
} // namespace armnnDelegate
