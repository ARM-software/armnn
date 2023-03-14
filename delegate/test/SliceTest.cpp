//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void SliceFixtureSimpleTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  { 3, 2, 3 };
    std::vector<int32_t> outputShape { 2, 1, 3 };
    std::vector<int32_t> beginShape  { 3 };
    std::vector<int32_t> sizeShape   { 3 };

    std::vector<int32_t> beginData { 1, 0, 0 };
    std::vector<int32_t> sizeData  { 2, 1, 3 };
    std::vector<float> inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                    5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float> outputData { 3.0f, 3.0f, 3.0f,
                                    5.0f, 5.0f, 5.0f };

    SliceTestImpl<float>(
        backends,
        inputData,
        outputData,
        beginData,
        sizeData,
        inputShape,
        beginShape,
        sizeShape,
        outputShape);
}

TEST_SUITE("Slice_CpuRefTests")
{

TEST_CASE ("Slice_Simple_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SliceFixtureSimpleTest(backends);
}

} // Slice_CpuRefTests TestSuite



TEST_SUITE("Slice_CpuAccTests")
{

TEST_CASE ("Slice_Simple_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SliceFixtureSimpleTest(backends);
}

} // Slice_CpuAccTests TestSuite



TEST_SUITE("StridedSlice_GpuAccTests")
{

TEST_CASE ("Slice_Simple_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SliceFixtureSimpleTest(backends);
}

} // Slice_GpuAccTests TestSuite

} // namespace armnnDelegate