//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void StridedSlice4DTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  { 3, 2, 3, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };
    std::vector<int32_t> beginShape  { 4 };
    std::vector<int32_t> endShape    { 4 };
    std::vector<int32_t> strideShape { 4 };

    std::vector<int32_t> beginData  { 1, 0, 0, 0 };
    std::vector<int32_t> endData    { 2, 2, 3, 1 };
    std::vector<int32_t> strideData { 1, 1, 1, 1 };
    std::vector<float> inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                    5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float> outputData { 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f };

    StridedSliceTestImpl<float>(
            backends,
            inputData,
            outputData,
            beginData,
            endData,
            strideData,
            inputShape,
            beginShape,
            endShape,
            strideShape,
            outputShape
            );
}

void StridedSlice4DReverseTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  { 3, 2, 3, 1 };
    std::vector<int32_t> outputShape { 1, 2, 3, 1 };
    std::vector<int32_t> beginShape  { 4 };
    std::vector<int32_t> endShape    { 4 };
    std::vector<int32_t> strideShape { 4 };

    std::vector<int32_t> beginData  { 1, -1, 0, 0 };
    std::vector<int32_t> endData    { 2, -3, 3, 1 };
    std::vector<int32_t> strideData { 1, -1, 1, 1 };
    std::vector<float>   inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                      3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                      5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float>   outputData { 4.0f, 4.0f, 4.0f, 3.0f, 3.0f, 3.0f };

    StridedSliceTestImpl<float>(
            backends,
            inputData,
            outputData,
            beginData,
            endData,
            strideData,
            inputShape,
            beginShape,
            endShape,
            strideShape,
            outputShape
    );
}

void StridedSliceSimpleStrideTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  { 3, 2, 3, 1 };
    std::vector<int32_t> outputShape { 2, 1, 2, 1 };
    std::vector<int32_t> beginShape  { 4 };
    std::vector<int32_t> endShape    { 4 };
    std::vector<int32_t> strideShape { 4 };

    std::vector<int32_t> beginData  { 0, 0, 0, 0 };
    std::vector<int32_t> endData    { 3, 2, 3, 1 };
    std::vector<int32_t> strideData { 2, 2, 2, 1 };
    std::vector<float>   inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                      3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                      5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float>   outputData { 1.0f, 1.0f,
                                      5.0f, 5.0f };

    StridedSliceTestImpl<float>(
            backends,
            inputData,
            outputData,
            beginData,
            endData,
            strideData,
            inputShape,
            beginShape,
            endShape,
            strideShape,
            outputShape
    );
}

void StridedSliceSimpleRangeMaskTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape  { 3, 2, 3, 1 };
    std::vector<int32_t> outputShape { 3, 2, 3, 1 };
    std::vector<int32_t> beginShape  { 4 };
    std::vector<int32_t> endShape    { 4 };
    std::vector<int32_t> strideShape { 4 };

    std::vector<int32_t> beginData  { 1, 1, 1, 1 };
    std::vector<int32_t> endData    { 1, 1, 1, 1 };
    std::vector<int32_t> strideData { 1, 1, 1, 1 };

    int beginMask = -1;
    int endMask   = -1;

    std::vector<float>   inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                      3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                      5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float>   outputData { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                      3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                      5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };

    StridedSliceTestImpl<float>(
            backends,
            inputData,
            outputData,
            beginData,
            endData,
            strideData,
            inputShape,
            beginShape,
            endShape,
            strideShape,
            outputShape,
            beginMask,
            endMask
    );
}


TEST_SUITE("StridedSlice_CpuRefTests")
{

TEST_CASE ("StridedSlice_4D_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    StridedSlice4DTest(backends);
}

TEST_CASE ("StridedSlice_4D_Reverse_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    StridedSlice4DReverseTest(backends);
}

TEST_CASE ("StridedSlice_SimpleStride_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    StridedSliceSimpleStrideTest(backends);
}

TEST_CASE ("StridedSlice_SimpleRange_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    StridedSliceSimpleRangeMaskTest(backends);
}

} // StridedSlice_CpuRefTests TestSuite



TEST_SUITE("StridedSlice_CpuAccTests")
{

TEST_CASE ("StridedSlice_4D_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    StridedSlice4DTest(backends);
}

TEST_CASE ("StridedSlice_4D_Reverse_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    StridedSlice4DReverseTest(backends);
}

TEST_CASE ("StridedSlice_SimpleStride_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    StridedSliceSimpleStrideTest(backends);
}

TEST_CASE ("StridedSlice_SimpleRange_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    StridedSliceSimpleRangeMaskTest(backends);
}

} // StridedSlice_CpuAccTests TestSuite



TEST_SUITE("StridedSlice_GpuAccTests")
{

TEST_CASE ("StridedSlice_4D_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    StridedSlice4DTest(backends);
}

TEST_CASE ("StridedSlice_4D_Reverse_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    StridedSlice4DReverseTest(backends);
}

TEST_CASE ("StridedSlice_SimpleStride_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    StridedSliceSimpleStrideTest(backends);
}

TEST_CASE ("StridedSlice_SimpleRange_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    StridedSliceSimpleRangeMaskTest(backends);
}

} // StridedSlice_GpuAccTests TestSuite

} // namespace armnnDelegate