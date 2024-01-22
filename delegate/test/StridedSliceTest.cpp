//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StridedSliceTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void StridedSlice4DTest()
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

void StridedSlice4DReverseTest()
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

void StridedSliceSimpleStrideTest()
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

void StridedSliceSimpleRangeMaskTest()
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
            {},
            beginMask,
            endMask
    );
}

TEST_SUITE("StridedSliceTests")
{

TEST_CASE ("StridedSlice_4D_Test")
{
    StridedSlice4DTest();
}

TEST_CASE ("StridedSlice_4D_Reverse_Test")
{
    StridedSlice4DReverseTest();
}

TEST_CASE ("StridedSlice_SimpleStride_Test")
{
    StridedSliceSimpleStrideTest();
}

TEST_CASE ("StridedSlice_SimpleRange_Test")
{
    StridedSliceSimpleRangeMaskTest();
}

} // StridedSliceTests TestSuite

} // namespace armnnDelegate