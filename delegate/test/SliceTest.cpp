//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void SliceFixtureSimpleTest(const std::vector<armnn::BackendId>& backends = {})
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
        inputData,
        outputData,
        beginData,
        sizeData,
        inputShape,
        beginShape,
        sizeShape,
        outputShape);
}

void SliceFixtureSizeTest(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  { 3, 2, 3 };
    std::vector<int32_t> outputShape { 2, 1, 3 };
    std::vector<int32_t> beginShape  { 3 };
    std::vector<int32_t> sizeShape   { 3 };

    std::vector<int32_t> beginData { 1, 0, 0 };
    std::vector<int32_t> sizeData  { 2, 1, -1 };
    std::vector<float> inputData  { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                    3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                    5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f };
    std::vector<float> outputData { 3.0f, 3.0f, 3.0f,
                                    5.0f, 5.0f, 5.0f };

    SliceTestImpl<float>(
            inputData,
            outputData,
            beginData,
            sizeData,
            inputShape,
            beginShape,
            sizeShape,
            outputShape);
}

TEST_SUITE("SliceTests")
{

TEST_CASE ("Slice_Simple_Test")
{
    SliceFixtureSimpleTest();
}

TEST_CASE ("Slice_Size_Test")
{
    SliceFixtureSizeTest();
}

} // SliceTests TestSuite

} // namespace armnnDelegate