//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ExpandDimsSimpleTest()
{
    // Set input data
    std::vector<int32_t> inputShape  { 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };
    std::vector<int32_t> axis { 0 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_EXPAND_DIMS,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        axis,
                        true);
}

void ExpandDimsWithNegativeAxisTest()
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2 };
    std::vector<int32_t> outputShape { 1, 2, 2, 1 };
    std::vector<int32_t> axis { -1 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_EXPAND_DIMS,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        axis,
                        true);
}

TEST_SUITE("ExpandDimsTests")
{

TEST_CASE ("ExpandDims_Simple_Test")
{
    ExpandDimsSimpleTest();
}

TEST_CASE ("ExpandDims_With_Negative_Axis_Test")
{
    ExpandDimsWithNegativeAxisTest();
}

} // TEST_SUITE("ExpandDimsTests")

} // namespace armnnDelegate