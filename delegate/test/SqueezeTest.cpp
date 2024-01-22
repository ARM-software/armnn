//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RedefineTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void SqueezeSimpleTest()
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 2, 2 };
    std::vector<int32_t> squeezeDims { };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_SQUEEZE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        squeezeDims);
}

void SqueezeWithDimsTest()
{
    // Set input data
    std::vector<int32_t> inputShape  { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 2, 2 };
    std::vector<int32_t> squeezeDims { -1 };

    std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> expectedOutputValues = { 1, 2, 3, 4 };

    RedefineTest<float>(tflite::BuiltinOperator_SQUEEZE,
                        ::tflite::TensorType_FLOAT32,
                        inputShape,
                        outputShape,
                        inputValues,
                        expectedOutputValues,
                        squeezeDims);
}

TEST_SUITE("SqueezeTests")
{

TEST_CASE ("Squeeze_Simple_Test")
{
    SqueezeSimpleTest();
}

TEST_CASE ("Squeeze_With_Dims_Test")
{
    SqueezeWithDimsTest();
}

} // TEST_SUITE("SqueezeTests")

} // namespace armnnDelegate