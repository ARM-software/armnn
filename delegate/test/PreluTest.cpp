//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreluTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate {

void PreluFloatSimpleTest(bool isAlphaConst, bool isDynamicOutput = false)
{
    std::vector<int32_t> inputShape { 1, 2, 3 };
    std::vector<int32_t> alphaShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 3 };

    if (isDynamicOutput)
    {
        outputShape.clear();
    }

    std::vector<float> inputData = { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f };
    std::vector<float> alphaData = { 0.5f };
    std::vector<float> expectedOutput = { -7.f, 2.f, 0.f, 1.f, -2.5f, 14.f };

    PreluTest(tflite::BuiltinOperator_PRELU,
              ::tflite::TensorType_FLOAT32,
              inputShape,
              alphaShape,
              outputShape,
              inputData,
              alphaData,
              expectedOutput,
              isAlphaConst);
}

TEST_SUITE("PreluTests")
{

TEST_CASE ("PreluFp32SimpleConstTest_Test")
{
    PreluFloatSimpleTest(true);
}

TEST_CASE ("PreluFp32SimpleTest_Test")
{
    PreluFloatSimpleTest(false);
}

TEST_CASE ("PreluFp32SimpleConstDynamicTest_Test")
{
    PreluFloatSimpleTest(true, true);
}

TEST_CASE ("PreluFp32SimpleDynamicTest_Test")
{
    PreluFloatSimpleTest(false, true);
}

} // TEST_SUITE("PreluTests")

}