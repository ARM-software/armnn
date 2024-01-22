//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2TestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ReverseV2Float32Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f
    };

    // The output data
    std::vector<float> expectedOutputValues =
    {
        3.0f, 2.0f, 1.0f,
        6.0f, 5.0f, 4.0f,
        9.0f, 8.0f, 7.0f,

        13.0f, 12.0f, 11.0f,
        16.0f, 15.0f, 14.0f,
        19.0f, 18.0f, 17.0f,

        23.0f, 22.0f, 21.0f,
        26.0f, 25.0f, 24.0f,
        29.0f, 28.0f, 27.0f
    };

    // The axis to reverse
    const std::vector<int32_t> axisValues = {2};

    // Shapes
    const std::vector<int32_t> inputShape = {3, 3, 3};
    const std::vector<int32_t> axisShapeDims = {1};
    const std::vector<int32_t> expectedOutputShape = {3, 3, 3};

    ReverseV2FP32TestImpl(tflite::BuiltinOperator_REVERSE_V2,
                          inputValues,
                          inputShape,
                          axisValues,
                          axisShapeDims,
                          expectedOutputValues,
                          expectedOutputShape);
}

void ReverseV2NegativeAxisFloat32Test(const std::vector<armnn::BackendId>& backends = {})
{
    // Set input data
    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,

        11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f,

        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,
        27.0f, 28.0f, 29.0f
    };

    // The output data
    std::vector<float> expectedOutputValues =
    {
        7.0f, 8.0f, 9.0f,
        4.0f, 5.0f, 6.0f,
        1.0f, 2.0f, 3.0f,

        17.0f, 18.0f, 19.0f,
        14.0f, 15.0f, 16.0f,
        11.0f, 12.0f, 13.0f,

        27.0f, 28.0f, 29.0f,
        24.0f, 25.0f, 26.0f,
        21.0f, 22.0f, 23.0f
    };

    // The axis to reverse
    const std::vector<int32_t> axisValues = {-2};

    // Shapes
    const std::vector<int32_t> inputShape = {3, 3, 3};
    const std::vector<int32_t> axisShapeDims = {1};
    const std::vector<int32_t> expectedOutputShape = {3, 3, 3};

    ReverseV2FP32TestImpl(tflite::BuiltinOperator_REVERSE_V2,
                          inputValues,
                          inputShape,
                          axisValues,
                          axisShapeDims,
                          expectedOutputValues,
                          expectedOutputShape);
}

TEST_SUITE("ReverseV2TestsTests")
{

    TEST_CASE ("ReverseV2_Float32_Test")
    {
        ReverseV2Float32Test();
    }

    TEST_CASE ("ReverseV2_NegativeAxis_Float32_Test")
    {
        ReverseV2NegativeAxisFloat32Test();
    }

} // TEST_SUITE("ReverseV2TestsTests")

} // namespace armnnDelegate
