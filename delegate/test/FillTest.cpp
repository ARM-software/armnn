//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FillTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void Fill2dTest(tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
                float fill = 2.0f )
{
    std::vector<int32_t> inputShape { 2 };
    std::vector<int32_t> tensorShape { 2, 2 };
    std::vector<float> expectedOutputValues = { fill, fill,
                                                fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void Fill3dTest(tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
                float fill = 5.0f )
{
    std::vector<int32_t> inputShape { 3 };
    std::vector<int32_t> tensorShape { 3, 3, 3 };
    std::vector<float> expectedOutputValues = { fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill,

                                                fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill,

                                                fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void Fill4dTest(tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
                float fill = 3.0f )
{
    std::vector<int32_t> inputShape { 4 };
    std::vector<int32_t> tensorShape { 2, 2, 4, 4 };
    std::vector<float> expectedOutputValues = { fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void FillInt32Test(tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
                   int32_t fill = 2 )
{
    std::vector<int32_t> inputShape { 2 };
    std::vector<int32_t> tensorShape { 2, 2 };
    std::vector<int32_t> expectedOutputValues = { fill, fill,
                                                  fill, fill };

    FillTest<int32_t>(fillOperatorCode,
                      ::tflite::TensorType_INT32,
                      inputShape,
                      tensorShape,
                      expectedOutputValues,
                      fill);
}

TEST_SUITE("FillTests")
{

TEST_CASE ("Fill2d_Test")
{
    Fill2dTest();
}

TEST_CASE ("Fill3d_Test")
{
    Fill3dTest();
}

TEST_CASE ("Fill3d_Test")
{
    Fill3dTest();
}

TEST_CASE ("Fill4d_Test")
{
    Fill4dTest();
}

TEST_CASE ("FillInt32_Test")
{
    FillInt32Test();
}
} // End of FillTests suite.

} // namespace armnnDelegate