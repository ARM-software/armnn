//
// Copyright © 2020, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Gather Operator
void GatherUint8Test()
{

    std::vector<int32_t> paramsShape{8};
    std::vector<int32_t> indicesShape{3};
    std::vector<int32_t> expectedOutputShape{3};

    int32_t              axis = 0;
    std::vector<uint8_t> paramsValues{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> indicesValues{7, 6, 5};
    std::vector<uint8_t> expectedOutputValues{8, 7, 6};

    GatherTest<uint8_t>(::tflite::TensorType_UINT8,
                        paramsShape,
                        indicesShape,
                        expectedOutputShape,
                        axis,
                        paramsValues,
                        indicesValues,
                        expectedOutputValues);
}

void GatherFp32Test()
{
    std::vector<int32_t> paramsShape{8};
    std::vector<int32_t> indicesShape{3};
    std::vector<int32_t> expectedOutputShape{3};

    int32_t              axis = 0;
    std::vector<float>   paramsValues{1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f};
    std::vector<int32_t> indicesValues{7, 6, 5};
    std::vector<float>   expectedOutputValues{8.8f, 7.7f, 6.6f};

    GatherTest<float>(::tflite::TensorType_FLOAT32,
                      paramsShape,
                      indicesShape,
                      expectedOutputShape,
                      axis,
                      paramsValues,
                      indicesValues,
                      expectedOutputValues);
}

// Gather Test Suite
TEST_SUITE("GatherTests")
{

TEST_CASE ("Gather_Uint8_Test")
{
    GatherUint8Test();
}

TEST_CASE ("Gather_Fp32_Test")
{
    GatherFp32Test();
}

}
// End of Gather Test Suite

} // namespace armnnDelegate