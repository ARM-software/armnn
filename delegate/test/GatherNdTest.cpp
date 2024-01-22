//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherNdTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Gather_Nd Operator
void GatherNdUint8Test()
{

    std::vector<int32_t> paramsShape{ 5, 2 };
    std::vector<int32_t> indicesShape{ 3, 1 };
    std::vector<int32_t> expectedOutputShape{ 3, 2 };

    std::vector<uint8_t> paramsValues{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    std::vector<int32_t> indicesValues{ 1, 0, 4 };
    std::vector<uint8_t> expectedOutputValues{ 3, 4, 1, 2, 9, 10 };

    GatherNdTest<uint8_t>(::tflite::TensorType_UINT8,
                          paramsShape,
                          indicesShape,
                          expectedOutputShape,
                          paramsValues,
                          indicesValues,
                          expectedOutputValues);
}

void GatherNdFp32Test()
{
    std::vector<int32_t> paramsShape{ 5, 2 };
    std::vector<int32_t> indicesShape{ 3, 1 };
    std::vector<int32_t> expectedOutputShape{ 3, 2 };

    std::vector<float>   paramsValues{ 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.10f };
    std::vector<int32_t> indicesValues{ 1, 0, 4 };
    std::vector<float>   expectedOutputValues{ 3.3f, 4.4f, 1.1f, 2.2f, 9.9f, 10.10f };

    GatherNdTest<float>(::tflite::TensorType_FLOAT32,
                        paramsShape,
                        indicesShape,
                        expectedOutputShape,
                        paramsValues,
                        indicesValues,
                        expectedOutputValues);
}

// Gather_Nd Test Suite
TEST_SUITE("Gather_NdTests")
{

TEST_CASE ("Gather_Nd_Uint8_Test")
{
    GatherNdUint8Test();
}

TEST_CASE ("Gather_Nd_Fp32_Test")
{
    GatherNdFp32Test();
}

}

// End of Gather_Nd Test Suite

} // namespace armnnDelegate