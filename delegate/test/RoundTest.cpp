//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RoundTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void FloorFp32Test(const std::vector<armnn::BackendId>& backends = {})
{
    std::vector<int32_t> inputShape  {1, 3, 2, 3};
    std::vector<int32_t> outputShape {1, 3, 2, 3};

    std::vector<float> inputValues { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
                                     1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f };

    std::vector<float> expectedOutputValues { -38.0f, -16.0f, -9.0f, -2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 0.0f,
                                              1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 8.0f, 15.0f, 37.0f };

    RoundTest<float>(tflite::BuiltinOperator_FLOOR,
                     ::tflite::TensorType_FLOAT32,
                     inputShape,
                     inputValues,
                     expectedOutputValues);
}

// FLOOR Test Suite
TEST_SUITE("FLOORTests")
{

TEST_CASE ("FLOOR_Fp32_Test")
{
    FloorFp32Test();
}

}
// End of FLOOR Test Suite

} // namespace armnnDelegate