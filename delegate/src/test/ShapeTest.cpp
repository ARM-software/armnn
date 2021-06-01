//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ShapeTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ShapeSimpleTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape{ 1, 3, 2, 3 };

    std::vector<int32_t> inputValues{ 1, 1, 1, 1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1, 1, 1, 1, };

    std::vector<int32_t> expectedOutputShape{ 4 };
    std::vector<int32_t> expectedOutputValues{ 1, 3, 2, 3 };

    ShapeTest<int32_t, int32_t>(::tflite::TensorType_INT32,
                                ::tflite::TensorType_INT32,
                                backends,
                                inputShape,
                                inputValues,
                                expectedOutputValues,
                                expectedOutputShape);
}

// SHAPE Test Suite
TEST_SUITE("SHAPE_CpuRefTests")
{

TEST_CASE("SHAPE_Simple_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ShapeSimpleTest(backends);
}

}
// End of SHAPE Test Suite

} // namespace armnnDelegate