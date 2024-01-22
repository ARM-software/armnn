//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TileTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{
void TileFloat32Test()
{
    // Set input data
    std::vector<float> inputValues =
    {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f
    };

    // Set output data
    std::vector<float> expectedOutputValues =
    {
        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f,

        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f
    };

    // The multiples
    const std::vector<int32_t> multiplesValues = { 2, 2 };

    // Set shapes
    const std::vector<int32_t> inputShape = { 2, 3 };
    const std::vector<int32_t> multiplesShape = { 2 };
    const std::vector<int32_t> expectedOutputShape = { 4, 6 };

    TileFP32TestImpl(tflite::BuiltinOperator_TILE,
                     inputValues,
                     inputShape,
                     multiplesValues,
                     multiplesShape,
                     expectedOutputValues,
                     expectedOutputShape);
}

TEST_SUITE("TileTestsTests")
{

    TEST_CASE ("Tile_Float32_Test")
    {
        TileFloat32Test();
    }

} // TEST_SUITE("Tile_Float32_Test")

} // namespace armnnDelegate