//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../NetworkExecutionUtils.hpp"

#include <doctest/doctest.h>

namespace
{

TEST_SUITE("NetworkExecutionUtilsTests")
{

TEST_CASE ("ComputeByteLevelRMSE")
{
    // Bytes.
    const uint8_t expected[] = {1, 128, 255};
    const uint8_t actual[] = {0, 127, 254};

    CHECK(ComputeByteLevelRMSE(expected, expected, 3) == 0);
    CHECK(ComputeByteLevelRMSE(expected, actual, 3) == 1.0);

    // Floats.
    const float expectedFloat[] =
        {55.20419f, 24.58061f, 67.76520f, 47.31617f, 55.58102f, 44.64565f, 105.76307f, 54.65538f, 80.41088f, 66.05208f};
    const float actualFloat[] =
        {13.87187f, 14.16160f, 49.28846f, 25.89192f, 97.70659f, 91.30055f, 15.88831f, 4.79960f, 102.99205f, 51.28290f};
    const double expectedResult = 74.059098023; // Calculated manually.
    CHECK(ComputeByteLevelRMSE(expectedFloat, expectedFloat, sizeof(float) * 10) == 0);
    CHECK(ComputeByteLevelRMSE(expectedFloat, actualFloat, sizeof(float) * 10) == doctest::Approx(expectedResult));
}

} // End of TEST_SUITE("NetworkExecutionUtilsTests")

} // anonymous namespace