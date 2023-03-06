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
        {55.20419, 24.58061, 67.76520, 47.31617, 55.58102, 44.64565, 105.76307, 54.65538, 80.41088, 66.05208};
    const float actualFloat[] =
        {13.87187, 14.16160, 49.28846, 25.89192, 97.70659, 91.30055, 15.88831, 4.79960, 102.99205, 51.28290};
    const double expectedResult = 74.059098023; // Calculated manually.
    CHECK(ComputeByteLevelRMSE(expectedFloat, expectedFloat, sizeof(float) * 10) == 0);
    CHECK(ComputeByteLevelRMSE(expectedFloat, actualFloat, sizeof(float) * 10) == doctest::Approx(expectedResult));
}

} // End of TEST_SUITE("NetworkExecutionUtilsTests")

} // anonymous namespace