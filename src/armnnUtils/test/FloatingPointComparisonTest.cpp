//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnUtils/FloatingPointComparison.hpp>

#include <doctest/doctest.h>

using namespace armnnUtils;

TEST_SUITE("FloatingPointComparisonSuite")
{
TEST_CASE("FloatingPointComparisonDefaultTolerance")
{
    // 1% range of 1.2 is 1.188 -> 1.212
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(1.2f, 1.17f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(1.2f, 1.213f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(1.2f, 1.189f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(1.2f, 1.210f));
    // Exact match
    CHECK(within_percentage_tolerance(1.2f, 1.2f));

    // Negative value tests.
    CHECK(!within_percentage_tolerance(-1.2f, -1.17f));
    CHECK(!within_percentage_tolerance(-1.2f, -1.213f));
    CHECK(within_percentage_tolerance(-1.2f, -1.189f));
    CHECK(within_percentage_tolerance(-1.2f, -1.210f));
    CHECK(within_percentage_tolerance(-1.2f, -1.2f));

    // Negative & positive tests
    CHECK(!within_percentage_tolerance(1.2f, -1.2f));
    CHECK(!within_percentage_tolerance(-1.2f, 1.2f));

    // Negative and positive test with large float values.
    CHECK(!within_percentage_tolerance(3.3E+38f, -1.17549435e38f));
    CHECK(!within_percentage_tolerance(-1.17549435e38f, 3.3E+38f));

    // 1% range of 0.04 is 0.0396 -> 0.0404
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(0.04f, 0.039f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(0.04f, 0.04041f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(0.04f, 0.0397f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(0.04f, 0.04039f));
    // Exact match
    CHECK(within_percentage_tolerance(0.04f, 0.04f));
}

TEST_CASE("FloatingPointComparisonLargePositiveNumbersDefaultTolerance")
{
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(3.3E+38f, (3.3E+38f * 0.989f)));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(3.3E+38f, (3.3E+38f * 1.011f)));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(3.3E+38f, (3.3E+38f * 0.992f)));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(3.3E+38f, (3.3E+38f * 1.009f)));
    // Exact match
    CHECK(within_percentage_tolerance(3.3E+38f, 3.3E+38f));
}

TEST_CASE("FloatingPointComparisonLargeNegativeNumbersDefaultTolerance")
{
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(-1.17549435e38f, (-1.17549435e38f * -1.009f)));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(-1.17549435e38f, (-1.17549435e38f * 1.011f)));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(-1.17549435e38f, -1.17549435e38f - (-1.17549435e38f * 0.0099f)));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(-1.17549435e38f, -1.17549435e38f + (-1.17549435e38f * 0.0099f)));
    // Exact match
    CHECK(within_percentage_tolerance(-1.17549435e38f, -1.17549435e38f));
}

TEST_CASE("FloatingPointComparisonSpecifiedTolerance")
{
    // 2% range of 1.2 is 1.176 -> 1.224
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(1.2f, 1.175f, 2.0f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(1.2f, 1.226f, 2.0f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(1.2f, 1.18f, 2.0f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(1.2f, 1.22f, 2.0f));
    // Exact match.
    CHECK(within_percentage_tolerance(1.2f, 1.2f, 2.0f));

    // 5% range of 6.2 is 5.89 -> 6.51
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(6.2f, 5.88f, 5.0f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(6.2f, 6.52f, 5.0f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(6.2f, 5.9f, 5.0f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(6.2f, 6.5f, 5.0f));

    // Larger tolerance (unlikely to be used).
    CHECK(within_percentage_tolerance(10.0f, 9.01f, 10.0f));
    CHECK(!within_percentage_tolerance(10.0f, 8.99f, 10.0f));
}

TEST_CASE("FloatingPointComparisonLargePositiveNumbersSpecifiedTolerance")
{
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(3.3E+38f, (3.3E+38f * 0.979f), 2.0f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(3.3E+38f, (3.3E+38f * 1.021f), 2.0f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(3.3E+38f, (3.3E+38f * 0.982f), 2.0f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(3.3E+38f, (3.3E+38f * 1.019f), 2.0f));
}

TEST_CASE("FloatingPointComparisonLargeNegativeNumbersSpecifiedTolerance")
{
    // Just below tolerance.
    CHECK(!within_percentage_tolerance(-1.17549435e38f, (-1.17549435e38f * -1.019f), 2.0f));
    // Just above tolerance.
    CHECK(!within_percentage_tolerance(-1.17549435e38f, (-1.17549435e38f * 1.021f), 2.0f));
    // Just inside the lower range.
    CHECK(within_percentage_tolerance(-1.17549435e38f, -1.17549435e38f - (-1.17549435e38f * 0.0089f), 2.0f));
    // Just inside the upper range.
    CHECK(within_percentage_tolerance(-1.17549435e38f, -1.17549435e38f + (-1.17549435e38f * 0.0089f), 2.0f));
}

}
