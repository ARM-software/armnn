//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NormalizationTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

TEST_SUITE("L2NormalizationTests")
{

TEST_CASE ("L2NormalizationFp32Test_Test")
{
    L2NormalizationTest();
}

} // TEST_SUITE("L2NormalizationTests")

TEST_SUITE("LocalResponseNormalizationTests")
{

TEST_CASE ("LocalResponseNormalizationTest_Test")
{
    LocalResponseNormalizationTest(3, 1.f, 1.f, 1.f);
}

} // TEST_SUITE("LocalResponseNormalizationTests")

} // namespace armnnDelegate