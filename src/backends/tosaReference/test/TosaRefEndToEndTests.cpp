//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/AdditionEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TosaRefEndToEnd")
{
std::vector<armnn::BackendId> tosaDefaultBackends = { "TosaRef" };

// Addition
TEST_CASE("TosaRefEndtoEndTestFloat32")
{
    AdditionEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

}