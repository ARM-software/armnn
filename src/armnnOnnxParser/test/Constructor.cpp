//
// Copyright © 2017, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"

#include <doctest/doctest.h>

ARMNN_NO_DEPRECATE_WARN_BEGIN
TEST_SUITE("OnnxParser_Constructor")
{
TEST_CASE("Create")
{
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
}

}
ARMNN_NO_DEPRECATE_WARN_END