//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"

#include <doctest/doctest.h>

TEST_SUITE("OnnxParser_Constructor")
{
TEST_CASE("Create")
{
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
}

}
