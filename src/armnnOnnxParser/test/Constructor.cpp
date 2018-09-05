//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnOnnxParser/IOnnxParser.hpp"

BOOST_AUTO_TEST_SUITE(OnnxParser)

BOOST_AUTO_TEST_CASE(Create)
{
    armnnOnnxParser::IOnnxParserPtr parser(armnnOnnxParser::IOnnxParser::Create());
}

BOOST_AUTO_TEST_SUITE_END()
