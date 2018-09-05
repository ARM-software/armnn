//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)

struct DropoutFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    DropoutFixture()
    {
        m_Prototext = "name: \"DropoutFixture\"\n"
            "layer {\n"
            "    name: \"data\"\n"
            "    type: \"Input\"\n"
            "    top: \"data\"\n"
            "    input_param { shape: { dim: 1 dim: 1 dim: 2 dim: 2 } }\n"
            "}\n"
            "layer {\n"
            "    bottom: \"data\"\n"
            "    top: \"drop1\"\n"
            "    name: \"drop1\"\n"
            "    type: \"Dropout\"\n"
            "}\n"
            "layer {\n"
            "    bottom: \"drop1\"\n"
            "    bottom: \"drop1\"\n"
            "    top: \"add\"\n"
            "    name: \"add\"\n"
            "    type: \"Eltwise\"\n"
            "}\n";
        SetupSingleInputSingleOutput("data", "add");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseDropout, DropoutFixture)
{
    RunTest<4>(
        {
            1, 2,
            3, 4,
        },
        {
            2, 4,
            6, 8
        });
}

BOOST_AUTO_TEST_SUITE_END()
