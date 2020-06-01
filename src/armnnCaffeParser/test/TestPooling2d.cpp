//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "armnnCaffeParser/ICaffeParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(CaffeParser)

struct GlobalPoolingFixture : public armnnUtils::ParserPrototxtFixture<armnnCaffeParser::ICaffeParser>
{
    GlobalPoolingFixture()
    {
        m_Prototext = "name: \"GlobalPooling\"\n"
            "layer {\n"
            "  name: \"data\"\n"
            "  type: \"Input\"\n"
            "  top: \"data\"\n"
            "  input_param { shape: { dim: 1 dim: 3 dim: 2 dim: 2 } }\n"
            "}\n"
            "layer {\n"
            "    bottom: \"data\"\n"
            "    top: \"pool1\"\n"
            "    name: \"pool1\"\n"
            "    type: \"Pooling\"\n"
            "    pooling_param {\n"
            "        pool: AVE\n"
            "        global_pooling: true\n"
            "    }\n"
            "}\n";
        SetupSingleInputSingleOutput("data", "pool1");
    }
};

BOOST_FIXTURE_TEST_CASE(GlobalPooling, GlobalPoolingFixture)
{
    RunTest<4>(
        {
            1,3,
            5,7,

            2,2,
            2,2,

            4,4,
            6,6
        },
        {
            4, 2, 5
        });
}

BOOST_AUTO_TEST_SUITE_END()
