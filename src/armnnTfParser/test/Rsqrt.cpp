//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct RsqrtFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    RsqrtFixture()
    {
        m_Prototext = "node {\n"
                      "  name: \"input\"\n"
                      "  op: \"Placeholder\"\n"
                      "  attr {\n"
                      "    key: \"dtype\"\n"
                      "    value {\n"
                      "      type: DT_FLOAT\n"
                      "    }\n"
                      "  }\n"
                      "  attr {\n"
                      "    key: \"shape\"\n"
                      "    value {\n"
                      "      shape {\n"
                      "      }\n"
                      "    }\n"
                      "  }\n"
                      "}\n"
                      "node {\n"
                      "  name: \"Rsqrt\"\n"
                      "  op: \"Rsqrt\"\n"
                      "  input: \"input\"\n"
                      "  attr {\n"
                      "    key: \"T\"\n"
                      "    value {\n"
                      "      type: DT_FLOAT\n"
                      "    }\n"
                      "  }\n"
                      "}\n";

        SetupSingleInputSingleOutput({ 2, 2 }, "input", "Rsqrt");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseRsqrt, RsqrtFixture)
{
    RunTest<2>({ 1.f, 4.f, 16.f, 25.f }, { 1.f, 0.5f, 0.25f, 0.2f });
}

BOOST_FIXTURE_TEST_CASE(ParseRsqrtZeroNegative, RsqrtFixture)
{
    RunTest<2>({ 0.f, -0.f, -25.f, -16.f }, { INFINITY, -INFINITY, -NAN, -NAN });
}

BOOST_AUTO_TEST_SUITE_END()