//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct SoftmaxFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SoftmaxFixture()
    {
        m_Prototext = "node {\n"
            "  name: \"blah\"\n"
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
            "  name: \"blah2\"\n"
            "  op: \"Softmax\"\n"
            "  input: \"blah\"\n"
            "  attr {\n"
            "    key: \"T\"\n"
            "    value {\n"
            "      type: DT_FLOAT\n"
            "    }\n"
            "  }\n"
            "}\n";

        SetupSingleInputSingleOutput({ 1, 7 }, "blah", "blah2");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseSoftmax, SoftmaxFixture)
{
    RunTest<2>({ 0, 0, 10000, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 });
}


BOOST_AUTO_TEST_SUITE_END()
