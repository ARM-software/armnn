//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct AdditionFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    AdditionFixture()
    {
        m_Prototext = "node { \n"
            "    name: \"graphInput\" \n"
            "    op: \"Placeholder\" \n"
            "    attr { \n"
            "      key: \"dtype\" \n"
            "      value { \n"
            "        type: DT_FLOAT \n"
            "      } \n"
            "    } \n"
            "    attr { \n"
            "      key: \"shape\" \n"
            "      value { \n"
            "        shape { \n"
            "        } \n"
            "      } \n"
            "    } \n"
            "  } \n"
            "  node { \n"
            "    name: \"softmax1\" \n"
            "    op: \"Softmax\" \n"
            "    input: \"graphInput\" \n"
            "    attr { \n"
            "      key: \"T\" \n"
            "      value { \n"
            "        type: DT_FLOAT \n"
            "      } \n"
            "    } \n"
            "  }\n"
            "  node {\n"
            "    name: \"softmax2\"\n"
            "    op : \"Softmax\"\n"
            "    input: \"graphInput\"\n"
            "    attr { \n"
            "      key: \"T\" \n"
            "      value { \n"
            "        type: DT_FLOAT \n"
            "      } \n"
            "    } \n"
            "  }\n"
            "  node {\n"
            "    name: \"addition\"\n"
            "    op : \"Add\"\n"
            "    input: \"softmax1\"\n"
            "    input: \"softmax2\"\n"
            "    attr { \n"
            "      key: \"T\" \n"
            "      value { \n"
            "        type: DT_FLOAT \n"
            "      } \n"
            "    } \n"
            "  }\n";

        SetupSingleInputSingleOutput({ 1, 7 }, "graphInput", "addition");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseAddition, AdditionFixture)
{
    RunTest<2>({ 0, 0, 10000, 0, 0, 0, 0 }, { 0, 0, 2, 0, 0, 0, 0 });
}


BOOST_AUTO_TEST_SUITE_END()
