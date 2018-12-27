//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct SplitFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SplitFixture() {
        m_Prototext =
            "node { \n"
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
            "  node {"
            "  name: \"splitInput\" \n"
            "  op: \"Const\" \n"
            "attr {\n"
            "   key: \"dtype\" \n"
            "    value {"
            "     type: DT_INT32"
            "    }"
            "}"
            "attr {"
            " key: \"value\"\n"
            "   value { "
            "  tensor {"
            "    dtype: DT_INT32"
            " tensor_shape {"
            "}"
            "int_val: 1"
            "}"
            "}"
            "}"
            "}"
            "node { \n"
            "  name: \"Split\" \n"
            "  op: \"Split\" \n"
            "input: \"graphInput\"\n"
            "input: \"splitInput\"\n"
            "attr { \n "
            "key: \"T\"\n"
            "value {\n"
            "type: DT_FLOAT\n"
            " }\n"
            "}\n"
            "\n"
            "  attr { \n"
            "    key: \"num_or_size_splits\" \n"
            "    value { \n"
            "        i:2 \n "
            "    } \n"
            "  } \n"
            "} \n"
            "node { \n"
            "name: \"Relu_1\"\n"
            "op: \"Relu\"\n"
            "input: \"Split:0\"\n"
            "attr { \n "
            "key: \"T\"\n"
            "value {\n"
            "type: DT_FLOAT\n"
            " }\n"
            "}\n"
            "}\n"
            "node { \n"
            "name: \"Relu_2\"\n"
            "op: \"Relu\"\n"
            "input: \"Split:1\"\n"
            "attr { \n "
            "key: \"T\"\n"
            "value {\n"
            "type: DT_FLOAT\n"
            " }\n"
            "}\n"
            "}\n";

        Setup( { { "graphInput", { 1,  2,  2 , 2} } },
               { "Relu_1", "Relu_2" });
    }
};

BOOST_FIXTURE_TEST_CASE(ParseAxisOneSplitTwo, SplitFixture)
{
    BOOST_TEST(
        (m_Parser->GetNetworkOutputBindingInfo("Relu_1").second.GetShape() == armnn::TensorShape({ 1, 1, 2, 2 })));

    BOOST_TEST(
        (m_Parser->GetNetworkOutputBindingInfo("Relu_2").second.GetShape() == armnn::TensorShape({ 1, 1, 2, 2 })));

    RunTest<4>({ { "graphInput", { -1.0f, -0.5f, 1.25f, -3.0f, 0.0f, 0.5f, -0.75f, 1.75f } } },
               { { "Relu_1", { 0.0f, 0.0f, 1.25f, 0.0f } },
                 { "Relu_2", { 0.0f, 0.5f, 0.0f, 1.75f } } });
}

BOOST_AUTO_TEST_SUITE_END()
