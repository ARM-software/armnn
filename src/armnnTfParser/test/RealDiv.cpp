//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct DivisionFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    DivisionFixture()
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
                      "    name: \"division\"\n"
                      "    op : \"RealDiv\"\n"
                      "    input: \"softmax1\"\n"
                      "    input: \"softmax2\"\n"
                      "    attr { \n"
                      "      key: \"T\" \n"
                      "      value { \n"
                      "        type: DT_FLOAT \n"
                      "      } \n"
                      "    } \n"
                      "  }\n";

        SetupSingleInputSingleOutput({ 4, 1 }, "graphInput", "division");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseDivision, DivisionFixture)
{
    RunTest<2>({ 2, 1.0f, 3, 1 }, { 1, 1.0f, 1, 1});
}

struct DivisionBroadcastFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    DivisionBroadcastFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1)
    {
        m_Prototext = R"(
                 node {
                   name: "input0"
                   op: "Placeholder"
                   attr {
                     key: "dtype"
                     value {
                       type: DT_FLOAT
                     }
                   }
                   attr {
                     key: "shape"
                     value {
                       shape {
                       }
                     }
                   }
                 }
                 node {
                   name: "input1"
                   op: "Placeholder"
                   attr {
                     key: "dtype"
                     value {
                       type: DT_FLOAT
                     }
                   }
                   attr {
                     key: "shape"
                     value {
                       shape {
                       }
                     }
                   }
                 }
                 node {
                   name: "output"
                   op: "RealDiv"
                   input: "input0"
                   input: "input1"
                   attr {
                     key: "T"
                     value {
                       type: DT_FLOAT
                     }
                   }
                 }
                 )";

        Setup({ { "input0", inputShape0 },
                { "input1", inputShape1 } },
              { "output" });
    }
};
struct DivisionBroadcastFixture4D1D : public DivisionBroadcastFixture
{
    DivisionBroadcastFixture4D1D() : DivisionBroadcastFixture({ 1, 2, 2, 3 }, { 1 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseDivisionBroadcast4D1D, DivisionBroadcastFixture4D1D)
{
    RunTest<4>({ { "input0", { 0.0f, 100.0f, 2.0f,
                               3.0f, 250.0f, 15.0f,
                               33.0f, 60.0f, 5.0f,
                               35.0f, 10.0f, 55.0f } },
                 { "input1", { 5.0f } } },
               { { "output", { 0, 20.0f, 0.4f,
                               0.6f, 50.0f, 3.0f,
                               6.6f, 12.0f, 1.0f,
                               7.0f, 2.0f, 11.0f } } });
}

BOOST_FIXTURE_TEST_CASE(ParseDivideByZeroBroadcast4D1D, DivisionBroadcastFixture4D1D)
{
    float Inf = std::numeric_limits<float>::infinity();
    float NaN = std::numeric_limits<float>::quiet_NaN();

    RunTest<4>({ { "input0", { 0.0f,  -100.0f,  2.0f,
                               3.0f,  -250.0f,  15.0f,
                               33.0f,  -0,  5.0f,
                               35.0f, -10.0f, 55.0f } },
                 { "input1", { 0 } } },
               { { "output", { NaN, -Inf, Inf,
                               Inf, -Inf, Inf,
                               Inf, NaN, Inf,
                               Inf, -Inf, Inf } } });
}

BOOST_AUTO_TEST_SUITE_END()
