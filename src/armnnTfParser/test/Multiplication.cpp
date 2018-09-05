//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MultiplicationFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiplicationFixture()
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
            "    name: \"multiplication\"\n"
            "    op : \"Mul\"\n"
            "    input: \"softmax1\"\n"
            "    input: \"softmax2\"\n"
            "    attr { \n"
            "      key: \"T\" \n"
            "      value { \n"
            "        type: DT_FLOAT \n"
            "      } \n"
            "    } \n"
            "  }\n";

        SetupSingleInputSingleOutput({ 1, 7 }, "graphInput", "multiplication");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseMultiplication, MultiplicationFixture)
{
    RunTest<2>({ 0, 0, 10000, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0 });
}

struct MultiplicationBroadcastFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MultiplicationBroadcastFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1)
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
  op: "Mul"
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

struct MultiplicationBroadcastFixture4D1D : public MultiplicationBroadcastFixture
{
    MultiplicationBroadcastFixture4D1D() : MultiplicationBroadcastFixture({ 1, 2, 2, 3 }, { 1 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMultiplicationBroadcast4D1D, MultiplicationBroadcastFixture4D1D)
{
    RunTest<4>({ { "input0", { 0.0f,  1.0f,  2.0f,
                               3.0f,  4.0f,  5.0f,
                               6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f } },
                 { "input1", { 5.0f } } },
               { { "output", { 0.0f,  5.0f, 10.0f,
                              15.0f, 20.0f, 25.0f,
                              30.0f, 35.0f, 40.0f,
                              45.0f, 50.0f, 55.0f } } });
}

struct MultiplicationBroadcastFixture1D4D : public MultiplicationBroadcastFixture
{
    MultiplicationBroadcastFixture1D4D() : MultiplicationBroadcastFixture({ 1 }, { 1, 2, 2, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMultiplicationBroadcast1D4D, MultiplicationBroadcastFixture1D4D)
{
    RunTest<4>({ { "input0", { 3.0f } },
                 { "input1", { 0.0f,  1.0f,  2.0f,
                               3.0f,  4.0f,  5.0f,
                               6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f } } },
               { { "output", { 0.0f,  3.0f,  6.0f,
                               9.0f, 12.0f, 15.0f,
                              18.0f, 21.0f, 24.0f,
                              27.0f, 30.0f, 33.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
