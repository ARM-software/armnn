//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct SubFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    SubFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1)
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
  op: "Sub"
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

struct SubFixture4D4D : public SubFixture
{
    SubFixture4D4D() : SubFixture({ 1, 2, 2, 3 }, { 1, 2, 2, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseSub, SubFixture4D4D)
{
    RunTest<4>({ { "input0", { 5.0f,   1.0f,  2.0f,
                               3.0f,   4.0f,  5.0f,
                               6.0f,   7.0f,  8.0f,
                               29.0f, 10.0f, 11.0f } },

                 { "input1", { 0.0f,   1.0f,  3.0f,
                               4.0f,   5.5f,  1.0f,
                               2.0f,  17.0f, 18.0f,
                               19.0f,  1.0f,  3.0f } } },

               { { "output", {  5.0f,    0.0f,  -1.0f,
                               -1.0f,   -1.5f,   4.0f,
                                4.0f,  -10.0f, -10.0f,
                               10.0f,    9.0f,   8.0f } } });
}

struct SubBroadcastFixture4D1D : public SubFixture
{
    SubBroadcastFixture4D1D() : SubFixture({ 1, 2, 2, 3 }, { 1 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseSubBroadcast4D1D, SubBroadcastFixture4D1D)
{
    RunTest<4>({ { "input0", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } },

                 { "input1", { 5.0f } } },

                 { { "output", { -5.0f, -4.0f, -3.0f,
                                 -2.0f, -1.0f,  0.0f,
                                  1.0f,  2.0f,  3.0f,
                                  4.0f,  5.0f,  6.0f } } });
}

struct SubBroadcastFixture1D4D : public SubFixture
{
    SubBroadcastFixture1D4D() : SubFixture({ 1 }, { 1, 2, 2, 3 }) {}
};

BOOST_FIXTURE_TEST_CASE(ParseSubBroadcast1D4D, SubBroadcastFixture1D4D)
{
    RunTest<4>({ { "input0", { 3.0f } },

                 { "input1", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f } } },

                 { { "output", {  3.0f,  2.0f,  1.0f,
                                  0.0f, -1.0f, -2.0f,
                                 -3.0f, -4.0f, -5.0f,
                                 -6.0f, -7.0f, -8.0f } } });
}


BOOST_AUTO_TEST_SUITE_END()