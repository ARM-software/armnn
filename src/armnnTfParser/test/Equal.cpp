//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

    struct EqualFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
    {
        EqualFixture()
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
  op: "Equal"
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
        }
    };

BOOST_FIXTURE_TEST_CASE(ParseEqualUnsupportedBroadcast, EqualFixture)
{
    BOOST_REQUIRE_THROW(Setup({ { "input0", {2, 3} },
                                { "input1", {1, 2, 2, 3} } },
                              { "output" }),
                              armnn::ParseException);
}

struct EqualFixtureAutoSetup : public EqualFixture
{
    EqualFixtureAutoSetup(const armnn::TensorShape& input0Shape,
                          const armnn::TensorShape& input1Shape)
                : EqualFixture()
    {
         Setup({ { "input0", input0Shape },
                 { "input1", input1Shape } },
               { "output" });
    }
};

struct EqualTwoByTwo : public EqualFixtureAutoSetup
{
    EqualTwoByTwo() : EqualFixtureAutoSetup({2,2}, {2,2}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseEqualTwoByTwo, EqualTwoByTwo)
{
    RunComparisonTest<2>({ { "input0", { 1.0f, 2.0f, 3.0f, 2.0f } },
                           { "input1", { 1.0f, 5.0f, 2.0f, 2.0f } } },
                         { { "output", { 1, 0, 0, 1 } } });
}

struct EqualBroadcast1DAnd4D : public EqualFixtureAutoSetup
{
    EqualBroadcast1DAnd4D() : EqualFixtureAutoSetup({1}, {1,1,2,2}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseEqualBroadcast1DToTwoByTwo, EqualBroadcast1DAnd4D)
{
    RunComparisonTest<4>({ { "input0", { 2.0f } },
                           { "input1", { 1.0f, 2.0f, 3.0f, 2.0f } } },
                         { { "output", { 0, 1, 0, 1 } } });
}

struct EqualBroadcast4DAnd1D : public EqualFixtureAutoSetup
{
    EqualBroadcast4DAnd1D() : EqualFixtureAutoSetup({1,1,2,2}, {1}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseEqualBroadcast4DAnd1D, EqualBroadcast4DAnd1D)
{
    RunComparisonTest<4>({ { "input0", { 1.0f, 2.0f, 3.0f, 2.0f } },
                           { "input1", { 3.0f } } },
                         { { "output", { 0, 0, 1, 0 } } });
}

struct EqualMultiDimBroadcast : public EqualFixtureAutoSetup
{
    EqualMultiDimBroadcast() : EqualFixtureAutoSetup({1,1,2,1}, {1,2,1,3}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseEqualMultiDimBroadcast, EqualMultiDimBroadcast)
{
    RunComparisonTest<4>({ { "input0", { 1.0f, 2.0f } },
                           { "input1", { 1.0f, 2.0f, 3.0f,
                                         3.0f, 2.0f, 2.0f } } },
                         { { "output", { 1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 0,
                                         0, 1, 1 } } });
}

BOOST_AUTO_TEST_SUITE_END()
