//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct GreaterFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    GreaterFixture()
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
  op: "Greater"
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

BOOST_FIXTURE_TEST_CASE(ParseGreaterUnsupportedBroadcast, GreaterFixture)
{
    BOOST_REQUIRE_THROW(Setup({ { "input0", {2, 3} },
                                { "input1", {1, 2, 2, 3} } },
                              { "output" }),
                        armnn::ParseException);
}

struct GreaterFixtureAutoSetup : public GreaterFixture
{
    GreaterFixtureAutoSetup(const armnn::TensorShape& input0Shape,
                            const armnn::TensorShape& input1Shape)
                : GreaterFixture()
    {
        Setup({ { "input0", input0Shape },
                { "input1", input1Shape } },
              { "output" });
    }
};

struct GreaterFixtureTwoByTwo : public GreaterFixtureAutoSetup
{
    GreaterFixtureTwoByTwo() : GreaterFixtureAutoSetup({2, 2}, {2, 2}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseGreaterTwoByTwo, GreaterFixtureTwoByTwo)
{
    RunComparisonTest<2>({ { "input0", { 1.0f, 2.0f, 3.0f, 4.0f} },
                           { "input1", { 1.0f, 5.0f, 2.0f, 2.0f} } },
                         { { "output", { 0, 0, 1, 1} } });
}

struct GreaterBroadcast1DAnd4D : public GreaterFixtureAutoSetup
{
    GreaterBroadcast1DAnd4D() : GreaterFixtureAutoSetup({1}, {1,1,2,2}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseGreaterBroadcast1DToTwoByTwo, GreaterBroadcast1DAnd4D)
{
    RunComparisonTest<4>({ { "input0", { 2.0f } },
                           { "input1", { 1.0f, 2.0f, 3.0f, 2.0f } } },
                         { { "output", { 1, 0, 0, 0 } } });
}

struct GreaterBroadcast4DAnd1D : public GreaterFixtureAutoSetup
{
    GreaterBroadcast4DAnd1D() : GreaterFixtureAutoSetup({1,1,2,2}, {1}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseGreaterBroadcast4DAnd1D, GreaterBroadcast4DAnd1D)
{
    RunComparisonTest<4>({ { "input0", { 1.0f, 2.0f, 3.0f, 2.0f } },
                           { "input1", { 3.0f } } },
                         { { "output", { 0, 0, 0, 0 } } });
}

struct GreaterMultiDimBroadcast : public GreaterFixtureAutoSetup
{
    GreaterMultiDimBroadcast() : GreaterFixtureAutoSetup({1,1,2,1}, {1,2,1,3}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseGreaterMultiDimBroadcast, GreaterMultiDimBroadcast)
{
    RunComparisonTest<4>({ { "input0", { 1.0f, 2.0f } },
                           { "input1", { 1.0f, 2.0f, 3.0f,
                                         3.0f, 2.0f, 2.0f } } },
                         { { "output", { 0, 0, 0,
                                         1, 0, 0,
                                         0, 0, 0,
                                         0, 0, 0 } } });
}

BOOST_AUTO_TEST_SUITE_END()
