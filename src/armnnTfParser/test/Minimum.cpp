//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MinimumFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    MinimumFixture()
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
              op: "Minimum"
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

BOOST_FIXTURE_TEST_CASE(ParseMininumUnsupportedBroadcast, MinimumFixture)
{
    BOOST_REQUIRE_THROW(Setup({ { "input0", {2, 3} },
                                { "input1", {1, 2, 2, 3} } },
                              { "output" }),
                        armnn::ParseException);
}

struct MinimumFixtureAutoSetup : public MinimumFixture
{
    MinimumFixtureAutoSetup(const armnn::TensorShape& input0Shape,
                            const armnn::TensorShape& input1Shape)
    : MinimumFixture()
    {
        Setup({ { "input0", input0Shape },
                { "input1", input1Shape } },
              { "output" });
    }
};

struct MinimumFixture4D : public MinimumFixtureAutoSetup
{
    MinimumFixture4D()
    : MinimumFixtureAutoSetup({1, 2, 2, 3}, {1, 2, 2, 3}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMinimum4D, MinimumFixture4D)
{
    RunTest<4>({ { "input0", { 0.0f,  1.0f,  2.0f,
                               3.0f,  4.0f,  5.0f,
                               6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f } },
                 { "input1", { 0.0f, 0.0f, 0.0f,
                               5.0f, 5.0f, 5.0f,
                               7.0f, 7.0f, 7.0f,
                               9.0f, 9.0f, 9.0f } } },
               { { "output", { 0.0f, 0.0f, 0.0f,
                               3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 7.0f,
                               9.0f, 9.0f, 9.0f } } });
}

struct MinimumBroadcastFixture4D : public MinimumFixtureAutoSetup
{
    MinimumBroadcastFixture4D()
    : MinimumFixtureAutoSetup({1, 1, 2, 1}, {1, 2, 1, 3}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMinimumBroadcast4D, MinimumBroadcastFixture4D)
{
    RunTest<4>({ { "input0", { 2.0f,
                               4.0f } },
                 { "input1", { 1.0f, 2.0f, 3.0f,
                               4.0f, 5.0f, 6.0f } } },
               { { "output", { 1.0f, 2.0f, 2.0f,
                               1.0f, 2.0f, 3.0f,
                               2.0f, 2.0f, 2.0f,
                               4.0f, 4.0f, 4.0f } } });
}

struct MinimumBroadcastFixture4D1D : public MinimumFixtureAutoSetup
{
    MinimumBroadcastFixture4D1D()
    : MinimumFixtureAutoSetup({1, 2, 2, 3}, {1}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMinimumBroadcast4D1D, MinimumBroadcastFixture4D1D)
{
    RunTest<4>({ { "input0", { 0.0f,  1.0f,  2.0f,
                               3.0f,  4.0f,  5.0f,
                               6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f } },
                 { "input1", { 5.0f } } },
               { { "output", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               5.0f, 5.0f, 5.0f,
                               5.0f, 5.0f, 5.0f } } });
}

struct MinimumBroadcastFixture1D4D : public MinimumFixtureAutoSetup
{
    MinimumBroadcastFixture1D4D()
    : MinimumFixtureAutoSetup({3}, {1, 2, 2, 3}) {}
};

BOOST_FIXTURE_TEST_CASE(ParseMinimumBroadcast1D4D, MinimumBroadcastFixture1D4D)
{
    RunTest<4>({ { "input0", { 5.0f,  6.0f,  7.0f } },
                 { "input1", { 0.0f,  1.0f,  2.0f,
                               3.0f,  4.0f,  5.0f,
                               6.0f,  7.0f,  8.0f,
                               9.0f, 10.0f, 11.0f } } },
               { { "output", { 0.0f, 1.0f, 2.0f,
                               3.0f, 4.0f, 5.0f,
                               5.0f, 6.0f, 7.0f,
                               5.0f, 6.0f, 7.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
