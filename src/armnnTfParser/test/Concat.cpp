//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ConcatFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit ConcatFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1,
                           unsigned int concatDim)
    {
        m_Prototext = R"(
        node {
          name: "graphInput0"
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
          name: "graphInput1"
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
          name: "concat/axis"
          op: "Const"
          attr {
            key: "dtype"
            value {
              type: DT_INT32
            }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                }
                int_val: )";

        m_Prototext += std::to_string(concatDim);

        m_Prototext += R"(
              }
            }
          }
        }
        node {
          name: "concat"
          op: "ConcatV2"
          input: "graphInput0"
          input: "graphInput1"
          input: "concat/axis"
          attr {
            key: "N"
            value {
              i: 2
            }
          }
          attr {
            key: "T"
            value {
              type: DT_FLOAT
            }
          }
          attr {
            key: "Tidx"
            value {
              type: DT_FLOAT
            }
          }
        }
        )";

        Setup({{"graphInput0", inputShape0 },
               {"graphInput1", inputShape1 }}, {"concat"});
    }
};

struct ConcatFixtureNCHW : ConcatFixture
{
    ConcatFixtureNCHW() : ConcatFixture({ 1, 1, 2, 2 }, { 1, 1, 2, 2 }, 1 ) {}
};

struct ConcatFixtureNHWC : ConcatFixture
{
    ConcatFixtureNHWC() : ConcatFixture({ 1, 1, 2, 2 }, { 1, 1, 2, 2 }, 3 ) {}
};

BOOST_FIXTURE_TEST_CASE(ParseConcatNCHW, ConcatFixtureNCHW)
{
    RunTest<4>({{"graphInput0", {0.0, 1.0, 2.0, 3.0}},
                {"graphInput1", {4.0, 5.0, 6.0, 7.0}}},
               {{"concat", { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 }}});
}

BOOST_FIXTURE_TEST_CASE(ParseConcatNHWC, ConcatFixtureNHWC)
{
    RunTest<4>({{"graphInput0", {0.0, 1.0, 2.0, 3.0}},
                {"graphInput1", {4.0, 5.0, 6.0, 7.0}}},
               {{"concat", { 0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0 }}});
}

struct ConcatFixtureDim1 : ConcatFixture
{
    ConcatFixtureDim1() : ConcatFixture({ 1, 2, 3, 4 }, { 1, 2, 3, 4 }, 1) {}
};

struct ConcatFixtureDim3 : ConcatFixture
{
    ConcatFixtureDim3() : ConcatFixture({ 1, 2, 3, 4 }, { 1, 2, 3, 4 }, 3) {}
};

BOOST_FIXTURE_TEST_CASE(ParseConcatDim1, ConcatFixtureDim1)
{
    RunTest<4>({ { "graphInput0", {  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
                                     12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0 } },
                 { "graphInput1", {  50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0,
                                     62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0 } } },
               { { "concat",      {  0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0,
                                     12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
                                     50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0,
                                     62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0 } } });
}

BOOST_FIXTURE_TEST_CASE(ParseConcatDim3, ConcatFixtureDim3)
{
    RunTest<4>({ { "graphInput0", {  0.0, 1.0, 2.0, 3.0,
                                     4.0, 5.0, 6.0, 7.0,
                                     8.0, 9.0, 10.0, 11.0,
                                     12.0, 13.0, 14.0, 15.0,
                                     16.0, 17.0, 18.0, 19.0,
                                     20.0, 21.0, 22.0, 23.0 } },
                 { "graphInput1", {  50.0, 51.0, 52.0, 53.0,
                                     54.0, 55.0, 56.0, 57.0,
                                     58.0, 59.0, 60.0, 61.0,
                                     62.0, 63.0, 64.0, 65.0,
                                     66.0, 67.0, 68.0, 69.0,
                                     70.0, 71.0, 72.0, 73.0 } } },
               { { "concat",      {  0.0,  1.0,  2.0,  3.0,
                                     50.0, 51.0, 52.0, 53.0,
                                     4.0,  5.0,  6.0,  7.0,
                                     54.0, 55.0, 56.0, 57.0,
                                     8.0,  9.0,  10.0, 11.0,
                                     58.0, 59.0, 60.0, 61.0,
                                     12.0, 13.0, 14.0, 15.0,
                                     62.0, 63.0, 64.0, 65.0,
                                     16.0, 17.0, 18.0, 19.0,
                                     66.0, 67.0, 68.0, 69.0,
                                     20.0, 21.0, 22.0, 23.0,
                                     70.0, 71.0, 72.0, 73.0 } } });
}

BOOST_AUTO_TEST_SUITE_END()