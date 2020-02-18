//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

#include <PrototxtConversions.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct StackFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit StackFixture(const armnn::TensorShape& inputShape0,
                          const armnn::TensorShape& inputShape1,
                          int axis = 0)
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
          op: "Stack"
          input: "input0"
          input: "input1"
          attr {
            key: "axis"
            value {
              i: )";
                m_Prototext += std::to_string(axis);
                m_Prototext += R"(
            }
          }
        })";

        Setup({{"input0", inputShape0 },
               {"input1", inputShape1 }}, {"output"});
    }
};

struct Stack3DFixture : StackFixture
{
    Stack3DFixture() : StackFixture({ 3, 2, 3 }, { 3, 2, 3 }, 3 ) {}
};

BOOST_FIXTURE_TEST_CASE(Stack3D, Stack3DFixture)
{

    RunTest<4>({ { "input0", {  1, 2, 3,
                                4, 5, 6,

                                7, 8, 9,
                                10, 11, 12,

                                13, 14, 15,
                                16, 17, 18 } },
                 { "input1", {  19, 20, 21,
                                22, 23, 24,

                                25, 26, 27,
                                28, 29, 30,

                                31, 32, 33,
                                34, 35, 36 } } },
               { { "output", {  1, 19,
                                2, 20,
                                3, 21,

                                4, 22,
                                5, 23,
                                6, 24,

                                7, 25,
                                8, 26,
                                9, 27,

                                10, 28,
                                11, 29,
                                12, 30,

                                13, 31,
                                14, 32,
                                15, 33,

                                16, 34,
                                17, 35,
                                18, 36 } } });
}

struct Stack3DNegativeAxisFixture : StackFixture
{
    Stack3DNegativeAxisFixture() : StackFixture({ 3, 2, 3 }, { 3, 2, 3 }, -1 ) {}
};

BOOST_FIXTURE_TEST_CASE(Stack3DNegativeAxis, Stack3DNegativeAxisFixture)
{

    RunTest<4>({ { "input0", {  1, 2, 3,
                                4, 5, 6,

                                7, 8, 9,
                                10, 11, 12,

                                13, 14, 15,
                                16, 17, 18 } },
                 { "input1", {  19, 20, 21,
                                22, 23, 24,

                                25, 26, 27,
                                28, 29, 30,

                                31, 32, 33,
                                34, 35, 36 } } },
               { { "output", {  1, 19,
                                2, 20,
                                3, 21,

                                4, 22,
                                5, 23,
                                6, 24,

                                7, 25,
                                8, 26,
                                9, 27,

                                10, 28,
                                11, 29,
                                12, 30,

                                13, 31,
                                14, 32,
                                15, 33,

                                16, 34,
                                17, 35,
                                18, 36 } } });
}

BOOST_AUTO_TEST_SUITE_END()
