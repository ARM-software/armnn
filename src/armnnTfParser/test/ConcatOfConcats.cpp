//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "armnnTfParser/ITfParser.hpp"
#include "ParserPrototxtFixture.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct ConcatOfConcatsFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit ConcatOfConcatsFixture(const armnn::TensorShape& inputShape0, const armnn::TensorShape& inputShape1,
                                    const armnn::TensorShape& inputShape2, const armnn::TensorShape& inputShape3,
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
              name: "graphInput2"
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
              name: "graphInput3"
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
              name: "Relu"
              op: "Relu"
              input: "graphInput0"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Relu_1"
              op: "Relu"
              input: "graphInput1"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Relu_2"
              op: "Relu"
              input: "graphInput2"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
                }
              }
            }
            node {
              name: "Relu_3"
              op: "Relu"
              input: "graphInput3"
              attr {
                key: "T"
                value {
                  type: DT_FLOAT
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
              input: "Relu"
              input: "Relu_1"
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
                  type: DT_INT32
                }
              }
            }
            node {
              name: "concat_1/axis"
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
              name: "concat_1"
              op: "ConcatV2"
              input: "Relu_2"
              input: "Relu_3"
              input: "concat_1/axis"
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
                  type: DT_INT32
                }
              }
            }
            node {
              name: "concat_2/axis"
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
              name: "concat_2"
              op: "ConcatV2"
              input: "concat"
              input: "concat_1"
              input: "concat_2/axis"
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
                  type: DT_INT32
                }
              }
            }
            )";

        Setup({{ "graphInput0", inputShape0 },
               { "graphInput1", inputShape1 },
               { "graphInput2", inputShape2 },
               { "graphInput3", inputShape3}}, {"concat_2"});
    }
};

struct ConcatOfConcatsFixtureNCHW : ConcatOfConcatsFixture
{
    ConcatOfConcatsFixtureNCHW() : ConcatOfConcatsFixture({ 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 },
                                                          { 1, 1, 2, 2 }, 1 ) {}
};

struct ConcatOfConcatsFixtureNHWC : ConcatOfConcatsFixture
{
    ConcatOfConcatsFixtureNHWC() : ConcatOfConcatsFixture({ 1, 1, 2, 2 }, { 1, 1, 2, 2 }, { 1, 1, 2, 2 },
                                                          { 1, 1, 2, 2 }, 3 ) {}
};

BOOST_FIXTURE_TEST_CASE(ParseConcatOfConcatsNCHW, ConcatOfConcatsFixtureNCHW)
{
    RunTest<4>({{"graphInput0", {0.0, 1.0, 2.0, 3.0}},
                {"graphInput1", {4.0, 5.0, 6.0, 7.0}},
                {"graphInput2", {8.0, 9.0, 10.0, 11.0}},
                {"graphInput3", {12.0, 13.0, 14.0, 15.0}}},
               {{"concat_2", { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                     8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 }}});
}

BOOST_FIXTURE_TEST_CASE(ParseConcatOfConcatsNHWC, ConcatOfConcatsFixtureNHWC)
{
    RunTest<4>({{"graphInput0", {0.0, 1.0, 2.0, 3.0}},
                {"graphInput1", {4.0, 5.0, 6.0, 7.0}},
                {"graphInput2", {8.0, 9.0, 10.0, 11.0}},
                {"graphInput3", {12.0, 13.0, 14.0, 15.0}}},
               {{"concat_2", { 0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0,
                                     2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0 }}});
}

BOOST_AUTO_TEST_SUITE_END()
