//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnTfParser/ITfParser.hpp"

#include <ParserPrototxtFixture.hpp>
#include <PrototxtConversions.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(TensorflowParser)

struct MeanFixture : public armnnUtils::ParserPrototxtFixture<armnnTfParser::ITfParser>
{
    explicit MeanFixture(const armnn::TensorShape& inputShape, const armnn::TensorShape& outputShape,
                         const std::vector<unsigned int>& axis, bool keepDims)
    {
        std::string protobufAxisString;
        std::vector<unsigned int> protobufAxis(axis);

        // If no axis range is specified, the reduction is applied to
        // all dimensions of the input tensor
        if (protobufAxis.size() == 0)
        {
            for (unsigned int i = 0; i < inputShape.GetNumDimensions(); ++i)
            {
                protobufAxis.push_back(i);
            }
        }

        for (unsigned int i = 0; i < protobufAxis.size(); ++i)
        {
            protobufAxisString.append(armnnUtils::ConvertInt32ToOctalString(static_cast<int>(protobufAxis[i])));
        }

        m_Prototext = R"(node {
              name: "input"
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
              name: "Const"
              op: "Const"
              attr {
                key: "dtype"
                value {
                  type: DT_INT32
                }
              }
              attr {
                key: "value"
                value { )";

        if (axis.size() == 1)
        {
            m_Prototext.append(R"(      tensor {
                    dtype: DT_INT32
                    tensor_shape {
                    }
                    int_val: )").append(std::to_string(protobufAxis[0])).append(R"(
                  } )");
        }
        else
        {
            m_Prototext.append(R"(      tensor {
                    dtype: DT_INT32
                    tensor_shape {
                      dim {
                        size: 2
                      }
                    }
                    tensor_content: ")").append(protobufAxisString).append(R"("
                  } )");
        }

        m_Prototext.append(R"(    }
              }
            }
            node {
              name: "output"
              op: "Mean"
              input: "input"
              input: "Const"
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
             attr {
               key: "keep_dims"
                 value {
                   b: )").append(keepDims ? "true" : "false").append(R"(
               }
             }
            })");

        SetupSingleInputSingleOutput(inputShape, outputShape, "input", "output");
    }
};

struct MeanNoAxisNoKeepDimsFixture: MeanFixture
{
    MeanNoAxisNoKeepDimsFixture() : MeanFixture({ 2, 3 }, { 1 }, {}, false) {}
};

struct MeanWithAxis0NoKeepDimsFixture: MeanFixture
{
    MeanWithAxis0NoKeepDimsFixture() : MeanFixture({ 2, 3 }, { 3 }, { 0 }, false) {}
};

struct MeanWithAxis1NoKeepDimsFixture: MeanFixture
{
    MeanWithAxis1NoKeepDimsFixture() : MeanFixture({ 2, 3 }, { 2 }, { 1 }, false) {}
};

struct MeanWithAxis0KeepDimsFixture: MeanFixture
{
    MeanWithAxis0KeepDimsFixture() : MeanFixture({ 2, 3 }, { 1, 3 }, { 0 }, true) {}
};

struct MeanWithAxis1KeepDimsFixture: MeanFixture
{
    MeanWithAxis1KeepDimsFixture() : MeanFixture({ 2, 3 }, { 2, 1 }, { 1 }, true) {}
};


BOOST_FIXTURE_TEST_CASE(MeanNoAxisNoKeepDims, MeanNoAxisNoKeepDimsFixture)
{
    RunTest<1>({ { "input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f } } },
               { { "output", { 1.5f } } });
}

BOOST_FIXTURE_TEST_CASE(MeanWithAxis0NoKeepDims, MeanWithAxis0NoKeepDimsFixture)
{
    RunTest<1>({ { "input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f } } },
               { { "output", { 1.5f, 1.5f, 1.5f } } });
}

BOOST_FIXTURE_TEST_CASE(MeanWithAxis1NoKeepDims, MeanWithAxis1NoKeepDimsFixture)
{
    RunTest<1>({ { "input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f } } },
               { { "output", { 1.f, 2.f } } });
}

BOOST_FIXTURE_TEST_CASE(MeanWithAxis0KeepDims, MeanWithAxis0KeepDimsFixture)
{
    RunTest<2>({ { "input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f } } },
               { { "output", { 1.5f, 1.5f, 1.5f } } });
}

BOOST_FIXTURE_TEST_CASE(MeanWithAxis1KeepDims, MeanWithAxis1KeepDimsFixture)
{
    RunTest<2>({ { "input", { 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f } } },
               { { "output", { 1.f, 2.f } } });
}

BOOST_AUTO_TEST_SUITE_END()
