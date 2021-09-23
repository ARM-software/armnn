//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

TEST_SUITE("OnnxParser_Unsqueeze")
{

struct UnsqueezeFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    UnsqueezeFixture(const std::vector<int>& axes,
                     const std::vector<int>& inputShape,
                     const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "Input"
                        output: "Output"
                        op_type: "Unsqueeze"
                        )" + armnnUtils::ConstructIntsAttribute("axes", axes) + R"(
                      }
                      name: "test-model"
                      input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputShape) + R"(
                            }
                          }
                        }
                      }
                      output {
                        name: "Output"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(outputShape) + R"(
                            }
                          }
                        }
                      }
                    })";
    }
};

struct UnsqueezeSingleAxesFixture : UnsqueezeFixture
{
    UnsqueezeSingleAxesFixture() : UnsqueezeFixture({ 0 }, { 2, 3 }, { 1, 2, 3 })
    {
        Setup();
    }
};

struct UnsqueezeMultiAxesFixture : UnsqueezeFixture
{
    UnsqueezeMultiAxesFixture() : UnsqueezeFixture({ 1, 3 }, { 3, 2, 5 }, { 3, 1, 2, 1, 5 })
    {
        Setup();
    }
};

struct UnsqueezeUnsortedAxesFixture : UnsqueezeFixture
{
    UnsqueezeUnsortedAxesFixture() : UnsqueezeFixture({ 3, 0, 1 }, { 2, 5 }, { 1, 1, 2, 1, 5 })
    {
        Setup();
    }
};

struct UnsqueezeScalarFixture : UnsqueezeFixture
{
    UnsqueezeScalarFixture() : UnsqueezeFixture({ 0 }, { }, { 1 })
    {
        Setup();
    }
};

TEST_CASE_FIXTURE(UnsqueezeSingleAxesFixture, "UnsqueezeSingleAxesTest")
{
    RunTest<3, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}});
}

TEST_CASE_FIXTURE(UnsqueezeMultiAxesFixture, "UnsqueezeMultiAxesTest")
{
    RunTest<5, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                   16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                   21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                   26.0f, 27.0f, 28.0f, 29.0f, 30.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                    21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                    26.0f, 27.0f, 28.0f, 29.0f, 30.0f }}});
}

TEST_CASE_FIXTURE(UnsqueezeUnsortedAxesFixture, "UnsqueezeUnsortedAxesTest")
{
    RunTest<5, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f }}});
}

TEST_CASE_FIXTURE(UnsqueezeScalarFixture, "UnsqueezeScalarTest")
{
    RunTest<1, float>({{"Input", { 1.0f }}},
                      {{"Output", { 1.0f }}});
}

struct UnsqueezeInputAxesFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    UnsqueezeInputAxesFixture()
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "Input"
                        input: "Axes"
                        output: "Output"
                        op_type: "Unsqueeze"
                      }
                      initializer {
                          dims: 2
                          data_type: 7
                          int64_data: 0
                          int64_data: 3
                          name: "Axes"
                        }
                      name: "test-model"
                      input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 2
                              }
                              dim {
                                dim_value: 5
                              }
                            }
                          }
                        }
                      }
                      output {
                        name: "Output"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 2
                              }
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 5
                              }
                            }
                          }
                        }
                      }
                    })";
        Setup();
    }
};

TEST_CASE_FIXTURE(UnsqueezeInputAxesFixture, "UnsqueezeInputAxesTest")
{
    RunTest<5, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                   16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                   21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                   26.0f, 27.0f, 28.0f, 29.0f, 30.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                    21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                    26.0f, 27.0f, 28.0f, 29.0f, 30.0f }}});
}

}
