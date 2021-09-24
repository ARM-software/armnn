//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

TEST_SUITE("OnnxParser_Concat")
{

struct ConcatFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ConcatFixture(const std::string& axis,
                  const std::vector<int>& input0Shape,
                  const std::vector<int>& input1Shape,
                  const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "Input0"
                        input: "Input1"
                        output: "Output"
                        op_type: "Concat"
                        attribute {
                          name: "axis"
                          i: )" + axis + R"(
                          type: INT
                        }
                      }
                      name: "concat-model"
                      input {
                        name: "Input0"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(input0Shape) + R"(
                            }
                          }
                        }
                      }
                      input {
                        name: "Input1"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(input1Shape) + R"(
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
        Setup();
    }
};

struct ConcatAxis0Fixture : ConcatFixture
{
    ConcatAxis0Fixture() : ConcatFixture("0", { 1, 3, 2, 5 }, { 1, 3, 2, 5 }, { 2, 3, 2, 5 }) {}
};

struct ConcatAxis1Fixture : ConcatFixture
{
    ConcatAxis1Fixture() : ConcatFixture("1", { 2, 2, 1, 3 }, { 2, 1, 1, 3 }, { 2, 3, 1, 3 }) {}
};

struct ConcatAxis2Fixture : ConcatFixture
{
    ConcatAxis2Fixture() : ConcatFixture("2", { 2, 3, 1, 1 }, { 2, 3, 2, 1 }, { 2, 3, 3, 1 }) {}
};

struct ConcatAxis3Fixture : ConcatFixture
{
    ConcatAxis3Fixture() : ConcatFixture("3", { 1, 3, 2, 2 }, { 1, 3, 2, 2 }, { 1, 3, 2, 4 }) {}
};

struct ConcatNegativeAxisFixture : ConcatFixture
{
    ConcatNegativeAxisFixture() : ConcatFixture("-1", { 1, 2, 5 }, { 1, 2, 3 }, { 1, 2, 8 }) {}
};

TEST_CASE_FIXTURE(ConcatAxis0Fixture, "ConcatAxis0Test")
{
    RunTest<4, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                    21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                    26.0f, 27.0f, 28.0f, 29.0f, 30.0f }},
                       {"Input1", { 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                    36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                                    41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                                    46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                                    51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                                    56.0f, 57.0f, 58.0f, 59.0f, 60.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                                    16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                                    21.0f, 22.0f, 23.0f, 24.0f, 25.0f,
                                    26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                                    31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
                                    36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                                    41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                                    46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                                    51.0f, 52.0f, 53.0f, 54.0f, 55.0f,
                                    56.0f, 57.0f, 58.0f, 59.0f, 60.0f }}});
}

TEST_CASE_FIXTURE(ConcatAxis1Fixture, "ConcatAxis1est")
{
    RunTest<4, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }},
                       {"Input1", { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    13.0f, 14.0f, 15.0f,
                                    7.0f, 8.0f, 9.0f,
                                    10.0f, 11.0f, 12.0f,
                                    16.0f, 17.0f, 18.0f }}});
}

TEST_CASE_FIXTURE(ConcatAxis2Fixture, "ConcatAxis2Test")
{
    RunTest<4, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }},
                       {"Input1", { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }}},
                      {{"Output", { 1.0f, 7.0f, 8.0f,
                                    2.0f, 9.0f, 10.0f,
                                    3.0f, 11.0f, 12.0f,
                                    4.0f, 13.0f, 14.0f,
                                    5.0f, 15.0f, 16.0f,
                                    6.0f, 17.0f, 18.0f }}});
}

TEST_CASE_FIXTURE(ConcatAxis3Fixture, "ConcatAxis3Test")
{
    RunTest<4, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }},
                       {"Input1", { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f }}},
                      {{"Output", { 1.0f, 2.0f, 13.0f, 14.0f,
                                    3.0f, 4.0f, 15.0f, 16.0f,
                                    5.0f, 6.0f, 17.0f, 18.0f,
                                    7.0f, 8.0f, 19.0f, 20.0f,
                                    9.0f, 10.0f, 21.0f, 22.0f,
                                    11.0f, 12.0f, 23.0f, 24.0f }}});
}

TEST_CASE_FIXTURE(ConcatNegativeAxisFixture, "ConcatNegativeAxisTest")
{
    RunTest<3, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f }},
                       {"Input1", { 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 11.0f, 12.0f, 13.0f,
                                    6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 14.0f, 15.0f, 16.0f }}});
}

struct ConcatMultipleInputsFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ConcatMultipleInputsFixture()
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "Input0"
                        input: "Input1"
                        input: "Input2"
                        output: "Output"
                        op_type: "Concat"
                        attribute {
                          name: "axis"
                          i: 1
                          type: INT
                        }
                      }
                      name: "concat-model"
                      input {
                        name: "Input0"
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
                            }
                          }
                        }
                      }
                      input {
                        name: "Input1"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 3
                              }
                            }
                          }
                        }
                      }
                      input {
                        name: "Input2"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 1
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
                                dim_value: 3
                              }
                              dim {
                                dim_value: 6
                              }
                            }
                          }
                        }
                      }
                    })";
        Setup();
    }
};

TEST_CASE_FIXTURE(ConcatMultipleInputsFixture, "ConcatMultipleInputsTest")
{
    RunTest<2, float>({{"Input0", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }},
                       {"Input1", { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f }},
                       {"Input2", { 16.0f, 17.0f, 18.0f }}},
                      {{"Output", { 1.0f, 2.0f, 7.0f, 8.0f, 9.0f, 16.0f,
                                    3.0f, 4.0f, 10.0f, 11.0f, 12.0f, 17.0f,
                                    5.0f, 6.0f, 13.0f, 14.0f, 15.0f, 18.0f }}});
}

}
