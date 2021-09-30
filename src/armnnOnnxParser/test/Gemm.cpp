//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

TEST_SUITE("OnnxParser_Gemm")
{

struct GemmFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GemmFixture(const std::string& alpha,
                const std::string& beta,
                const std::string& transA,
                const std::string& transB,
                const std::vector<int>& inputAShape,
                const std::vector<int>& inputBShape,
                const std::vector<int>& inputCShape,
                const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "A"
                        input: "B"
                        input: "C"
                        output: "Output"
                        op_type: "Gemm"
                        attribute {
                          name: "alpha"
                          f: )" + alpha + R"(
                          type: FLOAT
                        }
                        attribute {
                          name: "beta"
                          f: )" + beta + R"(
                          type: FLOAT
                        }
                        attribute {
                          name: "transA"
                          i: )" + transA + R"(
                          type: INT
                        }
                        attribute {
                          name: "transB"
                          i: )" + transB + R"(
                          type: INT
                        }
                      }
                      name: "gem-model"
                      input {
                        name: "A"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputAShape) + R"(
                            }
                          }
                        }
                      }
                      input {
                        name: "B"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputBShape) + R"(
                            }
                          }
                        }
                      }
                      input {
                        name: "C"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputCShape) + R"(
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

struct GemmAllAttributesFixture : GemmFixture
{
    GemmAllAttributesFixture() : GemmFixture("0.25", "0.35", "1", "1", { 4, 3 }, { 5, 4 }, { 5 }, { 3, 5 })
    {
        Setup();
    }
};

struct GemmSimpleFixture : GemmFixture
{
    GemmSimpleFixture() : GemmFixture("1", "1", "0", "0", { 3, 4 }, { 4, 5 }, { 5 }, { 3, 5 })
    {
        Setup();
    }
};

struct GemmTransAFixture : GemmFixture
{
    GemmTransAFixture() : GemmFixture("1", "1", "1", "0", { 4, 3 }, { 4, 5 }, { 5 }, { 3, 5 })
    {
        Setup();
    }
};

struct GemmTransBFixture : GemmFixture
{
    GemmTransBFixture() : GemmFixture("1", "1", "0", "1", { 3, 4 }, { 5, 4 }, { 5 }, { 3, 5 })
    {
        Setup();
    }
};

struct GemmParseExceptionFixture : GemmFixture
{
    GemmParseExceptionFixture() : GemmFixture("1", "1", "0", "1", { 3, 4 }, { 5, 4 }, { 3, 5 }, { 3, 5 }) {}
};

TEST_CASE_FIXTURE(GemmAllAttributesFixture, "GemmTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }},
                       {"C", { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f }}},
                      {{"Output", { 15.035f, 45.07f, 75.105f, 105.14f, 135.175f,
                                    12.535f, 38.57f, 64.605f, 90.64f, 116.675f,
                                    10.035f, 32.07f,  54.105f, 76.14f, 98.175f }}});
}

TEST_CASE_FIXTURE(GemmSimpleFixture, "GemmSimpleTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }},
                       {"C", { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f }}},
                      {{"Output", { 332.1f, 374.2f, 416.3f, 458.4f, 500.5f,
                                    196.1f, 222.2f, 248.3f, 274.4f, 300.5f,
                                    60.1f, 70.2f, 80.3f, 90.4f, 100.5f }}});
}

TEST_CASE_FIXTURE(GemmTransAFixture, "GemmTransposeATest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }},
                       {"C", { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f }}},
                      {{"Output", { 180.1f, 210.2f, 240.3f, 270.4f, 300.5f,
                                    146.1f, 172.2f, 198.3f, 224.4f, 250.5f,
                                    112.1f, 134.2f, 156.3f, 178.4f, 200.5f }}});
}

TEST_CASE_FIXTURE(GemmTransBFixture, "GemmTransposeBTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }},
                       {"C", { 0.10f, 0.20f, 0.30f, 0.40f, 0.50f }}},
                      {{"Output", { 100.1f, 268.2f, 436.3f, 604.4f, 772.5f,
                                    60.1f, 164.2f, 268.3f, 372.4f, 476.5f,
                                    20.1f, 60.2f, 100.3f, 140.4f, 180.5f }}});
}

TEST_CASE_FIXTURE(GemmParseExceptionFixture, "GemmParseExceptionTest")
{
    // ParseException because Input C is non-constant and has 2 dimension (should be 1 dimension)
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

struct GemmConstantFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GemmConstantFixture()
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "A"
                        input: "B"
                        input: "C"
                        output: "Output"
                        op_type: "Gemm"
                        attribute {
                          name: "alpha"
                          f: 0.25
                          type: FLOAT
                        }
                        attribute {
                          name: "beta"
                          f: 0.35
                          type: FLOAT
                        }
                        attribute {
                          name: "transA"
                          i: 1
                          type: INT
                        }
                        attribute {
                          name: "transB"
                          i: 1
                          type: INT
                        }
                      }
                      name: "gem-model"
                      initializer {
                        dims: 5
                        dims: 4
                        data_type: 1
                        float_data: 1.0
                        float_data: 2.0
                        float_data: 3.0
                        float_data: 4.0
                        float_data: 5.0
                        float_data: 6.0
                        float_data: 7.0
                        float_data: 8.0
                        float_data: 9.0
                        float_data: 10.0
                        float_data: 11.0
                        float_data: 12.0
                        float_data: 13.0
                        float_data: 14.0
                        float_data: 15.0
                        float_data: 16.0
                        float_data: 17.0
                        float_data: 18.0
                        float_data: 19.0
                        float_data: 20.0
                        name: "B"
                      }
                      initializer {
                        dims: 1
                        dims: 5
                        data_type: 1
                        float_data: 0.1
                        float_data: 0.2
                        float_data: 0.3
                        float_data: 0.4
                        float_data: 0.5
                        name: "C"
                      }
                      input {
                        name: "A"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 4
                              }
                              dim {
                                dim_value: 3
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

TEST_CASE_FIXTURE(GemmConstantFixture, "GemmConstantTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }}},
                      {{"Output", { 15.035f, 45.07f, 75.105f, 105.14f, 135.175f,
                                    12.535f, 38.57f, 64.605f, 90.64f, 116.675f,
                                    10.035f, 32.07f,  54.105f, 76.14f, 98.175f }}});
}

struct GemmConstantSimpleFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GemmConstantSimpleFixture()
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "A"
                        input: "B"
                        input: "C"
                        output: "Output"
                        op_type: "Gemm"
                        attribute {
                          name: "alpha"
                          f: 1
                          type: FLOAT
                        }
                        attribute {
                          name: "beta"
                          f: 1
                          type: FLOAT
                        }
                        attribute {
                          name: "transA"
                          i: 0
                          type: INT
                        }
                        attribute {
                          name: "transB"
                          i: 0
                          type: INT
                        }
                      }
                      name: "gem-model"
                      initializer {
                        dims: 4
                        dims: 5
                        data_type: 1
                        float_data: 1.0
                        float_data: 2.0
                        float_data: 3.0
                        float_data: 4.0
                        float_data: 5.0
                        float_data: 6.0
                        float_data: 7.0
                        float_data: 8.0
                        float_data: 9.0
                        float_data: 10.0
                        float_data: 11.0
                        float_data: 12.0
                        float_data: 13.0
                        float_data: 14.0
                        float_data: 15.0
                        float_data: 16.0
                        float_data: 17.0
                        float_data: 18.0
                        float_data: 19.0
                        float_data: 20.0
                        name: "B"
                      }
                      initializer {
                        dims: 1
                        dims: 5
                        data_type: 1
                        float_data: 0.1
                        float_data: 0.2
                        float_data: 0.3
                        float_data: 0.4
                        float_data: 0.5
                        name: "C"
                      }
                      input {
                        name: "A"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 3
                              }
                              dim {
                                dim_value: 4
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

TEST_CASE_FIXTURE(GemmConstantSimpleFixture, "GemmConstantSimpleTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }}},
                      {{"Output", { 332.1f, 374.2f, 416.3f, 458.4f, 500.5f,
                                    196.1f, 222.2f, 248.3f, 274.4f, 300.5f,
                                    60.1f, 70.2f, 80.3f, 90.4f, 100.5f }}});
}

struct GemmABFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GemmABFixture(const std::string& alpha,
                  const std::string& beta,
                  const std::string& transA,
                  const std::string& transB,
                  const std::vector<int>& inputAShape,
                  const std::vector<int>& inputBShape,
                  const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                    ir_version: 8
                    producer_name: "onnx-example"
                    graph {
                      node {
                        input: "A"
                        input: "B"
                        output: "Output"
                        op_type: "Gemm"
                        attribute {
                          name: "alpha"
                          f: )" + alpha + R"(
                          type: FLOAT
                        }
                        attribute {
                          name: "beta"
                          f: )" + beta + R"(
                          type: FLOAT
                        }
                        attribute {
                          name: "transA"
                          i: )" + transA + R"(
                          type: INT
                        }
                        attribute {
                          name: "transB"
                          i: )" + transB + R"(
                          type: INT
                        }
                      }
                      name: "gem-model"
                      input {
                        name: "A"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputAShape) + R"(
                            }
                          }
                        }
                      }
                      input {
                        name: "B"
                        type {
                          tensor_type {
                            elem_type: 1
                            shape {
                              )" + armnnUtils::ConstructTensorShapeString(inputBShape) + R"(
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

struct GemmAlphaTransAFixture : GemmABFixture
{
    GemmAlphaTransAFixture() : GemmABFixture("0.25", "0.35", "1", "0", { 4, 3 }, { 4, 5 }, { 3, 5 }) {}
};

struct GemmAlphaTransBFixture : GemmABFixture
{
    GemmAlphaTransBFixture() : GemmABFixture("0.25", "0.35", "0", "1", { 3, 4 }, { 5, 4 }, { 3, 5 }) {}
};

TEST_CASE_FIXTURE(GemmAlphaTransAFixture, "GemmAlphaTransATest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }}},
                      {{"Output", { 45.0f, 52.5f, 60.0f, 67.5f, 75.0f,
                                    36.5f, 43.0f, 49.5f, 56.0f, 62.5f,
                                    28.0f, 33.5f, 39.0f, 44.5f, 50.0f }}});
}

TEST_CASE_FIXTURE(GemmAlphaTransBFixture, "GemmAlphaTransBTest")
{
    RunTest<2, float>({{"A", { 12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f,
                               6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f }},
                       {"B", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                               6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                               11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f }}},
                      {{"Output", { 25.0f, 67.0f, 109.0f, 151.0f, 193.0f,
                                    15.0f, 41.0f, 67.0f, 93.0f, 119.0f,
                                    5.0f, 15.0f, 25.0f, 35.0f, 45.0f }}});
}

}
