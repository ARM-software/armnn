//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"
#include "OnnxParserTestUtils.hpp"

TEST_SUITE("OnnxParser_Reshape")
{
struct ReshapeMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeMainFixture(const std::string& dataType)
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: )" + dataType + R"(
                            shape {
                              dim {
                                dim_value: 4
                              }
                            }
                          }
                        }
                      }
                      input {
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: 7
                             shape {
                               dim {
                                 dim_value: 2
                               }
                             }
                           }
                         }
                       }
                     node {
                         input: "Input"
                         input: "Shape"
                         output: "Output"
                         name: "reshape"
                         op_type: "Reshape"

                      }
                      initializer {
                        dims: 2
                        data_type: 7
                        int64_data: 2
                        int64_data: 2
                        name: "Shape"
                     }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 2
                                   }
                                   dim {
                                       dim_value: 2
                                   }
                               }
                            }
                          }
                       }
                    }
                   opset_import {
                      version: 7
                    })";
    }
};

struct ReshapeRank4Fixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeRank4Fixture(const std::string& dataType)
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input"
                        type {
                          tensor_type {
                            elem_type: )" + dataType + R"(
                            shape {
                              dim {
                                dim_value: 2
                              }
                              dim {
                                dim_value: 2
                              }
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
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: 7
                             shape {
                               dim {
                                 dim_value: 2
                               }
                             }
                           }
                         }
                       }
                     node {
                         input: "Input"
                         input: "Shape"
                         output: "Output"
                         name: "reshape"
                         op_type: "Reshape"

                      }
                      initializer {
                        dims: 2
                        data_type: 7
                        int64_data: 2
                        int64_data: 2
                        name: "Shape"
                     }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 6
                                   }
                                   dim {
                                       dim_value: 6
                                   }
                               }
                            }
                          }
                       }
                    }
                   opset_import {
                      version: 7
                    })";
    }
};

struct ReshapeValidFixture : ReshapeMainFixture
{
    ReshapeValidFixture() : ReshapeMainFixture("1") {
        Setup();
    }
};

struct ReshapeValidRank4Fixture : ReshapeRank4Fixture
{
    ReshapeValidRank4Fixture() : ReshapeRank4Fixture("1") {
        Setup();
    }
};

struct ReshapeInvalidFixture : ReshapeMainFixture
{
    ReshapeInvalidFixture() : ReshapeMainFixture("10") { }
};

TEST_CASE_FIXTURE(ReshapeValidFixture, "ValidReshapeTest")
{
    RunTest<2>({{"Input", { 0.0f, 1.0f, 2.0f, 3.0f }}}, {{"Output", { 0.0f, 1.0f, 2.0f, 3.0f }}});
}

TEST_CASE_FIXTURE(ReshapeValidRank4Fixture, "ValidRank4ReshapeTest")
{
    RunTest<2>(
        {{"Input",
                   {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                    1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f}}},
        {{"Output",
                    {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                     1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f}}});
}

TEST_CASE_FIXTURE(ReshapeInvalidFixture, "IncorrectDataTypeReshape")
{
   CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

struct ReshapeNegativeReshapeFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeNegativeReshapeFixture(const std::vector<int>& inputShape,
                                  const std::vector<int>& shapeInputShape,
                                  const std::vector<int>& outputShape,
                                  const std::string& shape)
        {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name: "onnx-example"
                   graph {
                     name:  "ReshapeGrapn"
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
                      input {
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: 7
                             shape {
                               )" + armnnUtils::ConstructTensorShapeString(shapeInputShape) + R"(
                             }
                           }
                         }
                       }
                     node {
                         input: "Input"
                         input: "Shape"
                         output: "Output"
                         name: "reshape"
                         op_type: "Reshape"
                      }
                      initializer {
                        dims: 2
                        data_type: 7
                        )" + shape + R"(
                        name: "Shape"
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
                    }
                   opset_import {
                      version: 7
                   })";
    }
};

struct ReshapeNegativeReshape1DFixture : ReshapeNegativeReshapeFixture
{
    ReshapeNegativeReshape1DFixture() : ReshapeNegativeReshapeFixture({ 1, 3, 1, 2 }, { 1 }, { 6 }, "int64_data: -1")
    {
        Setup();
    }
};

struct ReshapeNegativeReshape2DFixture : ReshapeNegativeReshapeFixture
{
    ReshapeNegativeReshape2DFixture() : ReshapeNegativeReshapeFixture({ 2, 3, 1, 2 },
                                                                      { 2 },
                                                                      { 2, 6 },
                                                                      "int64_data: -1  int64_data: 6")
    {
        Setup();
    }
};

struct ReshapeNegativeReshape3DFixture : ReshapeNegativeReshapeFixture
{
    ReshapeNegativeReshape3DFixture() : ReshapeNegativeReshapeFixture({ 2, 3, 1, 2 },
                                                                      { 3 },
                                                                      { 3, 1, 4 },
                                                                      "int64_data: 3  int64_data: -1  int64_data: 4")
    {
        Setup();
    }
};

struct ReshapeNegativeReshape4DFixture : ReshapeNegativeReshapeFixture
{
    ReshapeNegativeReshape4DFixture() : ReshapeNegativeReshapeFixture(
        { 2, 3, 1, 2 },
        { 4 },
        { 3, 1, 2, 2 },
        "int64_data: 3  int64_data: 1  int64_data: 2  int64_data: -1")
    {
        Setup();
    }
};

TEST_CASE_FIXTURE(ReshapeNegativeReshape1DFixture, "ReshapeNegativeReshape1DTest")
{
    RunTest<1, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}});
}

TEST_CASE_FIXTURE(ReshapeNegativeReshape2DFixture, "ReshapeNegativeReshape2DTest")
{
    RunTest<2, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}});
}

TEST_CASE_FIXTURE(ReshapeNegativeReshape3DFixture, "ReshapeNegativeReshape3DTest")
{
    RunTest<3, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}});
}

TEST_CASE_FIXTURE(ReshapeNegativeReshape4DFixture, "ReshapeNegativeReshape4DTest")
{
    RunTest<4, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}});
}

struct ReshapeNonConstShapeFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    ReshapeNonConstShapeFixture(const std::vector<int>& inputShape,
                                const std::vector<int>& shapeInputShape,
                                const std::vector<int>& outputShape)
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name: "onnx-example"
                   graph {
                     name:  "ReshapeGrapn"
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
                      input {
                         name: "Shape"
                         type {
                           tensor_type {
                             elem_type: 7
                             shape {
                               )" + armnnUtils::ConstructTensorShapeString(shapeInputShape) + R"(
                             }
                           }
                         }
                       }
                     node {
                         input: "Input"
                         input: "Shape"
                         output: "Output"
                         name: "reshape"
                         op_type: "Reshape"
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
                    }
                   opset_import {
                      version: 7
                   })";
    }
};

struct ReshapeNonConst1DShapeFixture : ReshapeNonConstShapeFixture
{
    ReshapeNonConst1DShapeFixture() : ReshapeNonConstShapeFixture({ 1, 3, 1, 2 }, { 1 }, { 6 })
    {
        Setup();
    }
};

struct ReshapeNonConst2DShapeFixture : ReshapeNonConstShapeFixture
{
    ReshapeNonConst2DShapeFixture() : ReshapeNonConstShapeFixture({ 2, 3, 2, 2 }, { 2 }, { 2, 12 })
    {
        Setup();
    }
};

struct ReshapeInvalidNonConstShapeFixture : ReshapeNonConstShapeFixture
{
    ReshapeInvalidNonConstShapeFixture() : ReshapeNonConstShapeFixture({ 2, 3, 2, 2 }, { 3 }, { 2, 3, 4 })
    {
    }
};

struct ReshapeInvalidDimNonConstShapeFixture : ReshapeNonConstShapeFixture
{
    ReshapeInvalidDimNonConstShapeFixture() : ReshapeNonConstShapeFixture({ 2, 3, 2, 2 }, { 1, 2 }, { 2, 3, 4 })
    {
    }
};

TEST_CASE_FIXTURE(ReshapeNonConst1DShapeFixture, "ReshapeNonConst1DShapeTest")
{
    RunTest<1, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}});
}

TEST_CASE_FIXTURE(ReshapeNonConst2DShapeFixture, "ReshapeNonConst2DShapeTest")
{
    RunTest<2, float>({{"Input", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                   7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                   13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                   19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f }}},
                      {{"Output", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                                    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                                    19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f }}});
}

TEST_CASE_FIXTURE(ReshapeInvalidNonConstShapeFixture, "ReshapeInvalidNonConstShapeTest")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

TEST_CASE_FIXTURE(ReshapeInvalidDimNonConstShapeFixture, "ReshapeInvalidDimNonConstShapeTest")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

}
