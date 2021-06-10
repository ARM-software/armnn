//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_Pooling")
{
struct PoolingMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    PoolingMainFixture(const std::string& dataType, const std::string& op)
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
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "Pooling"
                         op_type: )" + op + R"(
                         attribute {
                           name: "kernel_shape"
                           ints: 2
                           ints: 2
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "pads"
                           ints: 0
                           ints: 0
                           ints: 0
                           ints: 0
                           type: INTS
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
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 1
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

struct MaxPoolValidFixture : PoolingMainFixture
{
    MaxPoolValidFixture() : PoolingMainFixture("1", "\"MaxPool\"") {
        Setup();
    }
};

struct MaxPoolInvalidFixture : PoolingMainFixture
{
    MaxPoolInvalidFixture() : PoolingMainFixture("10", "\"MaxPool\"") { }
};

TEST_CASE_FIXTURE(MaxPoolValidFixture, "ValidMaxPoolTest")
{
    RunTest<4>({{"Input", {1.0f, 2.0f, 3.0f, -4.0f}}}, {{"Output", {3.0f}}});
}

struct AvgPoolValidFixture : PoolingMainFixture
{
    AvgPoolValidFixture() : PoolingMainFixture("1", "\"AveragePool\"") {
        Setup();
    }
};

struct PoolingWithPadFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    PoolingWithPadFixture()
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
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 1
                              }
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "Pooling"
                         op_type: "AveragePool"
                         attribute {
                           name: "kernel_shape"
                           ints: 4
                           ints: 4
                           type: INTS
                         }
                         attribute {
                           name: "strides"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "pads"
                           ints: 1
                           ints: 1
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         attribute {
                           name: "count_include_pad"
                           i: 1
                           type: INT
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
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 1
                                   }
                               }
                            }
                        }
                        }
                    }
                   opset_import {
                      version: 7
                    })";
        Setup();
    }
};

TEST_CASE_FIXTURE(AvgPoolValidFixture, "AveragePoolValid")
{
    RunTest<4>({{"Input", {1.0f, 2.0f, 3.0f, -4.0f}}}, {{"Output", {0.5}}});
}

TEST_CASE_FIXTURE(PoolingWithPadFixture, "ValidAvgWithPadTest")
{
    RunTest<4>({{"Input", {1.0f, 2.0f, 3.0f, -4.0f}}}, {{"Output", {1.0/8.0}}});
}

struct GlobalAvgFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GlobalAvgFixture()
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
                            elem_type: 1
                            shape {
                              dim {
                                dim_value: 1
                              }
                              dim {
                                dim_value: 2
                              }
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
                     node {
                         input: "Input"
                         output: "Output"
                         name: "Pooling"
                         op_type: "GlobalAveragePool"
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
                                       dim_value: 2
                                   }
                                   dim {
                                       dim_value: 1
                                   }
                                   dim {
                                       dim_value: 1
                                   }
                               }
                            }
                        }
                        }
                    }
                   opset_import {
                      version: 7
                    })";
        Setup();
    }
};

TEST_CASE_FIXTURE(GlobalAvgFixture, "GlobalAvgTest")
{
    RunTest<4>({{"Input", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}}}, {{"Output", {10/4.0, 26/4.0}}});
}

TEST_CASE_FIXTURE(MaxPoolInvalidFixture, "IncorrectDataTypeMaxPool")
{
   CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

}
