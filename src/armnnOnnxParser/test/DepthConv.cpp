//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_DepthConv")
{
struct SimpleDepthConv2DFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    SimpleDepthConv2DFixture()
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
                                dim_value: 3
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
                      input {
                        name: "Weight"
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
                      initializer {
                          dims: 3
                          dims: 1
                          dims: 2
                          dims: 2
                          data_type: 1
                          float_data: 1
                          float_data: 1
                          float_data: 1
                          float_data: 1
                          float_data: 2
                          float_data: 2
                          float_data: 2
                          float_data: 2
                          float_data: 3
                          float_data: 3
                          float_data: 3
                          float_data: 3
                          name: "Weight"
                        }
                      node {
                         input: "Input"
                         input: "Weight"
                         output: "Output"
                         name: "Convolution"
                         op_type: "Conv"
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
                           name: "auto_pad"
                           s: "VALID"
                           type: STRING
                         }
                         attribute {
                           name: "group"
                           i: 3
                           type: INT
                         }
                         attribute {
                           name: "dilations"
                           ints: 1
                           ints: 1
                           type: INTS
                         }
                         doc_string: ""
                         domain: ""
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


TEST_CASE_FIXTURE(SimpleDepthConv2DFixture, "ValidDepthConvTest")
{
    RunTest<4>({{"Input", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
               {{"Output", { 10, 52, 126 }}});
}

}
