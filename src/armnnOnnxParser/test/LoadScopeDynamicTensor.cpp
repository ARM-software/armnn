//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "ParserPrototxtFixture.hpp"

TEST_SUITE("OnnxParser_LoadScopeDynamicTensor")
{

struct DynamicBatchTensorFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    DynamicBatchTensorFixture()
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
                                dim_value: 0
                              }
                              dim {
                                dim_value: 1
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
                        name: "Weight"
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
                                dim_value: 3
                              }
                              dim {
                                dim_value: 3
                              }
                            }
                          }
                        }
                      }
                      initializer {
                          dims: 1
                          dims: 1
                          dims: 3
                          dims: 3
                          data_type: 1
                          float_data: 2
                          float_data: 1
                          float_data: 0
                          float_data: 6
                          float_data: 2
                          float_data: 1
                          float_data: 4
                          float_data: 1
                          float_data: 2
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
                           ints: 3
                           ints: 3
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
                           i: 1
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
                                       dim_value: 0
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

TEST_CASE_FIXTURE(DynamicBatchTensorFixture, "DynamicBatchTensorTest")
{
    Setup({{"Input", armnn::TensorShape({1, 1, 3, 3})}});
    RunTest<4>({{"Input", {1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0}}},
              {{"Output", {1.0 * 2 + 2.0 * 1 + 3.0 * 0 +
                           4.0 * 6 + 5.0 * 2 + 6.0 * 1 +
                           7.0 * 4 + 8.0 * 1 + 9.0 * 2}}});
}

TEST_CASE_FIXTURE(DynamicBatchTensorFixture, "TensorShapeNotSpecifiedTest")
{
    CHECK_THROWS_AS(Setup(), armnn::ParseException);
}

TEST_CASE_FIXTURE(DynamicBatchTensorFixture, "IncorrectInputNameTest")
{
    CHECK_THROWS_AS(Setup({{"Incorrect", armnn::TensorShape({1, 1, 3, 3})}}), armnn::ParseException);
}

TEST_CASE_FIXTURE(DynamicBatchTensorFixture, "IncorrectBatchTensorTest")
{
    Setup({{"Input", armnn::TensorShape({2, 1, 3, 3}) }});
    CHECK_THROWS_AS(RunTest<4>({{"Input", { 1.0, 2.0, 3.0,
                                            4.0, 5.0, 6.0,
                                            7.0, 8.0, 9.0 }}},
                               {{"Output", {1.0 * 2 + 2.0 * 1 + 3.0 * 0 +
                                            4.0 * 6 + 5.0 * 2 + 6.0 * 1 +
                                            7.0 * 4 + 8.0 * 1 + 9.0 * 2 }}}), armnn::Exception);

}

}
