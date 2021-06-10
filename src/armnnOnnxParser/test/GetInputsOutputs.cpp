//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../OnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"
#include <onnx/onnx.pb.h>
#include "google/protobuf/stubs/logging.h"

using ModelPtr = std::unique_ptr<onnx::ModelProto>;

TEST_SUITE("OnnxParser_GetInputsOutputs")
{
struct GetInputsOutputsMainFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    explicit GetInputsOutputsMainFixture()
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
                                dim_value: 4
                              }
                            }
                          }
                        }
                      }
                     node {
                         input: "Input"
                         output: "Output"
                         name: "ActivationLayer"
                         op_type: "Relu"
                    }
                      output {
                          name: "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                   dim {
                                       dim_value: 4
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


TEST_CASE_FIXTURE(GetInputsOutputsMainFixture, "GetInput")
{
    ModelPtr model = armnnOnnxParser::OnnxParserImpl::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParserImpl::GetInputs(model);
    CHECK_EQ(1, tensors.size());
    CHECK_EQ("Input", tensors[0]);

}

TEST_CASE_FIXTURE(GetInputsOutputsMainFixture, "GetOutput")
{
    ModelPtr model = armnnOnnxParser::OnnxParserImpl::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParserImpl::GetOutputs(model);
    CHECK_EQ(1, tensors.size());
    CHECK_EQ("Output", tensors[0]);
}

struct GetEmptyInputsOutputsFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GetEmptyInputsOutputsFixture()
    {
        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK "
                   producer_version:  "2.5.1 "
                   domain:  "ai.cntk "
                   model_version: 1
                   graph {
                     name:  "CNTKGraph "
                     node {
                        output:  "Output"
                        attribute {
                          name: "value"
                          t {
                              dims: 7
                              data_type: 1
                              float_data: 0.0
                              float_data: 1.0
                              float_data: 2.0
                              float_data: 3.0
                              float_data: 4.0
                              float_data: 5.0
                              float_data: 6.0

                          }
                          type: 1
                        }
                        name:  "constantNode"
                        op_type:  "Constant"
                      }
                      output {
                          name:  "Output"
                          type {
                             tensor_type {
                               elem_type: 1
                               shape {
                                 dim {
                                    dim_value: 7
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

TEST_CASE_FIXTURE(GetEmptyInputsOutputsFixture, "GetEmptyInputs")
{
    ModelPtr model = armnnOnnxParser::OnnxParserImpl::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParserImpl::GetInputs(model);
    CHECK_EQ(0, tensors.size());
}

TEST_CASE("GetInputsNullModel")
{
    CHECK_THROWS_AS(armnnOnnxParser::OnnxParserImpl::LoadModelFromString(""), armnn::InvalidArgumentException);
}

TEST_CASE("GetOutputsNullModel")
{
    auto silencer = google::protobuf::LogSilencer(); //get rid of errors from protobuf
    CHECK_THROWS_AS(armnnOnnxParser::OnnxParserImpl::LoadModelFromString("nknnk"), armnn::ParseException);
}

struct GetInputsMultipleFixture : public armnnUtils::ParserPrototxtFixture<armnnOnnxParser::IOnnxParser>
{
    GetInputsMultipleFixture() {

        m_Prototext = R"(
                   ir_version: 3
                   producer_name:  "CNTK"
                   producer_version:  "2.5.1"
                   domain:  "ai.cntk"
                   model_version: 1
                   graph {
                     name:  "CNTKGraph"
                     input {
                        name: "Input0"
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
                                dim_value: 4
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
                                   dim_value: 4
                                 }
                             }
                           }
                         }
                       }
                       node {
                            input: "Input0"
                            input: "Input1"
                            output: "Output"
                            name: "addition"
                            op_type: "Add"
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
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 1
                                       }
                                       dim {
                                           dim_value: 4
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

TEST_CASE_FIXTURE(GetInputsMultipleFixture, "GetInputsMultipleInputs")
{
    ModelPtr model = armnnOnnxParser::OnnxParserImpl::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParserImpl::GetInputs(model);
    CHECK_EQ(2, tensors.size());
    CHECK_EQ("Input0", tensors[0]);
    CHECK_EQ("Input1", tensors[1]);
}

}
