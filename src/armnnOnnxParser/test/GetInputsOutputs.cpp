//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "../OnnxParser.hpp"
#include  "ParserPrototxtFixture.hpp"
#include <onnx/onnx.pb.h>
#include "google/protobuf/stubs/logging.h"


using ModelPtr = std::unique_ptr<onnx::ModelProto>;

BOOST_AUTO_TEST_SUITE(OnnxParser)

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


BOOST_FIXTURE_TEST_CASE(GetInput, GetInputsOutputsMainFixture)
{
    ModelPtr model = armnnOnnxParser::OnnxParser::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParser::GetInputs(model);
    BOOST_CHECK_EQUAL(1, tensors.size());
    BOOST_CHECK_EQUAL("Input", tensors[0]);

}

BOOST_FIXTURE_TEST_CASE(GetOutput, GetInputsOutputsMainFixture)
{
    ModelPtr model = armnnOnnxParser::OnnxParser::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParser::GetOutputs(model);
    BOOST_CHECK_EQUAL(1, tensors.size());
    BOOST_CHECK_EQUAL("Output", tensors[0]);
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

BOOST_FIXTURE_TEST_CASE(GetEmptyInputs, GetEmptyInputsOutputsFixture)
{
    ModelPtr model = armnnOnnxParser::OnnxParser::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParser::GetInputs(model);
    BOOST_CHECK_EQUAL(0, tensors.size());
}

BOOST_AUTO_TEST_CASE(GetInputsNullModel)
{
    BOOST_CHECK_THROW(armnnOnnxParser::OnnxParser::LoadModelFromString(""), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(GetOutputsNullModel)
{
    auto silencer = google::protobuf::LogSilencer(); //get rid of errors from protobuf
    BOOST_CHECK_THROW(armnnOnnxParser::OnnxParser::LoadModelFromString("nknnk"), armnn::ParseException);
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

BOOST_FIXTURE_TEST_CASE(GetInputsMultipleInputs, GetInputsMultipleFixture)
{
    ModelPtr model = armnnOnnxParser::OnnxParser::LoadModelFromString(m_Prototext.c_str());
    std::vector<std::string> tensors = armnnOnnxParser::OnnxParser::GetInputs(model);
    BOOST_CHECK_EQUAL(2, tensors.size());
    BOOST_CHECK_EQUAL("Input0", tensors[0]);
    BOOST_CHECK_EQUAL("Input1", tensors[1]);
}



BOOST_AUTO_TEST_SUITE_END()
