//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>
#include <iostream>

int main()
{
    // Raw protobuf text for a single layer CONV2D model.
    std::string m_Prototext = R"(
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

    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options;    // default options
    IRuntimePtr runtime = IRuntime::Create(options);
    // Create the parser.
    armnnOnnxParser::IOnnxParserPtr parser = armnnOnnxParser::IOnnxParser::Create();
    try
    {
        // Parse the proto text.
        armnn::INetworkPtr network = parser->CreateNetworkFromString(m_Prototext);
        auto optimized             = Optimize(*network, { armnn::Compute::CpuRef }, runtime->GetDeviceSpec());
        if (!optimized)
        {
            std::cout << "Error: Failed to optimise the input network." << std::endl;
            return 1;
        }
        armnn::NetworkId networkId;
        std::string errorMsg;
        Status status = runtime->LoadNetwork(networkId, std::move(optimized), errorMsg);
        if (status != Status::Success)
        {
            std::cout << "Error: Failed to load the optimized network." << std::endl;
            return -1;
        }

        // Setup the input and output.
        std::vector<armnnOnnxParser::BindingPointInfo> inputBindings;
        // Coz we know the model we know the input tensor is called Input and output is Output.
        inputBindings.push_back(parser->GetNetworkInputBindingInfo("Input"));
        std::vector<armnnOnnxParser::BindingPointInfo> outputBindings;
        outputBindings.push_back(parser->GetNetworkOutputBindingInfo("Output"));
        // Allocate input tensors
        armnn::InputTensors inputTensors;
        std::vector<float> in_data(inputBindings[0].second.GetNumElements());
        TensorInfo inputTensorInfo(inputBindings[0].second);
        inputTensorInfo.SetConstant(true);
        // Set some kind of values in the input.
        for (int i = 0; i < inputBindings[0].second.GetNumElements(); i++)
        {
            in_data[i] = 1.0f + i;
        }
        inputTensors.push_back({ inputBindings[0].first, armnn::ConstTensor(inputTensorInfo, in_data.data()) });

        // Allocate output tensors
        armnn::OutputTensors outputTensors;
        std::vector<float> out_data(outputBindings[0].second.GetNumElements());
        outputTensors.push_back({ outputBindings[0].first, armnn::Tensor(outputBindings[0].second, out_data.data()) });

        runtime->EnqueueWorkload(networkId, inputTensors, outputTensors);
        runtime->UnloadNetwork(networkId);
        // We're finished with the parser.
        armnnOnnxParser::IOnnxParser::Destroy(parser.get());
        parser.release();
    }
    catch (const std::exception& e)    // Could be an InvalidArgumentException or a ParseException.
    {
        std::cout << "Unable to create parser for the passed protobuf string. Reason: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
