//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>

#include <iostream>

int main()
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options;    // default options
    IRuntimePtr runtime = IRuntime::Create(options);
    // Parse a TfLite file.
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    try
    {
        INetworkPtr myNetwork = parser->CreateNetworkFromBinaryFile("./simple_conv2d_1_op.tflite");
        // Optimise ArmNN network
        IOptimizedNetworkPtr optNet = Optimize(*myNetwork, { Compute::CpuRef }, runtime->GetDeviceSpec());
        if (!optNet)
        {
            std::cout << "Error: Failed to optimise the input network." << std::endl;
            return 1;
        }
        NetworkId networkId;
        // Load graph into runtime
        Status loaded = runtime->LoadNetwork(networkId, std::move(optNet));
        if (loaded != Status::Success)
        {
            std::cout << "Error: Failed to load the optimized network." << std::endl;
            return 1;
        }

        // Setup the input and output.
        std::vector<armnnTfLiteParser::BindingPointInfo> inputBindings;
        std::vector<std::string> inputTensorNames = parser->GetSubgraphInputTensorNames(0);
        inputBindings.push_back(parser->GetNetworkInputBindingInfo(0, inputTensorNames[0]));

        std::vector<armnnTfLiteParser::BindingPointInfo> outputBindings;
        std::vector<std::string> outputTensorNames = parser->GetSubgraphOutputTensorNames(0);
        outputBindings.push_back(parser->GetNetworkOutputBindingInfo(0, outputTensorNames[0]));
        TensorInfo inputTensorInfo(inputBindings[0].second);
        inputTensorInfo.SetConstant(true);

        // Allocate input tensors
        armnn::InputTensors inputTensors;
        std::vector<float> in_data(inputBindings[0].second.GetNumElements());
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
        armnnTfLiteParser::ITfLiteParser::Destroy(parser.get());
        parser.release();
    }
    catch (const std::exception& e)    // Could be: InvalidArgumentException, ParseException or a FileNotFoundException.
    {
        std::cout << "Unable to create parser for \"./simple_conv2d_1_op.tflite\". Reason: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
