//
// Copyright © 2020-2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>

#include <iostream>

/// A simple example of using the ArmNN SDK API with the standalone sample dynamic backend.
/// In this example, an addition layer is used to add 2 input tensors to produce a result output tensor.
int main()
{
    using namespace armnn;

    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    INetworkPtr myNetwork = INetwork::Create();

    IConnectableLayer* input0 = myNetwork->AddInputLayer(0);
    IConnectableLayer* input1 = myNetwork->AddInputLayer(1);
    IConnectableLayer* add    = myNetwork->AddElementwiseBinaryLayer(BinaryOperation::Add);
    IConnectableLayer* output = myNetwork->AddOutputLayer(0);

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo tensorInfo(TensorShape({2, 1}), DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    armnn::IRuntimePtr run(armnn::IRuntime::Create(options));

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {"SampleDynamic"}, run->GetDeviceSpec());
    if (!optNet)
    {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        return 1;
    }

    // Load graph into runtime
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    // input data
    std::vector<float> input0Data
        {
            5.0f, 3.0f
        };
    std::vector<float> input1Data
        {
            10.0f, 8.0f
        };
    std::vector<float> outputData(2);

    TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
        {
            {0,armnn::ConstTensor(inputTensorInfo, input0Data.data())},
            {1,armnn::ConstTensor(inputTensorInfo, input1Data.data())}
        };
    OutputTensors outputTensors
        {
            {0,armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData.data())}
        };

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    std::cout << "Addition operator result is {" << outputData[0] << "," << outputData[1] << "}" << std::endl;
    return 0;
}
