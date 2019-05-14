//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <iostream>
#include "armnn/ArmNN.hpp"

/// A simple example of using the ArmNN SDK API. In this sample, the users single input number is multiplied by 1.0f
/// using a fully connected layer with a single neuron to produce an output number that is the same as the input.
int main()
{
    using namespace armnn;

    float number;
    std::cout << "Please enter a number: " << std::endl;
    std::cin >> number;

    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    INetworkPtr myNetwork = INetwork::Create();

    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    float weightsData[] = {1.0f}; // Identity
    TensorInfo weightsInfo(TensorShape({1, 1}), DataType::Float32);
    armnn::ConstTensor weights(weightsInfo, weightsData);
    IConnectableLayer *fullyConnected = myNetwork->AddFullyConnectedLayer(fullyConnectedDesc,
                                                                          weights,
                                                                          EmptyOptional(),
                                                                          "fully connected");

    IConnectableLayer *InputLayer = myNetwork->AddInputLayer(0);
    IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(0));
    fullyConnected->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    IRuntimePtr run = IRuntime::Create(options);

    //Set the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    fullyConnected->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {Compute::CpuRef}, run->GetDeviceSpec());

    // Load graph into runtime
    run->LoadNetwork(networkIdentifier, std::move(optNet));

    //Creates structures for inputs and outputs.
    std::vector<float> inputData{number};
    std::vector<float> outputData(1);


    armnn::InputTensors inputTensors{{0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0),
                                                            inputData.data())}};
    armnn::OutputTensors outputTensors{{0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0),
                                                         outputData.data())}};

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    std::cout << "Your number was " << outputData[0] << std::endl;
    return 0;

}
