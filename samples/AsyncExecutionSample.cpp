//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>

#include <iostream>
#include <thread>

/// A simple example of using the ArmNN SDK API to run a network multiple times with different inputs in an asynchronous
/// manner.
///
/// Background info: The usual runtime->EnqueueWorkload, which is used to trigger the execution of a network, is not
///                  thread safe. Each workload has memory assigned to it which would be overwritten by each thread.
///                  Before we added support for this you had to load a network multiple times to execute it at the
///                  same time. Every time a network is loaded, it takes up memory on your device. Making the
///                  execution thread safe helps to reduce the memory footprint for concurrent executions significantly.
///                  This example shows you how to execute a model concurrently (multiple threads) while still only
///                  loading it once.
///
/// As in most of our simple samples, the network in this example will ask the user for a single input number for each
/// execution of the network.
/// The network consists of a single fully connected layer with a single neuron. The neurons weight is set to 1.0f
/// to produce an output number that is the same as the input.
int main()
{
    using namespace armnn;

    // The first part of this code is very similar to the SimpleSample.cpp you should check it out for comparison
    // The interesting part starts when the graph is loaded into the runtime

    std::vector<float> inputs;
    float number1;
    std::cout << "Please enter a number for the first iteration: " << std::endl;
    std::cin >> number1;
    float number2;
    std::cout << "Please enter a number for the second iteration: " << std::endl;
    std::cin >> number2;

    // Turn on logging to standard output
    // This is useful in this sample so that users can learn more about what is going on
    ConfigureLogging(true, false, LogSeverity::Warning);

    // Construct ArmNN network
    NetworkId networkIdentifier;
    INetworkPtr myNetwork = INetwork::Create();

    float weightsData[] = {1.0f}; // Identity
    TensorInfo weightsInfo(TensorShape({1, 1}), DataType::Float32, 0.0f, 0, true);
    weightsInfo.SetConstant();
    ConstTensor weights(weightsInfo, weightsData);

    // Constant layer that now holds weights data for FullyConnected
    IConnectableLayer* const constantWeightsLayer = myNetwork->AddConstantLayer(weights, "const weights");

    FullyConnectedDescriptor fullyConnectedDesc;
    IConnectableLayer* const fullyConnectedLayer = myNetwork->AddFullyConnectedLayer(fullyConnectedDesc,
                                                                                     "fully connected");
    IConnectableLayer* InputLayer  = myNetwork->AddInputLayer(0);
    IConnectableLayer* OutputLayer = myNetwork->AddOutputLayer(0);

    InputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    constantWeightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    fullyConnectedLayer->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    IRuntimePtr run = IRuntime::Create(options);

    //Set the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
    constantWeightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {Compute::CpuRef}, run->GetDeviceSpec());
    if (!optNet)
    {
        // This shouldn't happen for this simple sample, with reference backend.
        // But in general usage Optimize could fail if the hardware at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        return 1;
    }

    // Load graph into runtime.
    std::string errmsg; // To hold an eventual error message if loading the network fails
    // Add network properties to enable async execution. The MemorySource::Undefined variables indicate
    // that neither inputs nor outputs will be imported. Importing will be covered in another example.
    armnn::INetworkProperties networkProperties(true, MemorySource::Undefined, MemorySource::Undefined);
    run->LoadNetwork(networkIdentifier,
                     std::move(optNet),
                     errmsg,
                     networkProperties);

    // Creates structures for inputs and outputs. A vector of float for each execution.
    std::vector<std::vector<float>> inputData{{number1}, {number2}};
    std::vector<std::vector<float>> outputData;
    outputData.resize(2, std::vector<float>(1));

    inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo.SetConstant(true);
    std::vector<InputTensors> inputTensors
    {
        {{0, armnn::ConstTensor(inputTensorInfo, inputData[0].data())}},
        {{0, armnn::ConstTensor(inputTensorInfo, inputData[1].data())}}
    };
    std::vector<OutputTensors> outputTensors
    {
        {{0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData[0].data())}},
        {{0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), outputData[1].data())}}
    };

    // Lambda function to execute the network. We use it as thread function.
    auto execute = [&](unsigned int executionIndex)
    {
        auto memHandle = run->CreateWorkingMemHandle(networkIdentifier);
        run->Execute(*memHandle, inputTensors[executionIndex], outputTensors[executionIndex]);
    };

    // Prepare some threads and let each execute the network with a different input
    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < inputTensors.size(); ++i)
    {
        threads.emplace_back(std::thread(execute, i));
    }

    // Wait for the threads to finish
    for (std::thread& t : threads)
    {
        if(t.joinable())
        {
            t.join();
        }
    }

    std::cout << "Your numbers were " << outputData[0][0] << " and " << outputData[1][0] << std::endl;
    return 0;

}
