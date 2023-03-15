//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>

#include <iostream>

// A simple example application to show the usage of Memory Management Pre Importing of Inputs and Outputs. In this
// sample, the users single input number is added to itself using an add layer and outputted to console as a number
// that is double the input. The code does not use EnqueueWorkload but instead uses runtime->Execute

int main()
{
    using namespace armnn;

    float number;
    std::cout << "Please enter a number: " << std::endl;
    std::cin >> number;

    // Turn on logging to standard output
    // This is useful in this sample so that users can learn more about what is going on
    armnn::ConfigureLogging(true, false, LogSeverity::Info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr               runtime(armnn::IRuntime::Create(options));

    armnn::NetworkId   networkIdentifier1 = 0;

    armnn::INetworkPtr testNetwork(armnn::INetwork::Create());
    auto inputLayer1 = testNetwork->AddInputLayer(0, "input 1 layer");
    auto inputLayer2 = testNetwork->AddInputLayer(1, "input 2 layer");
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    auto addLayer = testNetwork->AddAdditionLayer("add layer");
    ARMNN_NO_DEPRECATE_WARN_END
    auto outputLayer = testNetwork->AddOutputLayer(2, "output layer");

    // Set the tensors in the network.
    TensorInfo tensorInfo{{4}, armnn::DataType::Float32};

    inputLayer1->GetOutputSlot(0).Connect(addLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    inputLayer2->GetOutputSlot(0).Connect(addLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    addLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    addLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Set preferred backend to CpuRef
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };

    // To hold an eventual error message if loading the network fails
    std::string er;

    // Initialize network properties with asyncEnabled and MemorySources != MemorySource::Undefined
    armnn::INetworkProperties networkProperties(true, MemorySource::Malloc, MemorySource::Malloc);

    // Optimize and Load the network into runtime
    runtime->LoadNetwork(networkIdentifier1,
                         Optimize(*testNetwork, backends, runtime->GetDeviceSpec()),
                         er,
                         networkProperties);

    // Create structures for input & output
    std::vector<float> inputData1(4, number);
    std::vector<float> inputData2(4, number);
    ConstTensor inputTensor1(tensorInfo, inputData1.data());
    ConstTensor inputTensor2(tensorInfo, inputData2.data());

    std::vector<float> outputData1(4);
    Tensor outputTensor1{tensorInfo, outputData1.data()};

    // ImportInputs separates the importing and mapping of InputTensors from network execution.
    // Allowing for a set of InputTensors to be imported and mapped once, but used in execution many times.
    // ImportInputs is not thread safe and must not be used while other threads are calling Execute().
    // Only compatible with AsyncEnabled networks

    // PreImport inputTensors giving pre-imported ids of 1 and 2
    std::vector<ImportedInputId> importedInputVec = runtime->ImportInputs(networkIdentifier1,
                                                                          {{0, inputTensor1}, {1, inputTensor2}});

    // Create a new unique WorkingMemHandle object. Create multiple handles if you wish to have
    // overlapped Execution by calling this function from different threads.
    auto memHandle = runtime->CreateWorkingMemHandle(networkIdentifier1);

    // Execute evaluates a network using input in inputTensors and outputs filled into outputTensors.
    // This function performs a thread safe execution of the network. Returns once execution is complete.
    // Will block until this and any other thread using the same workingMem object completes.
    // Execute with PreImported inputTensor1 as well as Non-PreImported inputTensor2
    runtime->Execute(*memHandle.get(), {}, {{2, outputTensor1}}, importedInputVec /* pre-imported ids */);

    // ImportOutputs separates the importing and mapping of OutputTensors from network execution.
    // Allowing for a set of OutputTensors to be imported and mapped once, but used in execution many times.
    // This function is not thread safe and must not be used while other threads are calling Execute().
    // Only compatible with AsyncEnabled networks
    // Provide layerBinding Id to outputTensor1
    std::pair<LayerBindingId, class Tensor> output1{2, outputTensor1};
    // PreImport outputTensor1
    std::vector<ImportedOutputId> importedOutputVec = runtime->ImportOutputs(networkIdentifier1, {output1});

    // Execute with Non-PreImported inputTensor1 as well as PreImported inputTensor2
    runtime->Execute(*memHandle.get(), {{0, inputTensor1}}, {{2, outputTensor1}}, {1 /* pre-imported id */});

    // Clear the previously PreImportedInput with the network Id and inputIds returned from ImportInputs()
    // Note: This will happen automatically during destructor of armnn::LoadedNetwork
    runtime->ClearImportedInputs(networkIdentifier1, importedInputVec);

    // Clear the previously PreImportedOutputs with the network Id and outputIds returned from ImportOutputs()
    // Note: This will happen automatically during destructor of armnn::LoadedNetwork
    runtime->ClearImportedOutputs(networkIdentifier1, importedOutputVec);

    // Execute with Non-PreImported inputTensor1, inputTensor2 and the PreImported outputTensor1
    runtime->Execute(*memHandle.get(), {{0, inputTensor1}, {1, inputTensor2}}, {{2, outputTensor1}});

    std::cout << "Your number was " << outputData1.data()[0] << std::endl;

    return 0;
}
