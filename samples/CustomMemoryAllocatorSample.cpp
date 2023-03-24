//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>
#include <armnn/backends/ICustomAllocator.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <iostream>

/** Sample implementation of ICustomAllocator for use with the ClBackend.
 *  Note: any memory allocated must be host addressable with write access
 *  in order for ArmNN to be able to properly use it. */
class SampleClBackendCustomAllocator : public armnn::ICustomAllocator
{
public:
    SampleClBackendCustomAllocator() = default;

    void* allocate(size_t size, size_t alignment) override
    {
        // If alignment is 0 just use the CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE for alignment
        if (alignment == 0)
        {
            alignment = arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
        }
        size_t space = size + alignment + alignment;
        auto allocatedMemPtr = std::malloc(space * sizeof(size_t));

        if (std::align(alignment, size, allocatedMemPtr, space) == nullptr)
        {
            throw armnn::Exception("SampleClBackendCustomAllocator::Alignment failed");
        }
        return allocatedMemPtr;
    }

    void free(void* ptr) override
    {
        std::free(ptr);
    }

    armnn::MemorySource GetMemorySourceType() override
    {
        return armnn::MemorySource::Malloc;
    }
};


// A simple example application to show the usage of a custom memory allocator. In this sample, the users single
// input number is multiplied by 1.0f using a fully connected layer with a single neuron to produce an output
// number that is the same as the input. All memory required to execute this mini network is allocated with
// the provided custom allocator.
//
// Using a Custom Allocator is required for use with Protected Mode and Protected Memory.
// This example is provided using only unprotected malloc as Protected Memory is platform
// and implementation specific.
//
// Note: This example is similar to the SimpleSample application that can also be found in armnn/samples.
//       The differences are in the use of a custom allocator, the backend is GpuAcc, and the inputs/outputs
//       are being imported instead of copied. (Import must be enabled when using a Custom Allocator)
//       You might find this useful for comparison.
int main()
{
    using namespace armnn;

    float number;
    std::cout << "Please enter a number: " << std::endl;
    std::cin >> number;

    // Turn on logging to standard output
    // This is useful in this sample so that users can learn more about what is going on
    ConfigureLogging(true, false, LogSeverity::Info);

    // Construct ArmNN network
    NetworkId networkIdentifier;
    INetworkPtr network = INetwork::Create();
    FullyConnectedDescriptor fullyConnectedDesc;
    float weightsData[] = {1.0f}; // Identity
    TensorInfo weightsInfo(TensorShape({1, 1}), DataType::Float32, 0.0f, 0, true);
    weightsInfo.SetConstant(true);
    ConstTensor weights(weightsInfo, weightsData);

    IConnectableLayer* inputLayer   = network->AddInputLayer(0);
    IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Weights");
    IConnectableLayer* fullyConnectedLayer =
            network->AddFullyConnectedLayer(fullyConnectedDesc, "fully connected");
    IConnectableLayer* outputLayer  = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    // Create ArmNN runtime:
    //
    // This is the interesting bit when executing a model with a custom allocator.
    // You can have different allocators for different backends. To support this
    // the runtime creation option has a map that takes a BackendId and the corresponding
    // allocator that should be used for that backend.
    // Only GpuAcc supports a Custom Allocator for now
    //
    // Note: This is not covered in this example but if you want to run a model on
    //       protected memory a custom allocator needs to be provided that supports
    //       protected memory allocations and the MemorySource of that allocator is
    //       set to MemorySource::DmaBufProtected
    IRuntime::CreationOptions options;
    auto customAllocator = std::make_shared<SampleClBackendCustomAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr runtime = IRuntime::Create(options);

    //Set the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    TensorInfo outputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(true);
    IOptimizedNetworkPtr optNet =
                Optimize(*network, {"GpuAcc"}, runtime->GetDeviceSpec(), optOptions);
    if (!optNet)
    {
        // This shouldn't happen for this simple sample, with GpuAcc backend.
        // But in general usage Optimize could fail if the backend at runtime cannot
        // support the model that has been provided.
        std::cerr << "Error: Failed to optimise the input network." << std::endl;
        return 1;
    }

    // Load graph into runtime
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(networkIdentifier, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
            arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, alignment);

    // Input with negative values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);

    void* alignedOutputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, alignment);
    auto* outputPtr = reinterpret_cast<float*>(alignedOutputPtr);
    std::fill_n(outputPtr, numElements, -10.0f);

    inputTensorInfo = runtime->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0, ConstTensor(inputTensorInfo, alignedInputPtr)},
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime->GetOutputTensorInfo(networkIdentifier, 0), alignedOutputPtr)}
    };

    // Execute network
    runtime->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);

    // Tell the CLBackend to sync memory so we can read the output.
    arm_compute::CLScheduler::get().sync();
    auto* outputResult = reinterpret_cast<float*>(alignedOutputPtr);
    std::cout << "Your number was " << outputResult[0] << std::endl;
    runtime->UnloadNetwork(networkIdentifier);
    return 0;

}
