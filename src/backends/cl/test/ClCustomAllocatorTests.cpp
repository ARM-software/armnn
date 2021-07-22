//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/ICustomAllocator.hpp>
#include <armnn/Descriptors.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/BackendRegistry.hpp>
#include <cl/ClBackend.hpp>

#include <doctest/doctest.h>

// Contains the OpenCl interfaces for mapping memory in the Gpu Page Tables
// Requires the OpenCl backend to be included (GpuAcc)
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <CL/cl_ext.h>
#include <arm_compute/runtime/CL/CLScheduler.h>


/** Sample implementation of ICustomAllocator for use with the ClBackend.
 *  Note: any memory allocated must be host accessible with write access to allow for weights and biases
 *  to be passed in. Read access is not required.. */
class SampleClBackendCustomAllocator : public armnn::ICustomAllocator
{
public:
    SampleClBackendCustomAllocator() = default;

    void* allocate(size_t size, size_t alignment)
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

    /** Interface to be implemented by the child class to free the allocated tensor */
    void free(void* ptr)
    {
        std::free(ptr);
    }

    armnn::MemorySource GetMemorySourceType()
    {
        return armnn::MemorySource::Malloc;
    }
};

TEST_SUITE("ClCustomAllocatorTests")
{

// This is a copy of the SimpleSample app modified to use a custom
// allocator for the clbackend. It creates a FullyConnected network with a single layer
// taking a single number as an input
TEST_CASE("ClCustomAllocatorTest")
{
    using namespace armnn;

    float number = 3;

    // Construct ArmNN network
    armnn::NetworkId networkIdentifier;
    INetworkPtr myNetwork = INetwork::Create();

    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    float weightsData[] = {1.0f}; // Identity
    TensorInfo weightsInfo(TensorShape({1, 1}), DataType::Float32);
    weightsInfo.SetConstant(true);
    armnn::ConstTensor weights(weightsInfo, weightsData);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    IConnectableLayer* fullyConnected = myNetwork->AddFullyConnectedLayer(fullyConnectedDesc,
                                                                          weights,
                                                                          EmptyOptional(),
                                                                          "fully connected");
    ARMNN_NO_DEPRECATE_WARN_END
    IConnectableLayer* InputLayer = myNetwork->AddInputLayer(0);
    IConnectableLayer* OutputLayer = myNetwork->AddOutputLayer(0);
    InputLayer->GetOutputSlot(0).Connect(fullyConnected->GetInputSlot(0));
    fullyConnected->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));


    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<SampleClBackendCustomAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    //Set the tensors in the network.
    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    fullyConnected->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimise ArmNN network
    OptimizerOptions optOptions;
    optOptions.m_ImportEnabled = true;
    armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {"GpuAcc"}, run->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Load graph into runtime
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    run->LoadNetwork(networkIdentifier, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    unsigned int numElements = inputTensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    const size_t alignment =
            arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();

    void* alignedInputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, alignment);

    // Input with negative values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(inputPtr, numElements, number);

    void* alignedOutputPtr = options.m_CustomAllocatorMap["GpuAcc"]->allocate(totalBytes, alignment);
    auto* outputPtr = reinterpret_cast<float*>(alignedOutputPtr);
    std::fill_n(outputPtr, numElements, -10.0f);

    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(run->GetInputTensorInfo(networkIdentifier, 0), alignedInputPtr)},
    };
    armnn::OutputTensors outputTensors
    {
        {0, armnn::Tensor(run->GetOutputTensorInfo(networkIdentifier, 0), alignedOutputPtr)}
    };

    // Execute network
    run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
    run->UnloadNetwork(networkIdentifier);


    // Tell the CLBackend to sync memory so we can read the output.
    arm_compute::CLScheduler::get().sync();
    auto* outputResult = reinterpret_cast<float*>(alignedOutputPtr);

    run->UnloadNetwork(networkIdentifier);
    CHECK(outputResult[0] == number);
    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(ClBackend::GetIdStatic());
}

} // test suite ClCustomAllocatorTests