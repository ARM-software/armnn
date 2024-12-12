//
// Copyright © 2021, 2023-2024 Arm Ltd and Contributors. All rights reserved.
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
#if defined(ARMCOMPUTENEON_ENABLED)
#include <neon/NeonBackend.hpp>
#endif
#include <doctest/doctest.h>
#include <armnn/utility/IgnoreUnused.hpp>
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

    /** Interface to be implemented by the child class to free the allocated tensor */
    void free(void* ptr) override
    {
        std::free(ptr);
    }

    armnn::MemorySource GetMemorySourceType() override
    {
        return armnn::MemorySource::Malloc;
    }
};

armnn::INetworkPtr CreateTestNetwork(armnn::TensorInfo& inputTensorInfo)
{
    using namespace armnn;

    armnn::FullyConnectedDescriptor fullyConnectedDesc;
    float weightsData[] = {1.0f}; // Identity
    TensorInfo weightsInfo(TensorShape({1, 1}), DataType::Float32, 0.0f, 0, true);
    weightsInfo.SetConstant(true);
    armnn::ConstTensor weights(weightsInfo, weightsData);

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer = network->AddInputLayer(0);
    armnn::IConnectableLayer* const weightsLayer = network->AddConstantLayer(weights, "Weights");
    armnn::IConnectableLayer* const fullyConnectedLayer =
        network->AddFullyConnectedLayer(fullyConnectedDesc, "fully connected");
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    fullyConnectedLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    //Set the tensors in the network.

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

    TensorInfo outputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    fullyConnectedLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    return network;
}

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

    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);

    INetworkPtr myNetwork = CreateTestNetwork(inputTensorInfo);

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<SampleClBackendCustomAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);

    // Optimise ArmNN network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(true);
    optOptions.SetExportEnabled(true);
    armnn::IOptimizedNetworkPtr optNet = Optimize(*myNetwork, {"GpuAcc"}, run->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Load graph into runtime
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(MemorySource::Malloc, MemorySource::Malloc);
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

    armnn::TensorInfo inputTensorInfo2 = run->GetInputTensorInfo(networkIdentifier, 0);
    inputTensorInfo2.SetConstant(true);
    armnn::InputTensors inputTensors
    {
        {0, armnn::ConstTensor(inputTensorInfo2, alignedInputPtr)},
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

// Only run this test if NEON is enabled
#if defined(ARMCOMPUTENEON_ENABLED)

TEST_CASE("ClCustomAllocatorCpuAccNegativeTest")
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<SampleClBackendCustomAllocator>();
    options.m_CustomAllocatorMap = {{"CpuAcc", std::move(customAllocator)}};
    IRuntimePtr run = IRuntime::Create(options);
    TensorInfo inputTensorInfo(TensorShape({1, 1}), DataType::Float32);
    INetworkPtr myNetwork = CreateTestNetwork(inputTensorInfo);

    // Optimise ArmNN network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(true);
    IOptimizedNetworkPtr optNet(nullptr, nullptr);
    std::vector<std::string> errMessages;

    CHECK_THROWS_AS_MESSAGE(Optimize(*myNetwork, {"CpuAcc"}, run->GetDeviceSpec(), optOptions, errMessages),
                            armnn::InvalidArgumentException,
                            "Expected an exception as GetAvailablePreferredBackends() should be empty in Optimize().");

    auto& backendRegistry = armnn::BackendRegistryInstance();
    backendRegistry.DeregisterAllocator(NeonBackend::GetIdStatic());
}

#endif

TEST_CASE("ClCustomAllocatorGpuAccNullptrTest")
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customAllocator = std::make_shared<SampleClBackendCustomAllocator>();
    options.m_CustomAllocatorMap = {{"GpuAcc", nullptr}};

    CHECK_THROWS_AS_MESSAGE(IRuntimePtr run = IRuntime::Create(options),
                            armnn::Exception,
                            "Expected exception in RuntimeImpl::RuntimeImpl() as allocator was nullptr.");
}

} // test suite ClCustomAllocatorTests
