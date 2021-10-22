//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

#include <cl/ClImportTensorHandle.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>

#include <doctest/doctest.h>


#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>

using namespace armnn;

TEST_SUITE("ClImportTensorHandleTests")
{
TEST_CASE_FIXTURE(ClContextControlFixture, "ClMallocImport")
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 24, 16, 3 }, DataType::Float32);
    unsigned int numElements = info.GetNumElements();

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClImportTensorHandle*>(handle.get())->GetTensor();

    // Create and configure activation function
    const arm_compute::ActivationLayerInfo act_info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    arm_compute::CLActivationLayer act_func;
    act_func.configure(&tensor, nullptr, act_info);

    // Allocate user memory
    const size_t totalBytes = tensor.info()->total_size();
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Import memory
    CHECK(handle->Import(alignedPtr, armnn::MemorySource::Malloc));

    // Input with negative values
    auto* typedPtr = reinterpret_cast<float*>(alignedPtr);
    std::fill_n(typedPtr, numElements, -5.0f);

    // Execute function and sync
    act_func.run();
    arm_compute::CLScheduler::get().sync();

    // Validate result by checking that the output has no negative values
    for(unsigned int i = 0; i < numElements; ++i)
    {
        CHECK(typedPtr[i] == 0);
    }
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClIncorrectMemorySourceImport")
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 24, 16, 3 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClImportTensorHandle*>(handle.get())->GetTensor();

    // Allocate user memory
    const size_t totalBytes = tensor.info()->total_size();
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Import memory
    CHECK_THROWS_AS(handle->Import(alignedPtr, armnn::MemorySource::Undefined), MemoryImportException);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClInvalidMemorySourceImport")
{
    MemorySource invalidMemSource = static_cast<MemorySource>(256);
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(invalidMemSource),
                                              static_cast<MemorySourceFlags>(invalidMemSource));

    TensorInfo info({ 1, 2, 2, 1 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info);

    // Allocate user memory
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    // Import non-support memory
    CHECK_THROWS_AS(handle->Import(inputData.data(), invalidMemSource), MemoryImportException);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClImportEndToEnd")
{
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "Input");

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::ReLu;
    IConnectableLayer* activation = net->AddActivationLayer(descriptor, "Activation");

    IConnectableLayer* output = net->AddOutputLayer(0, "Output");

    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo tensorInfo = TensorInfo({ 1, 24, 16, 3 }, DataType::Float32);
    unsigned int numElements = tensorInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    input->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    activation->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // Optimize the network
    OptimizerOptions optOptions;
    optOptions.m_ImportEnabled = true;
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(space);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytes, alignedInputPtr, space));

    // Input with negative values
    auto* intputPtr = reinterpret_cast<float*>(alignedInputPtr);
    std::fill_n(intputPtr, numElements, -5.0f);

    auto outputData = std::make_unique<uint8_t[]>(space);
    void* alignedOutputPtr = outputData.get();
    CHECK(std::align(alignment, totalBytes, alignedOutputPtr, space));
    auto* outputPtr = reinterpret_cast<float*>(alignedOutputPtr);
    std::fill_n(outputPtr, numElements, -10.0f);

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(inputTensorInfo, alignedInputPtr)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), alignedOutputPtr)}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains ActivationWorkload
    std::size_t found = dump.find("ActivationWorkload");
    CHECK(found != std::string::npos);

    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found != std::string::npos);

    // Does not contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    CHECK(found == std::string::npos);

    runtime->UnloadNetwork(netId);

    // Check output is as expected
    // Validate result by checking that the output has no negative values
    auto* outputResult = reinterpret_cast<float*>(alignedOutputPtr);
    CHECK(outputResult);
    for(unsigned int i = 0; i < numElements; ++i)
    {
        CHECK(outputResult[i] >= 0);
    }
}

}
