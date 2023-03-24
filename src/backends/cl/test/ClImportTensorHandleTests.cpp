//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <arm_compute/runtime/CL/functions/CLActivationLayer.h>

#include <cl/ClImportTensorHandle.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>

#include <doctest/doctest.h>

#include <armnn/IRuntime.hpp>
#include <armnn/INetwork.hpp>
#include "Network.hpp"

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
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(true);
    optOptions.SetExportEnabled(true);
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

TEST_CASE_FIXTURE(ClContextControlFixture, "ClCanBeImported")
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 24, 16, 3 }, DataType::Float32);

    // create TensorHandle for memory import
    auto handle = handleFactory.CreateTensorHandle(info, DataLayout::NHWC);

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
    CHECK_THROWS_AS(handle->CanBeImported(alignedPtr, armnn::MemorySource::Undefined), MemoryImportException);

}

TEST_CASE("ClCanBeImportedAlignedMemory")
{
    ClImportTensorHandleFactory handleFactory(static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                              static_cast<MemorySourceFlags>(MemorySource::Malloc));

    TensorInfo info({ 1, 1, 1, 1 }, DataType::Float32);

    // create TensorHandle (Memory Managed status is irrelevant)
    auto handle = handleFactory.CreateTensorHandle(info, DataLayout::NHWC);
    // Get CLtensor
    arm_compute::CLTensor& tensor = PolymorphicDowncast<ClImportTensorHandle*>(handle.get())->GetTensor();

    // Create an aligned buffer
    const size_t totalBytes = tensor.info()->total_size();
    const size_t alignment =
            arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto testData = std::make_unique<uint8_t[]>(space);
    void* alignedPtr = testData.get();
    CHECK(std::align(alignment, totalBytes, alignedPtr, space));

    // Check aligned buffers return true
    CHECK(handle->CanBeImported(alignedPtr, MemorySource::Malloc) == true);

    // Due to the nature of how GPU memory is mapped it is entirely possible for memory which is misaligned on cpu
    // to be successfully import on GPU. As such there is no way to create a misaligned pointer that will always fail.
    // Rather it will succeed on some devices and fail on others. As long as a correctly aligned buffer returns true
    // we can be confident that it will be successfully imported. All other cases will need to be handled by the user.
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportConv2dEndToEnd")
{
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr network(INetwork::Create());

    armnn::TensorInfo inputInfo({ 1, 3, 4, 1 }, DataType::Float32);
    armnn::TensorInfo kernelInfo({ 1, 3, 3, 1 }, DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 4, 1 }, DataType::Float32);

    kernelInfo.SetConstant(true);

    std::vector<float> kernel =
    {
        4, 5, 6,
        0, 0, 0,
        3, 2, 1
    };

    const std::vector<float> expectedOutput =
    {
        23, 41, 33, 21,
        44, 65, 76, 52,
        82, 85, 79, 42
    };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    IConnectableLayer* const inputLayer = network->AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::ConstTensor weights(kernelInfo, kernel);

    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_StrideX = 1;
    convDesc2d.m_StrideY = 1;
    convDesc2d.m_PadLeft = 1;
    convDesc2d.m_PadRight = 1;
    convDesc2d.m_PadTop = 1;
    convDesc2d.m_PadBottom = 1;
    convDesc2d.m_DataLayout = DataLayout::NHWC;

    armnn::IConnectableLayer* const convLayer = network->AddConvolution2dLayer(convDesc2d, "conv");
    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights);

    ARMNN_ASSERT(convLayer);

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1u));

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network->AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(space);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytes, alignedInputPtr, space));

    // Input with negative values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    inputPtr[0] = 1;
    inputPtr[1] = 5;
    inputPtr[2] = 2;
    inputPtr[3] = 3;
    inputPtr[4] = 8;
    inputPtr[5] = 7;
    inputPtr[6] = 3;
    inputPtr[7] = 6;
    inputPtr[8] = 3;
    inputPtr[9] = 3;
    inputPtr[10] = 9;
    inputPtr[11] = 1;


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

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedOutputIds.size() == 1);
    // Do the inference
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("Convolution2dWorkload");
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

    // Check the output is correct
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportConvertFp16toFp32EndToEnd")
{
    using namespace half_float::literal;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    NetworkImpl network;

    armnn::TensorInfo inputInfo({1, 3, 2, 3}, armnn::DataType::Float16);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);

    std::vector<float> expectedOutput =
    {
        -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
        1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f
    };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytesInput = numElements * sizeof(Half);
    size_t totalBytesOutput = numElements * sizeof(float);

    IConnectableLayer* const inputLayer = network.AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::IConnectableLayer* const convLayer = network.AddConvertFp16ToFp32Layer("convert");
    ARMNN_ASSERT(convLayer);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network.AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(network.GetGraph(), backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t spaceInput = totalBytesInput + alignment + alignment;
    size_t spaceOutput = totalBytesOutput + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(spaceInput);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytesInput, alignedInputPtr, spaceInput));

    // Input with negative values
    auto* inputPtr = reinterpret_cast<Half*>(alignedInputPtr);
    inputPtr[0] = -37.5_h;
    inputPtr[1] = -15.2_h;
    inputPtr[2] = -8.76_h;
    inputPtr[3] = -2.0_h;
    inputPtr[4] = -1.5_h;
    inputPtr[5] = -1.3_h;
    inputPtr[6] = -0.5_h;
    inputPtr[7] = -0.4_h;
    inputPtr[8] = 0.0_h;
    inputPtr[9] = 1.0_h;
    inputPtr[10] = 0.4_h;
    inputPtr[11] = 0.5_h;
    inputPtr[12] = 1.3_h;
    inputPtr[13] = 1.5_h;
    inputPtr[14] = 2.0_h;
    inputPtr[15] = 8.76_h;
    inputPtr[16] = 15.2_h;
    inputPtr[17] = 37.5_h;

    auto outputData = std::make_unique<uint8_t[]>(spaceOutput);
    void* alignedOutputPtr = outputData.get();
    CHECK(std::align(alignment, totalBytesOutput, alignedOutputPtr, spaceOutput));
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

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedOutputIds.size() == 1);

    // Do the inference
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("ConvertFp16ToFp32Workload");
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

    // Check the output is correct
    for (size_t i = 0; i < numElements; ++i)
    {
        DOCTEST_CHECK_MESSAGE(outputResult[i] == doctest::Approx(expectedOutput[i]).epsilon(0.0004),
                              "outputValue[" << i << "]: " << outputResult[i] << " != " << expectedOutput[i]);
    }
}


TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportConvertFp32toFp16EndToEnd")
{
    using namespace half_float::literal;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    NetworkImpl network;

    armnn::TensorInfo inputInfo({1, 3, 2, 3}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float16);

    std::vector<Half> expectedOutput =
    {
        -37.5_h, -15.2_h, -8.76_h, -2.0_h, -1.5_h, -1.3_h, -0.5_h, -0.4_h, 0.0_h,
        1.0_h, 0.4_h, 0.5_h, 1.3_h, 1.5_h, 2.0_h, 8.76_h, 15.2_h, 37.5_h
    };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytesInput = numElements * sizeof(float);
    size_t totalBytesOutput = numElements * sizeof(Half);

    IConnectableLayer* const inputLayer = network.AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::IConnectableLayer* const convLayer = network.AddConvertFp32ToFp16Layer("convert");
    ARMNN_ASSERT(convLayer);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network.AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(network.GetGraph(), backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t spaceInput = totalBytesInput + alignment + alignment;
    size_t spaceOutput = totalBytesOutput + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(spaceInput);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytesInput, alignedInputPtr, spaceInput));

    // Input with negative values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    inputPtr[0] = -37.5f;
    inputPtr[1] = -15.2f;
    inputPtr[2] = -8.76f;
    inputPtr[3] = -2.0f;
    inputPtr[4] = -1.5f;
    inputPtr[5] = -1.3f;
    inputPtr[6] = -0.5f;
    inputPtr[7] = -0.4f;
    inputPtr[8] = 0.0f;
    inputPtr[9] = 1.0f;
    inputPtr[10] = 0.4f;
    inputPtr[11] = 0.5f;
    inputPtr[12] = 1.3f;
    inputPtr[13] = 1.5f;
    inputPtr[14] = 2.0f;
    inputPtr[15] = 8.76f;
    inputPtr[16] = 15.2f;
    inputPtr[17] = 37.5f;

    auto outputData = std::make_unique<uint8_t[]>(spaceOutput);
    void* alignedOutputPtr = outputData.get();
    CHECK(std::align(alignment, totalBytesOutput, alignedOutputPtr, spaceOutput));
    auto* outputPtr = reinterpret_cast<Half*>(alignedOutputPtr);
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

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedOutputIds.size() == 1);

    // Do the inference
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("ConvertFp32ToFp16Workload");
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
    auto* outputResult = reinterpret_cast<Half*>(alignedOutputPtr);
    CHECK(outputResult);

    // Check the output is correct
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportSimpleConvertFp32toFp16EndToEnd")
{
    using namespace half_float::literal;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    NetworkImpl network;

    armnn::TensorInfo inputInfo({1}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1}, armnn::DataType::Float16);

    std::vector<Half> expectedOutput = { 1.0_h };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytesInput = numElements * sizeof(float);
    size_t totalBytesOutput = numElements * sizeof(Half);

    IConnectableLayer* const inputLayer = network.AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::IConnectableLayer* const convLayer = network.AddConvertFp32ToFp16Layer("convert");
    ARMNN_ASSERT(convLayer);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network.AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(network.GetGraph(), backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t spaceInput = totalBytesInput + alignment + alignment;
    size_t spaceOutput = totalBytesOutput + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(spaceInput);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytesInput, alignedInputPtr, spaceInput));

    // Input with negative values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    inputPtr[0] = 1.0f;

    auto outputData = std::make_unique<uint8_t[]>(spaceOutput);
    void* alignedOutputPtr = outputData.get();
    CHECK(std::align(alignment, totalBytesOutput, alignedOutputPtr, spaceOutput));
    auto* outputPtr = reinterpret_cast<Half*>(alignedOutputPtr);
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

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    CHECK(importedOutputIds.size() == 1);

    // Do the inference
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("ConvertFp32ToFp16Workload");
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
    auto* outputResult = reinterpret_cast<Half*>(alignedOutputPtr);
    CHECK(outputResult);

    // Check the output is correct
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportRepeatedInferencesEndToEndTest")
{
/*
 * This is a test to check the functionality of the Forced Import functionality when using repeated inferences that
 * require switching from importing to copy. For the first inference we create aligned Pointers and check they are
 * imported correctly. For the second we use similar pointers but don't use PreImporting.
 */
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr network(INetwork::Create());

    armnn::TensorInfo inputInfo({ 1, 3, 4, 1 }, DataType::Float32);
    armnn::TensorInfo kernelInfo({ 1, 3, 3, 1 }, DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 4, 1 }, DataType::Float32);

    kernelInfo.SetConstant(true);

    std::vector<float> kernel =
    {
        4, 5, 6,
        0, 0, 0,
        3, 2, 1
    };

    const std::vector<float> expectedOutput =
    {
        23, 41, 33, 21,
        44, 65, 76, 52,
        82, 85, 79, 42
    };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    IConnectableLayer* const inputLayer = network->AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::ConstTensor weights(kernelInfo, kernel);

    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_StrideX = 1;
    convDesc2d.m_StrideY = 1;
    convDesc2d.m_PadLeft = 1;
    convDesc2d.m_PadRight = 1;
    convDesc2d.m_PadTop = 1;
    convDesc2d.m_PadBottom = 1;
    convDesc2d.m_DataLayout = DataLayout::NHWC;
    armnn::IConnectableLayer* const convLayer = network->AddConvolution2dLayer(convDesc2d, "conv");
    ARMNN_ASSERT(convLayer);

    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights);

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1u));

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network->AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(space);
    void* alignedInputPtr = inputData.get();
    CHECK(std::align(alignment, totalBytes, alignedInputPtr, space));

    // Fill input with values
    auto* inputPtr = reinterpret_cast<float*>(alignedInputPtr);
    inputPtr[0] = 1;
    inputPtr[1] = 5;
    inputPtr[2] = 2;
    inputPtr[3] = 3;
    inputPtr[4] = 8;
    inputPtr[5] = 7;
    inputPtr[6] = 3;
    inputPtr[7] = 6;
    inputPtr[8] = 3;
    inputPtr[9] = 3;
    inputPtr[10] = 9;
    inputPtr[11] = 1;


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

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // We expect the import to have succeeded.
    CHECK(importedOutputIds.size() == 1);

    // Do the inference
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("Convolution2dWorkload");
    CHECK(found != std::string::npos);

    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found != std::string::npos);

    // Does not contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    CHECK(found == std::string::npos);

    // Sync the outputs so we can read the data
    arm_compute::CLScheduler::get().sync();

    // Check output is as expected
    auto* outputResult = reinterpret_cast<float*>(alignedOutputPtr);
    CHECK(outputResult);
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));

    // Repeat the inference, with new tensors and without using PreImporting to force it to fall back to copying

    // Creates structures for input & output
    auto inputDataCopy = std::make_unique<uint8_t[]>(space);
    void* copyInputPtr = inputDataCopy.get();

    // Fill input with values
    auto* inputCopyPtr = reinterpret_cast<float*>(copyInputPtr);
    inputCopyPtr[0] = 1;
    inputCopyPtr[1] = 5;
    inputCopyPtr[2] = 2;
    inputCopyPtr[3] = 3;
    inputCopyPtr[4] = 8;
    inputCopyPtr[5] = 7;
    inputCopyPtr[6] = 3;
    inputCopyPtr[7] = 6;
    inputCopyPtr[8] = 3;
    inputCopyPtr[9] = 3;
    inputCopyPtr[10] = 9;
    inputCopyPtr[11] = 1;

    // Output pre-filled with -10.0f
    auto outputDataCopy = std::make_unique<uint8_t[]>(space);
    void* copyOutputPtr = outputDataCopy.get();
    auto* outputCopyPtr = reinterpret_cast<float*>(copyOutputPtr);
    std::fill_n(outputCopyPtr, numElements, -10.0f);

    InputTensors inputTensorsCopy
    {
        {0,armnn::ConstTensor(inputTensorInfo, copyInputPtr)},
    };
    OutputTensors outputTensorsCopy
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), copyOutputPtr)}
    };

    // Do the inference without any pre-imported input/output ids
    runtime->EnqueueWorkload(netId, inputTensorsCopy, outputTensorsCopy);
    // Sync the outputs so we can read the data
    arm_compute::CLScheduler::get().sync();

    // Check the output is correct
    outputResult = reinterpret_cast<float*>(copyOutputPtr);
    CHECK(outputResult);
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));

    // Query the profiler again, this will contain the results of both inferences
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    dump = ss.str();

    // Contains Convolution2dWorkload
    found = dump.find("Convolution2dWorkload");
    CHECK(found != std::string::npos);

    // Should still contain the SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found != std::string::npos);

    // Should now also contain a CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    CHECK(found != std::string::npos);
    runtime->UnloadNetwork(netId);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClForceImportRepeatedInferencesInvertedEndToEndTest")
{
/*
 * This test is similar to the test above but instead of importing and then copying, we start by copying and then do
 * the import.
 */
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr network(INetwork::Create());

    armnn::TensorInfo inputInfo({ 1, 3, 4, 1 }, DataType::Float32);
    armnn::TensorInfo kernelInfo({ 1, 3, 3, 1 }, DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 4, 1 }, DataType::Float32);

    kernelInfo.SetConstant(true);

    std::vector<float> kernel =
    {
        4, 5, 6,
        0, 0, 0,
        3, 2, 1
    };

    const std::vector<float> expectedOutput =
    {
        23, 41, 33, 21,
        44, 65, 76, 52,
        82, 85, 79, 42
    };

    unsigned int numElements = inputInfo.GetNumElements();
    size_t totalBytes = numElements * sizeof(float);

    IConnectableLayer* const inputLayer = network->AddInputLayer(0, "input");
    ARMNN_ASSERT(inputLayer);

    armnn::ConstTensor weights(kernelInfo, kernel);

    armnn::Convolution2dDescriptor convDesc2d;
    convDesc2d.m_StrideX = 1;
    convDesc2d.m_StrideY = 1;
    convDesc2d.m_PadLeft = 1;
    convDesc2d.m_PadRight = 1;
    convDesc2d.m_PadTop = 1;
    convDesc2d.m_PadBottom = 1;
    convDesc2d.m_DataLayout = DataLayout::NHWC;

    armnn::IConnectableLayer* const convLayer = network->AddConvolution2dLayer(convDesc2d, "conv");
    ARMNN_ASSERT(convLayer);

    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights);

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1u));

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    IConnectableLayer* output = network->AddOutputLayer(0, "output");
    convLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    // Optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetImportEnabled(false);
    optOptions.SetExportEnabled(false);
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optOptions);
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    const size_t alignment =
        arm_compute::CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    size_t space = totalBytes + alignment + alignment;
    auto inputData = std::make_unique<uint8_t[]>(space);
    void* copyInputPtr = inputData.get();

    // Fill input with values
    auto* inputPtr = reinterpret_cast<float*>(copyInputPtr);
    inputPtr[0] = 1;
    inputPtr[1] = 5;
    inputPtr[2] = 2;
    inputPtr[3] = 3;
    inputPtr[4] = 8;
    inputPtr[5] = 7;
    inputPtr[6] = 3;
    inputPtr[7] = 6;
    inputPtr[8] = 3;
    inputPtr[9] = 3;
    inputPtr[10] = 9;
    inputPtr[11] = 1;

    // Create output buffer and fill it with -10.0f
    auto outputData = std::make_unique<uint8_t[]>(space);
    void* copyOutputPtr = outputData.get();
    auto* outputPtr = reinterpret_cast<float*>(copyOutputPtr);
    std::fill_n(outputPtr, numElements, -10.0f);

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(inputTensorInfo, copyInputPtr)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), copyOutputPtr)}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference without any pre-imported inputs/outputs
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    std::string dump = ss.str();

    // Contains Convolution2dWorkload
    std::size_t found = dump.find("Convolution2dWorkload");
    CHECK(found != std::string::npos);

    // Does not contain SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found == std::string::npos);

    // Does contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    CHECK(found != std::string::npos);

    // Sync the outputs so we can read the data
    arm_compute::CLScheduler::get().sync();

    // Check output is as expected
    auto* outputResult = reinterpret_cast<float*>(copyOutputPtr);
    CHECK(outputResult);
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));

    // Repeat the inference, with new tensors and while using pre-importing to force it to import

    // Creates structures for input & output
    auto inputDataImport = std::make_unique<uint8_t[]>(space);
    void* alignedInputImportPtr = inputDataImport.get();
    CHECK(std::align(alignment, totalBytes, alignedInputImportPtr, space));

    // Fill input with values
    auto* inputImportPtr = reinterpret_cast<float*>(alignedInputImportPtr);
    inputImportPtr[0] = 1;
    inputImportPtr[1] = 5;
    inputImportPtr[2] = 2;
    inputImportPtr[3] = 3;
    inputImportPtr[4] = 8;
    inputImportPtr[5] = 7;
    inputImportPtr[6] = 3;
    inputImportPtr[7] = 6;
    inputImportPtr[8] = 3;
    inputImportPtr[9] = 3;
    inputImportPtr[10] = 9;
    inputImportPtr[11] = 1;

    // Output pre-filled with -10.0f
    auto outputDataImport = std::make_unique<uint8_t[]>(space);
    void* alignedOutputImportPtr = outputDataImport.get();
    CHECK(std::align(alignment, totalBytes, alignedOutputImportPtr, space));
    auto* outputImportPtr = reinterpret_cast<float*>(alignedOutputImportPtr);
    std::fill_n(outputImportPtr, numElements, -10.0f);

    InputTensors inputTensorsImport
    {
        {0,armnn::ConstTensor(inputTensorInfo, alignedInputImportPtr)},
    };
    OutputTensors outputTensorsImport
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), alignedOutputImportPtr)}
    };

    INFO("Run ImportInputs");
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensorsImport, MemorySource::Malloc);
    CHECK(importedInputIds.size() == 1);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensorsImport, MemorySource::Malloc);
    CHECK(importedOutputIds.size() == 1);

    // Do the inference with pre-imported inputs/outputs
    runtime->EnqueueWorkload(netId, InputTensors(), OutputTensors(), importedInputIds, importedOutputIds);
    // Sync the outputs so we can read the data
    arm_compute::CLScheduler::get().sync();

    // Check the output is correct
    outputResult = reinterpret_cast<float*>(alignedOutputImportPtr);
    CHECK(outputResult);
    CHECK(std::equal(outputResult, outputResult + numElements, expectedOutput.begin(), expectedOutput.end()));


    // Query the profiler again, this will contain the results of both inferences
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    dump = ss.str();

    // Contains Convolution2dWorkload
    found = dump.find("Convolution2dWorkload");
    CHECK(found != std::string::npos);

    // Should now contain the SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found != std::string::npos);

    // Should still contain a CopyMemGeneric from the first inference
    found = dump.find("CopyMemGeneric");
    CHECK(found != std::string::npos);
    runtime->UnloadNetwork(netId);
}

}
