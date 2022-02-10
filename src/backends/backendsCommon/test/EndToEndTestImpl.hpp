//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>

#include <Profiling.hpp>
#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

using namespace armnn;

template<typename T>
bool ConstantUsageTest(const std::vector<BackendId>& computeDevice,
                       const TensorInfo& commonTensorInfo,
                       const std::vector<T>& inputData,
                       const std::vector<T>& constantData,
                       const std::vector<T>& expectedOutputData)
{
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);
    IConnectableLayer* constant = net->AddConstantLayer(ConstTensor(commonTensorInfo, constantData));
    IConnectableLayer* add = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Sets the tensors in the network.
    input->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    constant->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(commonTensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, computeDevice, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    // Creates structures for input & output.
    std::vector<T> outputData(inputData.size());

    InputTensors inputTensors
    {
        {0, ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())}
    };
    OutputTensors outputTensors
    {
        {0, Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    return outputData == expectedOutputData;
}

inline bool ConstantUsageFloat32Test(const std::vector<BackendId>& backends)
{
    TensorInfo commonTensorInfo({ 2, 3 }, DataType::Float32);
    commonTensorInfo.SetConstant(true);

    return ConstantUsageTest(backends,
        commonTensorInfo,
        std::vector<float>{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, // Input.
        std::vector<float>{ 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }, // Const input.
        std::vector<float>{ 7.f, 7.f, 7.f, 7.f, 7.f, 7.f }  // Expected output.
    );
}

inline bool ConstantUsageUint8Test(const std::vector<BackendId>& backends)
{
    TensorInfo commonTensorInfo({ 2, 3 }, DataType::QAsymmU8);

    const float scale = 0.023529f;
    const int8_t offset = -43;

    commonTensorInfo.SetQuantizationScale(scale);
    commonTensorInfo.SetQuantizationOffset(offset);
    commonTensorInfo.SetConstant(true);

    return ConstantUsageTest(backends,
        commonTensorInfo,
        armnnUtils::QuantizedVector<uint8_t>({ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, scale, offset), // Input.
        armnnUtils::QuantizedVector<uint8_t>({ 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }, scale, offset), // Const input.
        armnnUtils::QuantizedVector<uint8_t>({ 7.f, 7.f, 7.f, 7.f, 7.f, 7.f }, scale, offset)  // Expected output.
    );
}

// Utility function to find the number of instances of a substring within a string.
int SubStringCounter(std::string& string, std::string&& substring)
{
    std::size_t found = 0;
    int count = 0;
    // Look for the substring starting from where we last found the substring
    while((found = string.find(substring, found)) != std::string::npos)
    {
        count++;
        // Offset by substring length to avoid finding the same substring twice
        found += substring.length();
    }
    return count;
}

template<DataType ArmnnIType, DataType ArmnnOType,
         typename TInput = ResolveType<ArmnnIType>, typename TOutput = ResolveType<ArmnnOType>>
void EndToEndLayerTestImpl(INetworkPtr network,
                           const std::map<int, std::vector<TInput>>& inputTensorData,
                           const std::map<int, std::vector<TOutput>>& expectedOutputData,
                           std::vector<BackendId> backends,
                           float tolerance = 0.000001f)
{
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    InputTensors inputTensors;
    inputTensors.reserve(inputTensorData.size());
    for (auto&& it : inputTensorData)
    {
        inputTensors.push_back({it.first,
                                ConstTensor(runtime->GetInputTensorInfo(netId, it.first), it.second.data())});
    }
    OutputTensors outputTensors;
    outputTensors.reserve(expectedOutputData.size());
    std::map<int, std::vector<TOutput>> outputStorage;
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out(it.second.size());
        outputStorage.emplace(it.first, out);
        outputTensors.push_back({it.first,
                                 Tensor(runtime->GetOutputTensorInfo(netId, it.first),
                                               outputStorage.at(it.first).data())});
    }

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    for (auto&& it : expectedOutputData)
    {
        std::vector<TOutput> out = outputStorage.at(it.first);
        for (unsigned int i = 0; i < out.size(); ++i)
        {
            CHECK_MESSAGE(Compare<ArmnnOType>(it.second[i], out[i], tolerance) == true,
                    "Actual output: " << out[i] << ". Expected output:" << it.second[i]);

        }
    }
}

inline void ImportNonAlignedInputPointerTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Undefined);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    // Misaligned input
    float* misalignedInputData = reinterpret_cast<float*>(reinterpret_cast<char*>(inputData.data()) + 1);

    std::vector<float> outputData(4);

    // Aligned output
    float* alignedOutputData = outputData.data();

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), misalignedInputData)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), alignedOutputData)}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference and expect it to fail with a ImportMemoryException
    CHECK_THROWS_AS(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryImportException);
}

inline void ExportNonAlignedOutputPointerTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing and Exporting
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f
    };

    // Aligned input
    float* alignedInputData = inputData.data();

    std::vector<float> outputData(5);

    // Misaligned output
    float* misalignedOutputData = reinterpret_cast<float*>(reinterpret_cast<char*>(outputData.data()) + 1);

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), alignedInputData)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), misalignedOutputData)}
    };

    // Do the inference and expect it to fail with a ExportMemoryException
    if (backends[0] == Compute::CpuAcc)
    {
        // For CpuAcc the NeonTensorHandle will throw its own exception on misaligned memory
        CHECK_THROWS_AS(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryImportException);
    }
    else
    {
        CHECK_THROWS_AS(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryExportException);
    }
}

inline void ImportAlignedPointerTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData(4);

    std::vector<float> expectedOutput
    {
        1.0f, 4.0f, 9.0f, 16.0f
    };

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
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

    // Check output is as expected
    CHECK(outputData == expectedOutput);
}

inline void ImportOnlyWorkload(std::vector<BackendId> backends)
{
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;

    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Undefined);

    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    INFO("Generate Data");
    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData(4);

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create Inference");

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    INFO("Get Profiler");
    runtime->GetProfiler(netId)->EnableProfiling(true);

    INFO("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    INFO("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // Check there are no SyncMemGeneric workloads as we didn't export
    INFO("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    CHECK(count == 0);

    // Should only be 1 CopyMemGeneric for the output as we imported
    INFO("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    CHECK(count == 1);

    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
}

inline void ExportOnlyWorkload(std::vector<BackendId> backends)
{
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Malloc);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    INFO("Generate Data");
    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData(4);

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create Inference");

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    INFO("Get Profiler");
    runtime->GetProfiler(netId)->EnableProfiling(true);

    INFO("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    INFO("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // Check there is a SyncMemGeneric workload as we exported
    INFO("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    CHECK(count == 1);

    // Should be 1 CopyMemGeneric for the output as we did not import
    INFO("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    CHECK(count == 1);

    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
}

inline void ImportAndExportWorkload(std::vector<BackendId> backends)
{
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* pooling = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;

    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);

    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    INFO("Generate Data");
    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData(4);

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create inference");

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    INFO("Get Profiler");
    runtime->GetProfiler(netId)->EnableProfiling(true);

    INFO("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    INFO("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // Check there is a SyncMemGeneric workload as we exported
    INFO("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    CHECK(count == 1);

    // Shouldn't be any CopyMemGeneric workloads
    INFO("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    CHECK(count == 0);

    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
}

inline void ExportOutputWithSeveralOutputSlotConnectionsTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activation = net->AddActivationLayer(descriptor);

    IConnectableLayer* output0 = net->AddOutputLayer(0);
    IConnectableLayer* output1 = net->AddOutputLayer(1);

    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output0->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output1->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32, 0.0f, 0, true));
    activation->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(false, MemorySource::Malloc, MemorySource::Malloc);
    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    std::vector<float> outputData0(4);
    std::vector<float> outputData1(4);

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData0.data())},
        {1,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 1), outputData1.data())}
    };

    // The result of the inference is not important, just the fact that there
    // should not be CopyMemGeneric workloads.
    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    std::size_t found = std::string::npos;

    if (backends[0] == Compute::CpuRef)
    {
        found = dump.find("RefActivationWorkload");
    }
    else if (backends[0] == Compute::CpuAcc)
    {
        found = dump.find("NeonActivationWorkload");
    }
    else if (backends[0] == Compute::GpuAcc)
    {
        found = dump.find("ClActivationWorkload");
    }

    CHECK(found != std::string::npos);
    // No contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    CHECK(found == std::string::npos);
    // Contains CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    CHECK(found != std::string::npos);

    // Check that the outputs are correct
    CHECK(std::equal(outputData0.begin(), outputData0.end(),
                                  expectedOutput.begin(), expectedOutput.end()));
    CHECK(std::equal(outputData1.begin(), outputData1.end(),
                                  expectedOutput.begin(), expectedOutput.end()));
}

inline void StridedSliceInvalidSliceEndToEndTest(std::vector<BackendId> backends)
{
    using namespace armnn;

    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(armnn::IRuntime::Create(options));

    // build up the structure of the network
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // Configure a strided slice with a stride the same size as the input but with a ShrinkAxisMask on the first
    // dim of the output to make it too small to hold the specified slice.
    StridedSliceDescriptor descriptor;
    descriptor.m_Begin          = {0, 0};
    descriptor.m_End            = {2, 3};
    descriptor.m_Stride         = {1, 1};
    descriptor.m_BeginMask      = 0;
    descriptor.m_EndMask        = 0;
    descriptor.m_ShrinkAxisMask = 1;
    IConnectableLayer* stridedSlice = net->AddStridedSliceLayer(descriptor);

    IConnectableLayer* output0 = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(stridedSlice->GetInputSlot(0));
    stridedSlice->GetOutputSlot(0).Connect(output0->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 2, 3 }, DataType::Float32, 0.0f, 0, true));
    stridedSlice->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 3 }, DataType::Float32));

    // Attempt to optimize the network and check that the correct exception is thrown
    CHECK_THROWS_AS(Optimize(*net, backends, runtime->GetDeviceSpec()), armnn::LayerValidationException);
}

inline void ForceImportWithAlignedBuffersEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this case all inputs and outputs should be imported
     */
    using namespace armnn;
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);
    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);
    IConnectableLayer* output = net->AddOutputLayer(0);
    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");

    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> outputData(4);
    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    // Check our input and output pointers are actually aligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK(!(reinterpret_cast<uintptr_t>(inputData.data()) % alignment));
    CHECK(!(reinterpret_cast<uintptr_t>(outputData.data()) % alignment));

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // Do the inference and force the import as the memory is aligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    if (backends[0] == Compute::CpuAcc)
    {
        // Reconfigure has not been implemented for CpuAcc so it will always copy, this will break whenever
        // reconfigure is implemented
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should be 2 CopyMemGeneric workloads
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count == 2);
    }
    else
    {
        // Check there is a SyncMemGeneric workload as we exported
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 1);
        // Shouldn't be any CopyMemGeneric workloads
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count == 0);
    }
    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
}

inline void ForceImportWithMisalignedInputBuffersEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this case all only the output should be imported
     */
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // This code looks a little funky but the idea is to create a buffer of floats but offset by the size of a char
    // this will guarantee that the resultant buffer is misaligned and thus should always be copied.
    auto memPtr = std::malloc(4 * sizeof(float) + sizeof(char));

    float* misalignedMemPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(memPtr) + 1);

    // Check if our pointer is truly misaligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK (reinterpret_cast<uintptr_t>(misalignedMemPtr) % alignment);

    std::vector<float> inputData
    {
         1.0f, 2.0f, 3.0f, 4.0f
    };

    std::memcpy(misalignedMemPtr, inputData.data(), 4*sizeof(float));

    std::vector<float> outputData(4);
    // Check our output buffer is aligned
    CHECK(!(reinterpret_cast<uintptr_t>(outputData.data()) % alignment));

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), misalignedMemPtr)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };
    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);

    // Do the inference and force the import as the memory is misaligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // GpuAcc is a different case to CpuRef and CpuAcc, it doesn't use the buffer directly but instead maps it to a
    // new set of addresses within Gpu Memory. This will almost always be auto-aligned, so we don't need to check
    // for imports/copies. Only that the output is correct.
    if (backends[0] != Compute::GpuAcc)
    {
        if (backends[0] == Compute::CpuAcc)
        {
            // Reconfigure has not been implemented for CpuAcc so it will always copy, this will break whenever
            // reconfigure is implemented
            // We should get 0 SyncMemGeneric for the Output
            int count = SubStringCounter(dump, "SyncMemGeneric");
            CHECK(count == 0);
            // Should be 2 CopyMemGeneric as we copied the input
            count = SubStringCounter(dump, "CopyMemGeneric");
            CHECK(count == 2);
        }
        else
        {
            // We should get 1 SyncMemGeneric for the Output
            int count = SubStringCounter(dump, "SyncMemGeneric");
            CHECK(count == 1);
            // Should only be 1 CopyMemGeneric as we copied the input
            count = SubStringCounter(dump, "CopyMemGeneric");
            CHECK(count == 1);
        }
    }
    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
    std::free(memPtr);
}

inline void ForceImportWithMisalignedOutputBuffersEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this case all only the input should be imported
     */
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // This code looks a little funky but the idea is to create a buffer of floats but offset by the size of a char
    // this will guarantee that the resultant buffer is misaligned and thus should always be copied.
    auto memPtr = std::malloc(4 * sizeof(float) + sizeof(char));

    float* misalignedMemPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(memPtr) + 1);

    // Check if our pointer is truly misaligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK (reinterpret_cast<uintptr_t>(misalignedMemPtr) % alignment);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };

    // Check our input buffer is aligned
    CHECK(!(reinterpret_cast<uintptr_t>(inputData.data()) % alignment));
    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), misalignedMemPtr)}
    };
    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);

    // Do the inference and force the import as the memory is misaligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // GpuAcc is a different case to CpuRef and CpuAcc, it doesn't use the buffer directly but instead maps it to a
    // new set of addresses within Gpu Memory. This will almost always be auto-aligned, so we don't need to check
    // for imports/copies. Only that the output is correct.
    if (backends[0] != Compute::GpuAcc)
    {
        // Even though we Imported the Input we still shouldn't have a SyncMemGeneric
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should only be 1 CopyMemGeneric as we copied the input
        count = SubStringCounter(dump, "CopyMemGeneric");
        if (backends[0] == Compute::CpuAcc)
        {
            // Reconfigure has not been implemented for CpuAcc so it will always copy, this will break whenever
            // reconfigure is implemented
            CHECK(count == 2);
        }
        else
        {
            CHECK(count == 1);
        }
        // Check the output is correct
    }
    unsigned int index = 0;
    std::vector<float> outputData(expectedOutput.size(), 0);
    std::memcpy(outputData.data(), misalignedMemPtr, expectedOutput.size() * sizeof(float));
    for (auto outputValue : expectedOutput)
    {
        CHECK(outputValue == outputData[index]);
        ++index;
    }
    std::free(memPtr);
}

inline void ForceImportWithMisalignedInputAndOutputBuffersEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this case all inputs and outputs should be copied
     */
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // This code looks a little funky but the idea is to create a buffer of floats but offset by the size of a char
    // this will guarantee that the resultant buffer is misaligned and thus should always be copied.
    auto inputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedInputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(inputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK (reinterpret_cast<uintptr_t>(misalignedInputPtr) % alignment);
    std::vector<float> inputData
    {
         1.0f, 2.0f, 3.0f, 4.0f
    };
    std::memcpy(misalignedInputPtr, inputData.data(), 4*sizeof(float));

    auto outputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedOutputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(outputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    CHECK (reinterpret_cast<uintptr_t>(misalignedOutputPtr) % alignment);

    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), misalignedInputPtr)},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), misalignedOutputPtr)}
    };
    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);

    // Do the inference and force the import as the memory is misaligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);
    std::string dump = ss.str();

    // GpuAcc is a different case to CpuRef and CpuAcc, it doesn't use the buffer directly but instead maps it to a
    // new set of addresses within Gpu Memory. This will almost always be auto-aligned, so we don't need to check
    // for imports/copies. Only that the output is correct.
    if (backends[0] != Compute::GpuAcc)
    {
        // We can only copy so there should be no SyncMemGeneric
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should only be CopyMemGeneric workloads as we copied all buffers
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count == 2);
    }
    // Check the output is correct
    unsigned int index = 0;
    std::vector<float> outputData(expectedOutput.size(), 0);
    std::memcpy(outputData.data(), misalignedOutputPtr, expectedOutput.size() * sizeof(float));
    for (auto expectedValue : expectedOutput)
    {
        CHECK(expectedValue == outputData[index]);
        ++index;
    }
    std::free(inputMemPtr);
    std::free(outputMemPtr);
}

inline void ForceImportRepeatedInferencesEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this we create some aligned buffers, import them into a network and validate the output and number of
     * SynMemGeneric/CopyMemgeneric. Then we try the same network again with misaligned buffers to make sure it falls
     * back to copying correctly.
     */
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> outputData(4);
    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    // Check our input and output pointers are actually aligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK(!(reinterpret_cast<uintptr_t>(inputData.data()) % alignment));
    CHECK(!(reinterpret_cast<uintptr_t>(outputData.data()) % alignment));

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId> importedInputIds =
        runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // Do the inference and force the import as the memory is aligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    std::string dump = ss.str();

    if (backends[0] == Compute::CpuAcc)
    {
        // Reconfigure has not been implemented for CpuAcc so it will always copy, this will break whenever
        // reconfigure is implemented
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should be 2 CopyMemGeneric workloads
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count >= 1);
    }
    else
    {
        // Check there is at least 1 SyncMemGeneric workload as we exported
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count >= 1);
        // Shouldn't be any CopyMemGeneric workloads
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count == 0);
    }
    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));

    // This code looks a little funky but the idea is to create a buffer of floats but offset by the size of a char
    // this will guarantee that the resultant buffer is misaligned and thus should always be copied.
    auto inputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedInputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(inputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    CHECK (reinterpret_cast<uintptr_t>(misalignedInputPtr) % alignment);

    std::vector<float> inputValues
    {
         2.0f, 3.0f, 4.0f, 5.0f
    };

    std::memcpy(misalignedInputPtr, inputValues.data(), inputValues.size()*sizeof(float));

    auto outputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedOutputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(outputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    CHECK (reinterpret_cast<uintptr_t>(misalignedOutputPtr) % alignment);

    std::vector<float> expectedMisalignedOutput
    {
         4.0f, 9.0f, 16.0f, 25.0f
    };

    INFO("Create Second Inference");
    InputTensors inputTensorsMisaligned
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), misalignedInputPtr)},
    };
    OutputTensors outputTensorsMisaligned
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), misalignedOutputPtr)}
    };
    importedInputIds = runtime->ImportInputs(netId, inputTensorsMisaligned, MemorySource::Malloc);
    importedOutputIds = runtime->ImportOutputs(netId, outputTensorsMisaligned, MemorySource::Malloc);

    // Do the inference and force the import as the memory is misaligned.
    runtime->EnqueueWorkload(netId,
                             inputTensorsMisaligned,
                             outputTensorsMisaligned,
                             importedInputIds,
                             importedOutputIds);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    // We need to use AnalyzeEventsAndWriteResults here to make sure the second inference has been profiled
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    dump = ss.str();

    // GpuAcc is a different case to CpuRef and CpuAcc, it doesn't use the buffer directly but instead maps it to a
    // new set of addresses within Gpu Memory. This will almost always be auto-aligned, so we don't need to check
    // for imports/copies. Only that the output is correct.
    if (backends[0] != Compute::GpuAcc)
    {
        // The SyncMemGeneric will still be in the profiling log from the first inference
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count >= 1);
        // We should now see CopyMemGeneric workloads as we copied all buffers
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count >= 1);
    }
    // Check the output is correct
    unsigned int index = 0;
    std::vector<float> alignedOutputData(expectedMisalignedOutput.size(), 0);
    std::memcpy(alignedOutputData.data(), misalignedOutputPtr, expectedMisalignedOutput.size() * sizeof(float));
    for (auto outputValue : expectedMisalignedOutput)
    {
        CHECK(outputValue == alignedOutputData[index]);
        ++index;
    }
    // Clean up to avoid interfering with other tests
    runtime->UnloadNetwork(netId);
    std::free(inputMemPtr);
    std::free(outputMemPtr);
}


inline void ForceImportRepeatedInferencesInvertedEndToEndTest(std::vector<BackendId> backends)
{
    /**
     * This test is similar to the Import tests above, we create a network with a square function and pass in a vector
     * with 4 floats, square them. and validate the output. We then check the profiling logs to see if input/output
     * tensors are copied (CopyMemGeneric) or imported (SyncMemGeneric)
     * In this we create some misaligned buffers, copy them into a network and validate the output and number of
     * SynMemGeneric/CopyMemgeneric. Then we try the same network again with aligned buffers to make sure it switches
     * to importing correctly.
     */
    using namespace armnn;

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* input = net->AddInputLayer(0);

    ActivationDescriptor descriptor;
    descriptor.m_Function = ActivationFunction::Square;
    IConnectableLayer* activationLayer = net->AddActivationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
    activationLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32, 0.0f, 0, true));
    activationLayer->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    INFO("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, MemorySource::Undefined, MemorySource::Undefined);
    CHECK(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);
    INFO("Generate Data");

    // This code looks a little funky but the idea is to create a buffer of floats but offset by the size of a char
    // this will guarantee that the resultant buffer is misaligned and thus should always be copied.
    auto inputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedInputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(inputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    uintptr_t alignment = GetDataTypeSize(DataType::Float32);
    CHECK (reinterpret_cast<uintptr_t>(misalignedInputPtr) % alignment);
    std::vector<float> inputValues
    {
         2.0f, 3.0f, 4.0f, 5.0f
    };
    std::memcpy(misalignedInputPtr, inputValues.data(), inputValues.size() * sizeof(float));

    auto outputMemPtr = std::malloc(4 * sizeof(float) + sizeof(char));
    float* misalignedOutputPtr = reinterpret_cast<float*>(reinterpret_cast<char*>(outputMemPtr) + 1);

    // Check if our pointer is truly misaligned
    CHECK (reinterpret_cast<uintptr_t>(misalignedOutputPtr) % alignment);

    std::vector<float> expectedMisalignedOutput
    {
         4.0f, 9.0f, 16.0f, 25.0f
    };

    INFO("Create Second Inference");
    InputTensors inputTensorsMisaligned
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), misalignedInputPtr)},
    };
    OutputTensors outputTensorsMisaligned
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), misalignedOutputPtr)}
    };
    runtime->GetProfiler(netId)->EnableProfiling(true);
    std::vector<ImportedInputId>  importedInputIds =
        runtime->ImportInputs(netId, inputTensorsMisaligned, MemorySource::Malloc);
    std::vector<ImportedOutputId> importedOutputIds =
        runtime->ImportOutputs(netId, outputTensorsMisaligned, MemorySource::Malloc);

    // Do the inference and force the import as the memory is misaligned.
    runtime->EnqueueWorkload(netId,
                             inputTensorsMisaligned,
                             outputTensorsMisaligned,
                             importedInputIds,
                             importedOutputIds);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    std::string dump = ss.str();

    // GpuAcc is a different case to CpuRef and CpuAcc, it doesn't use the buffer directly but instead maps it to a
    // new set of addresses within Gpu Memory. This will almost always be auto-aligned, so we don't need to check
    // for imports/copies. Only that the output is correct.
    if (backends[0] != Compute::GpuAcc)
    {
        // We can only copy so there should be no SyncMemGeneric
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should only be CopyMemGeneric workloads as we copied all buffers
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count >= 1);
    }
    // Check the output is correct
    unsigned int index = 0;
    std::vector<float> alignedOutput(expectedMisalignedOutput.size());
    std::memcpy(alignedOutput.data(), misalignedOutputPtr, expectedMisalignedOutput.size()*sizeof(float));
    for (auto outputValue : expectedMisalignedOutput)
    {
        CHECK(outputValue == alignedOutput[index]);
        ++index;
    }
    std::free(inputMemPtr);
    std::free(outputMemPtr);

    // Creates structures for input & output
    std::vector<float> inputData
    {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> outputData(4);
    std::vector<float> expectedOutput
    {
         1.0f, 4.0f, 9.0f, 16.0f
    };

    // Check our input and output pointers are actually aligned
    CHECK(!(reinterpret_cast<uintptr_t>(inputData.data()) % alignment));
    CHECK(!(reinterpret_cast<uintptr_t>(outputData.data()) % alignment));

    INFO("Create Inference");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    importedInputIds = runtime->ImportInputs(netId, inputTensors, MemorySource::Malloc);
    importedOutputIds = runtime->ImportOutputs(netId, outputTensors, MemorySource::Malloc);
    // Do the inference and force the import as the memory is aligned.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors, importedInputIds, importedOutputIds);

    // Retrieve the Profiler.AnalyzeEventsAndWriteResults() output to get the workload execution
    // We need to use AnalyzeEventsAndWriteResults here to make sure the second inference has been profiled
    profilerManager.GetProfiler()->AnalyzeEventsAndWriteResults(ss);
    dump = ss.str();

    if (backends[0] == Compute::CpuAcc)
    {
        // Reconfigure has not been implemented for CpuAcc so it will always copy, this will break whenever
        // reconfigure is implemented
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count == 0);
        // Should be 2 CopyMemGeneric workloads
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count >= 1);
    }
    else
    {
        // Repeated inferences make it difficult to check for an accurate count. So we just validate that we have a
        // SyncMemGeneric Workload when we previously didn't
        int count = SubStringCounter(dump, "SyncMemGeneric");
        CHECK(count >= 1);
        // Should still be some CopyMemGeneric Workloads from the last inference
        count = SubStringCounter(dump, "CopyMemGeneric");
        CHECK(count >= 1);
    }
    // Check the output is correct
    CHECK(std::equal(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end()));
    // Clean up to avoid interfering with other tests
    runtime->UnloadNetwork(netId);
}

} // anonymous namespace
