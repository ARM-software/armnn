//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Descriptors.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>

#include <Profiling.hpp>
#include <QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <boost/test/unit_test.hpp>

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
    const TensorInfo commonTensorInfo({ 2, 3 }, DataType::Float32);

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

    return ConstantUsageTest(backends,
        commonTensorInfo,
        armnnUtils::QuantizedVector<uint8_t>({ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f }, scale, offset), // Input.
        armnnUtils::QuantizedVector<uint8_t>({ 6.f, 5.f, 4.f, 3.f, 2.f, 1.f }, scale, offset), // Const input.
        armnnUtils::QuantizedVector<uint8_t>({ 7.f, 7.f, 7.f, 7.f, 7.f, 7.f }, scale, offset)  // Expected output.
    );
}

// Utility template for comparing tensor elements
template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
bool Compare(T a, T b, float tolerance = 0.000001f)
{
    if (ArmnnType == DataType::Boolean)
    {
        // NOTE: Boolean is represented as uint8_t (with zero equals
        // false and everything else equals true), therefore values
        // need to be casted to bool before comparing them
        return static_cast<bool>(a) == static_cast<bool>(b);
    }

    // NOTE: All other types can be cast to float and compared with
    // a certain level of tolerance
    return std::fabs(static_cast<float>(a) - static_cast<float>(b)) <= tolerance;
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
            BOOST_CHECK_MESSAGE(Compare<ArmnnOType>(it.second[i], out[i], tolerance) == true,
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(true, false);
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
    BOOST_CHECK_THROW(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryImportException);
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing and Exporting
    INetworkProperties networkProperties(true, true);
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
        BOOST_CHECK_THROW(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryImportException);
    }
    else
    {
        BOOST_CHECK_THROW(runtime->EnqueueWorkload(netId, inputTensors, outputTensors), MemoryExportException);
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(true, true);
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
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains ActivationWorkload
    std::size_t found = dump.find("ActivationWorkload");
    BOOST_TEST(found != std::string::npos);

    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Does not contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    BOOST_TEST(found == std::string::npos);

    // Check output is as expected
    BOOST_TEST(outputData == expectedOutput);
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    BOOST_TEST_CHECKPOINT("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, false);
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    BOOST_TEST_CHECKPOINT("Generate Data");
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

    BOOST_TEST_CHECKPOINT("Create Network");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    BOOST_TEST_CHECKPOINT("Get Profiler");

    runtime->GetProfiler(netId)->EnableProfiling(true);

    BOOST_TEST_CHECKPOINT("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    BOOST_TEST_CHECKPOINT("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Check there are no SyncMemGeneric workloads as we didn't export
    BOOST_TEST_CHECKPOINT("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    BOOST_TEST(count == 0);

    // Should only be 1 CopyMemGeneric for the output as we imported
    BOOST_TEST_CHECKPOINT("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    BOOST_TEST(count == 1);

    // Check the output is correct
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end());
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    BOOST_TEST_CHECKPOINT("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(false, true);
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    BOOST_TEST_CHECKPOINT("Generate Data");
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

    BOOST_TEST_CHECKPOINT("Create Network");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    BOOST_TEST_CHECKPOINT("Get Profiler");

    runtime->GetProfiler(netId)->EnableProfiling(true);

    BOOST_TEST_CHECKPOINT("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    BOOST_TEST_CHECKPOINT("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Check there is a SyncMemGeneric workload as we exported
    BOOST_TEST_CHECKPOINT("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    BOOST_TEST(count == 1);

    // Should be 1 CopyMemGeneric for the output as we did not import
    BOOST_TEST_CHECKPOINT("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    BOOST_TEST(count == 1);

    // Check the output is correct
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end());
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 1, 4 }, DataType::Float32));

    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    BOOST_TEST_CHECKPOINT("Load Network");
    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, true);
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet),ignoredErrorMessage, networkProperties)
               == Status::Success);

    BOOST_TEST_CHECKPOINT("Generate Data");
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

    BOOST_TEST_CHECKPOINT("Create Network");
    InputTensors inputTensors
    {
        {0,armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData.data())},
    };
    OutputTensors outputTensors
    {
        {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
    };

    BOOST_TEST_CHECKPOINT("Get Profiler");

    runtime->GetProfiler(netId)->EnableProfiling(true);

    BOOST_TEST_CHECKPOINT("Run Inference");
    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    BOOST_TEST_CHECKPOINT("Print Profiler");
    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Check there is a SyncMemGeneric workload as we exported
    BOOST_TEST_CHECKPOINT("Find SyncMemGeneric");
    int count = SubStringCounter(dump, "SyncMemGeneric");
    BOOST_TEST(count == 1);

    // Shouldn't be any CopyMemGeneric workloads
    BOOST_TEST_CHECKPOINT("Find CopyMemGeneric");
    count = SubStringCounter(dump, "CopyMemGeneric");
    BOOST_TEST(count == 0);

    // Check the output is correct
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData.begin(), outputData.end(), expectedOutput.begin(), expectedOutput.end());
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32));
    activation->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 1 }, DataType::Float32));

    // Optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    std::string ignoredErrorMessage;
    // Enable Importing
    INetworkProperties networkProperties(true, true);
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

    BOOST_TEST(found != std::string::npos);
    // No contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found == std::string::npos);
    // Contains CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Check that the outputs are correct
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData0.begin(), outputData0.end(),
                                  expectedOutput.begin(), expectedOutput.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(outputData1.begin(), outputData1.end(),
                                  expectedOutput.begin(), expectedOutput.end());
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

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 2, 3 }, DataType::Float32));
    stridedSlice->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 3 }, DataType::Float32));

    // Attempt to optimize the network and check that the correct exception is thrown
    BOOST_CHECK_THROW(Optimize(*net, backends, runtime->GetDeviceSpec()), armnn::LayerValidationException);
}

} // anonymous namespace
