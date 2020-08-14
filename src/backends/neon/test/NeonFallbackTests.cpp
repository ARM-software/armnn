//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/CommonTestUtils.hpp>
#include <backendsCommon/test/mockBackend/MockImportBackend.hpp>

#include <test/GraphUtils.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonFallback)

std::vector<armnn::BackendId> defaultBackends = { armnn::Compute::CpuAcc };

BOOST_AUTO_TEST_CASE(FallbackImportToCpuAcc)
{
    using namespace armnn;

    // Create a mock backend object
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        BOOST_FAIL(message);
    }

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* input2 = net->AddInputLayer(2, "input2");
    IConnectableLayer* add = net->AddAdditionLayer("add");
    IConnectableLayer* sub = net->AddSubtractionLayer("sub");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(sub->GetInputSlot(0));
    add->GetOutputSlot(0).Connect(sub->GetInputSlot(1));
    sub->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 3, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    sub->GetOutputSlot(0).SetTensorInfo(info);

    // optimize the network
    std::vector<BackendId> backends = { "MockRef", Compute::CpuAcc };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    OptimizedNetwork* optNetObjPtr = PolymorphicDowncast<OptimizedNetwork*>(optNet.get());
    Graph& graph = optNetObjPtr->GetGraph();

    armnn::Layer* const layer0 = GetFirstLayerWithName(graph, "input0");
    armnn::Layer* const layer1 = GetFirstLayerWithName(graph, "input1");
    armnn::Layer* const layer2 = GetFirstLayerWithName(graph, "input2");
    armnn::Layer* const layer3 = GetFirstLayerWithName(graph, "add");
    armnn::Layer* const layer4 = GetFirstLayerWithName(graph, "[ add (0) -> sub (1) ]");
    armnn::Layer* const layer5 = GetFirstLayerWithName(graph, "sub");
    armnn::Layer* const layer6 = GetFirstLayerWithName(graph, "output");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layer0, layer1));
    BOOST_TEST(CheckOrder(graph, layer1, layer2));
    BOOST_TEST(CheckOrder(graph, layer2, layer3));
    BOOST_TEST(CheckOrder(graph, layer3, layer4));
    BOOST_TEST(CheckOrder(graph, layer4, layer5));
    BOOST_TEST(CheckOrder(graph, layer5, layer6));

    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, true);

    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData0
    {
        1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f
    };
    std::vector<float> inputData1
    {
        0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f
    };
    std::vector<float> inputData2
    {
        12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };

    std::vector<float> outputData(12);

    std::vector<float> expectedOutput
    {
        11.0f, 9.0f, 7.0f, 5.0f, 3.0f, 1.0f, -1.0f, -3.0f, -5.0f, -7.0f, -9.0f, -11.0f
    };

    InputTensors inputTensors
    {
        { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData0.data()) },
        { 1, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 1), inputData1.data()) },
        { 2, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 2), inputData2.data()) }
    };
    OutputTensors outputTensors
    {
        { 0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data()) }
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains ImportMemGeneric
    std::size_t found = dump.find("ImportMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Does not contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    BOOST_TEST(found == std::string::npos);

    // Use memory import between backends
    BOOST_TEST((layer4->GetType() == LayerType::MemImport));

    // Check output is as expected
    BOOST_TEST(outputData == expectedOutput);
}

BOOST_AUTO_TEST_CASE(FallbackPaddingCopyToCpuAcc)
{
    using namespace armnn;

    // Create a mock backend object
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        BOOST_FAIL(message);
    }

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    Pooling2dDescriptor desc;

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* add = net->AddAdditionLayer("add");
    IConnectableLayer* pooling = net->AddPooling2dLayer(desc, "pooling");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 3, 2 }, DataType::Float32);
    TensorInfo poolingInfo = TensorInfo({ 1, 2, 1, 1 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    pooling->GetOutputSlot(0).SetTensorInfo(poolingInfo);

    // optimize the network
    std::vector<BackendId> backends = { "MockRef", Compute::CpuAcc };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    OptimizedNetwork* optNetObjPtr = PolymorphicDowncast<OptimizedNetwork*>(optNet.get());
    Graph& graph = optNetObjPtr->GetGraph();

    armnn::Layer* const layer0 = GetFirstLayerWithName(graph, "input0");
    armnn::Layer* const layer1 = GetFirstLayerWithName(graph, "input1");
    armnn::Layer* const layer2 = GetFirstLayerWithName(graph, "add");
    armnn::Layer* const layer3 = GetFirstLayerWithName(graph, "[ add (0) -> pooling (0) ]");
    armnn::Layer* const layer4 = GetFirstLayerWithName(graph, "pooling");
    armnn::Layer* const layer5 = GetFirstLayerWithName(graph, "output");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layer0, layer1));
    BOOST_TEST(CheckOrder(graph, layer1, layer2));
    BOOST_TEST(CheckOrder(graph, layer2, layer3));
    BOOST_TEST(CheckOrder(graph, layer3, layer4));
    BOOST_TEST(CheckOrder(graph, layer4, layer5));

    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, true);

    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData0
    {
        1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f
    };
    std::vector<float> inputData1
    {
        0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f
    };

    std::vector<float> outputData(2);

    std::vector<float> expectedOutput
    {
        6.0f, 12.0f
    };

    InputTensors inputTensors
    {
        { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData0.data()) },
        { 1, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 1), inputData1.data()) }
    };
    OutputTensors outputTensors
    {
        { 0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data()) }
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains CopyMemGeneric between the backends
    std::size_t found = dump.find("CopyMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Contains SyncMemGeneric for the output
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Does not contain ImportMemGeneric
    found = dump.find("ImportMemGeneric");
    BOOST_TEST(found == std::string::npos);

    // Use memory import between backends
    BOOST_TEST((layer3->GetType() == LayerType::MemCopy));

    // Check output is as expected
    BOOST_TEST(outputData == expectedOutput);
}

BOOST_AUTO_TEST_CASE(FallbackImportFromCpuAcc)
{
    using namespace armnn;

    // Create a mock backend object
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        BOOST_FAIL(message);
    }

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* input2 = net->AddInputLayer(2, "input2");
    IConnectableLayer* sub = net->AddSubtractionLayer("sub");
    IConnectableLayer* add = net->AddAdditionLayer("add");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(sub->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(sub->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    sub->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 3, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    sub->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    // optimize the network
    std::vector<BackendId> backends = { "MockRef", Compute::CpuAcc };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    OptimizedNetwork* optNetObjPtr = PolymorphicDowncast<OptimizedNetwork*>(optNet.get());
    Graph& graph = optNetObjPtr->GetGraph();

    armnn::Layer* const layer0 = GetFirstLayerWithName(graph, "input0");
    armnn::Layer* const layer1 = GetFirstLayerWithName(graph, "input1");
    armnn::Layer* const layer2 = GetFirstLayerWithName(graph, "input2");
    armnn::Layer* const layer3 = GetFirstLayerWithName(graph, "sub");
    armnn::Layer* const layer4 = GetFirstLayerWithName(graph, "[ sub (0) -> add (1) ]");
    armnn::Layer* const layer5 = GetFirstLayerWithName(graph, "add");
    armnn::Layer* const layer6 = GetFirstLayerWithName(graph, "output");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layer0, layer1));
    BOOST_TEST(CheckOrder(graph, layer1, layer2));
    BOOST_TEST(CheckOrder(graph, layer2, layer3));
    BOOST_TEST(CheckOrder(graph, layer3, layer4));
    BOOST_TEST(CheckOrder(graph, layer4, layer5));
    BOOST_TEST(CheckOrder(graph, layer5, layer6));

    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, true);

    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData0
    {
        1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 0.0f
    };
    std::vector<float> inputData1
    {
        0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f
    };
    std::vector<float> inputData2
    {
        12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };

    std::vector<float> outputData(12);

    std::vector<float> expectedOutput
    {
        13.0f, 11.0f, 11.0f, 9.0f, 7.0f, 7.0f, 7.0f, 5.0f, 5.0f, 3.0f, 3.0f, -5.0f
    };

    InputTensors inputTensors
    {
        { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData0.data()) },
        { 1, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 1), inputData1.data()) },
        { 2, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 2), inputData2.data()) }
    };
    OutputTensors outputTensors
    {
        { 0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data()) }
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains ImportMemGeneric
    std::size_t found = dump.find("ImportMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Contains SyncMemGeneric
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Does not contain CopyMemGeneric
    found = dump.find("CopyMemGeneric");
    BOOST_TEST(found == std::string::npos);

    // Use memory import between backends
    BOOST_TEST((layer4->GetType() == LayerType::MemImport));

    // Check output is as expected
    BOOST_TEST(outputData == expectedOutput);
}

BOOST_AUTO_TEST_CASE(FallbackPaddingCopyFromCpuAcc)
{
    using namespace armnn;

    // Create a mock backend object
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        BOOST_FAIL(message);
    }

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    Pooling2dDescriptor desc;

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* pooling = net->AddPooling2dLayer(desc, "pooling");
    IConnectableLayer* add = net->AddAdditionLayer("add");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    pooling->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo inputInfo = TensorInfo({ 1, 2, 3, 2 }, DataType::Float32);
    TensorInfo poolingInfo = TensorInfo({ 1, 2, 1, 1 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(inputInfo);
    input1->GetOutputSlot(0).SetTensorInfo(poolingInfo);
    pooling->GetOutputSlot(0).SetTensorInfo(poolingInfo);
    add->GetOutputSlot(0).SetTensorInfo(poolingInfo);

    // optimize the network
    std::vector<BackendId> backends = { "MockRef", Compute::CpuAcc };
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    OptimizedNetwork* optNetObjPtr = PolymorphicDowncast<OptimizedNetwork*>(optNet.get());
    Graph& graph = optNetObjPtr->GetGraph();

    armnn::Layer* const layer0 = GetFirstLayerWithName(graph, "input0");
    armnn::Layer* const layer1 = GetFirstLayerWithName(graph, "input1");
    armnn::Layer* const layer2 = GetFirstLayerWithName(graph, "pooling");
    armnn::Layer* const layer3 = GetFirstLayerWithName(graph, "[ pooling (0) -> add (0) ]");
    armnn::Layer* const layer4 = GetFirstLayerWithName(graph, "add");
    armnn::Layer* const layer5 = GetFirstLayerWithName(graph, "output");

    // Checks order is valid.
    BOOST_TEST(CheckOrder(graph, layer0, layer1));
    BOOST_TEST(CheckOrder(graph, layer1, layer2));
    BOOST_TEST(CheckOrder(graph, layer2, layer3));
    BOOST_TEST(CheckOrder(graph, layer3, layer4));
    BOOST_TEST(CheckOrder(graph, layer4, layer5));

    // Load it into the runtime. It should pass.
    NetworkId netId;
    std::string ignoredErrorMessage;
    INetworkProperties networkProperties(true, true);

    runtime->LoadNetwork(netId, std::move(optNet), ignoredErrorMessage, networkProperties);

    // Creates structures for input & output
    std::vector<float> inputData0
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
    };
    std::vector<float> inputData1
    {
        -1.0f, 3.0f
    };

    std::vector<float> outputData(2);

    std::vector<float> expectedOutput
    {
        5.0f, 15.0f
    };

    InputTensors inputTensors
    {
        { 0, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 0), inputData0.data()) },
        { 1, armnn::ConstTensor(runtime->GetInputTensorInfo(netId, 1), inputData1.data()) }
    };
    OutputTensors outputTensors
    {
        { 0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data()) }
    };

    runtime->GetProfiler(netId)->EnableProfiling(true);

    // Do the inference
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Retrieve the Profiler.Print() output to get the workload execution
    ProfilerManager& profilerManager = armnn::ProfilerManager::GetInstance();
    std::stringstream ss;
    profilerManager.GetProfiler()->Print(ss);;
    std::string dump = ss.str();

    // Contains CopyMemGeneric between the backends
    std::size_t found = dump.find("CopyMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Contains SyncMemGeneric for the output
    found = dump.find("SyncMemGeneric");
    BOOST_TEST(found != std::string::npos);

    // Does not contain ImportMemGeneric
    found = dump.find("ImportMemGeneric");
    BOOST_TEST(found == std::string::npos);

    // Use memory import between backends
    BOOST_TEST((layer3->GetType() == LayerType::MemCopy));

    // Check output is as expected
    BOOST_TEST(outputData == expectedOutput);
}

BOOST_AUTO_TEST_SUITE_END()
