//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#if defined(ARMNNREF_ENABLED)
#include <CommonTestUtils.hpp>
#include <GraphUtils.hpp>
#include <backendsCommon/test/mockBackend/MockImportBackend.hpp>
#endif

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("Fp32NetworkToFp16OptimizationTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo infoFP32({ 2, 2, 1, 3 }, armnn::DataType::Float32);

    // Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(infoFP32);

    auto floor = graph.AddLayer<armnn::FloorLayer>("floor");
    floor->GetOutputSlot().SetTensorInfo(infoFP32);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::FloorLayer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(Fp32NetworkToFp16Converter()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::FloorLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    CHECK(floor->GetDataType() == armnn::DataType::Float16);
    CHECK(floor->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType() == armnn::DataType::Float16);
    CHECK(floor->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
}

#if defined(ARMNNREF_ENABLED)
TEST_CASE("ReduceFp32ToFp16EnabledBackendHasFp16SupportTest")
{
    using namespace armnn;
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* input2 = net->AddInputLayer(2, "input2");
    IConnectableLayer* add = net->AddElementwiseBinaryLayer(BinaryOperation::Add, "add");
    IConnectableLayer* sub = net->AddElementwiseBinaryLayer(BinaryOperation::Sub, "sub");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(sub->GetInputSlot(0));
    add->GetOutputSlot(0).Connect(sub->GetInputSlot(1));
    sub->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    sub->GetOutputSlot(0).SetTensorInfo(info);

    std::vector<BackendId> backends = { Compute::CpuRef };

    // optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetReduceFp32ToFp16(true);
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optOptions);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Layers are added to convert the inputs to FP16
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    Layer* const addLayer = GetFirstLayerWithName(graph, "add");
    Layer* const subLayer = GetFirstLayerWithName(graph, "sub");

    CHECK(addLayer->GetDataType() == armnn::DataType::Float16);
    CHECK(addLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float16);
    CHECK(addLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    CHECK(subLayer->GetDataType() == armnn::DataType::Float16);
    CHECK(subLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float16);
    CHECK(subLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
}

TEST_CASE("ReduceFp32ToFp16EnabledBackendNoFp16SupportTest")
{
    using namespace armnn;

    // Create a mock backend without FP16 support
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    CHECK((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        FAIL(message);
    }

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* add = net->AddElementwiseBinaryLayer(BinaryOperation::Add, "add");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    std::vector<BackendId> backends = { "MockRef" };

    // optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetReduceFp32ToFp16(true);
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optOptions);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Do not add layers to convert the inputs to FP16
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    // Checks that data type is FP32
    Layer* const addLayer = GetFirstLayerWithName(graph, "add");

    CHECK(addLayer->GetDataType() == armnn::DataType::Float32);
    CHECK(addLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float32);
    CHECK(addLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float32);
}

TEST_CASE("ReduceFp32ToFp16EnabledFirstBackendHasFp16SupportTest")
{
    using namespace armnn;

    // Create a mock backend without FP16 support
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    CHECK((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        FAIL(message);
    }
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* input2 = net->AddInputLayer(2, "input2");
    IConnectableLayer* add = net->AddElementwiseBinaryLayer(BinaryOperation::Add, "add");
    IConnectableLayer* sub = net->AddElementwiseBinaryLayer(BinaryOperation::Sub, "sub");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(sub->GetInputSlot(0));
    add->GetOutputSlot(0).Connect(sub->GetInputSlot(1));
    sub->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    sub->GetOutputSlot(0).SetTensorInfo(info);

    std::vector<BackendId> backends = { Compute::CpuRef, "MockRef" };

    // optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetReduceFp32ToFp16(true);
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optOptions);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Layers are added to convert the inputs to FP16
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    Layer* const addLayer = GetFirstLayerWithName(graph, "add");
    Layer* const subLayer = GetFirstLayerWithName(graph, "sub");

    CHECK(addLayer->GetDataType() == armnn::DataType::Float16);
    CHECK(addLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float16);
    CHECK(addLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    CHECK(subLayer->GetDataType() == armnn::DataType::Float16);
    CHECK(subLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float16);
    CHECK(subLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
}

TEST_CASE("ReduceFp32ToFp16EnabledFirstBackendNoFp16SupportTest")
{
    using namespace armnn;

    // Create a mock backend without FP16 support
    MockImportBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockImportBackendId());
    CHECK((backendObjPtr != nullptr));

    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("MockRef") == backendIds.end())
    {
        std::string message = "Cannot load MockRef";
        FAIL(message);
    }

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* input2 = net->AddInputLayer(2, "input2");
    IConnectableLayer* add = net->AddElementwiseBinaryLayer(BinaryOperation::Add, "add");
    IConnectableLayer* sub = net->AddElementwiseBinaryLayer(BinaryOperation::Sub, "sub");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    input2->GetOutputSlot(0).Connect(sub->GetInputSlot(0));
    add->GetOutputSlot(0).Connect(sub->GetInputSlot(1));
    sub->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    input2->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);
    sub->GetOutputSlot(0).SetTensorInfo(info);

    std::vector<BackendId> backends = { "MockRef", Compute::CpuRef };

    // optimize the network
    OptimizerOptionsOpaque optOptions;
    optOptions.SetReduceFp32ToFp16(true);
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec(), optOptions);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Do not add layers to convert the inputs to FP16
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    // Checks that data type is FP32
    Layer* const addLayer = GetFirstLayerWithName(graph, "add");
    Layer* const subLayer = GetFirstLayerWithName(graph, "sub");

    CHECK(addLayer->GetDataType() == armnn::DataType::Float32);
    CHECK(addLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float32);
    CHECK(addLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    CHECK(subLayer->GetDataType() == armnn::DataType::Float32);
    CHECK(subLayer->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType()
            == armnn::DataType::Float32);
    CHECK(subLayer->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float32);
}
#endif // ARMNNREF_ENABLED

}