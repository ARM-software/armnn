//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Graph.hpp>
#include <Network.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <GraphUtils.hpp>

#include <doctest/doctest.h>

TEST_SUITE("RefOptimizedNetwork")
{
TEST_CASE("OptimizeValidateCpuRefWorkloads")
{
    const armnn::TensorInfo desc({3, 5}, armnn::DataType::Float32);

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::NormalizationDescriptor nmDesc;
    armnn::ActivationDescriptor acDesc;

    //    in
    //     |
    //    nm
    //   /  |
    //  ac  |
    //   \  |
    //    ml
    //     |
    //    sm
    //     |
    //    ot
    armnn::IConnectableLayer* layer = net->AddInputLayer(0, "in");
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* const normLayer = net->AddNormalizationLayer(nmDesc, "nm");

    layer->GetOutputSlot(0).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(desc);

    layer = net->AddActivationLayer(acDesc, "ac");

    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* prevLayer = layer;
    layer = net->AddMultiplicationLayer("ml");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    armnn::SoftmaxDescriptor softmaxDescriptor;
    layer = net->AddSoftmaxLayer(softmaxDescriptor, "sm");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    layer = net->AddOutputLayer(0, "ot");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();
    CHECK(optNet);

    // Validates workloads.
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : graph)
    {
        CHECK_NOTHROW(layer->CreateWorkload(fact));
    }
}

TEST_CASE("OptimizeValidateWorkloadsCpuRefPermuteLayer")
{
    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::PermuteDescriptor descriptor({0, 2, 3, 1});
    armnn::IConnectableLayer* permute = net->AddPermuteLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(permute->GetInputSlot(0));
    permute->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    permute->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 4, 1, 4 }, armnn::DataType::Float32));

    // optimize the network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();

    for (auto&& layer : graph)
    {
        CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
    }
}

TEST_CASE("OptimizeValidateWorkloadsCpuRefMeanLayer")
{
    // Create runtime in which test will run
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    armnn::MeanDescriptor descriptor({ 0, 1 }, false);
    armnn::IConnectableLayer* meanLayer = net->AddMeanLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(meanLayer->GetInputSlot(0));
    meanLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 4, 3, 2 }, armnn::DataType::Float32));
    meanLayer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 2 }, armnn::DataType::Float32));

    // optimize the network
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();
    for (auto&& layer : graph)
    {
        CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
    }
}

TEST_CASE("DebugTestOnCpuRef")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;

    // Defines layers.
    auto input = net->AddInputLayer(0, "InputLayer");
    auto activation = net->AddActivationLayer(activation1Descriptor, "ActivationLayer");
    auto output = net->AddOutputLayer(0, "OutputLayer");

    // Connects layers.
    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

    armnn::OptimizerOptions optimizerOptions;
    optimizerOptions.m_Debug = true;

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);

    armnn::Graph& graph = GetGraphForTesting(optimizedNet.get());
    graph.AllocateDynamicBuffers();

    // Tests that all layers are present in the graph.
    CHECK(graph.GetNumLayers() == 5);

    // Tests that the vertices exist and have correct names.
    CHECK(GraphHasNamedLayer(graph, "InputLayer"));
    CHECK(GraphHasNamedLayer(graph, "DebugLayerAfterInputLayer_0"));
    CHECK(GraphHasNamedLayer(graph, "ActivationLayer"));
    CHECK(GraphHasNamedLayer(graph, "DebugLayerAfterActivationLayer_0"));
    CHECK(GraphHasNamedLayer(graph, "OutputLayer"));
}

}
