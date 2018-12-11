//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ArmNN.hpp>
#include <Graph.hpp>
#include <Network.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <boost/test/unit_test.hpp>
#include <test/GraphUtils.hpp>

BOOST_AUTO_TEST_SUITE(RefOptimizedNetwork)

BOOST_AUTO_TEST_CASE(OptimizeValidateCpuRefWorkloads)
{
    const armnn::TensorInfo desc({3, 5}, armnn::DataType::Float32);

    armnn::Network  net;

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
    armnn::IConnectableLayer* layer = net.AddInputLayer(0, "in");
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* const normLayer = net.AddNormalizationLayer(nmDesc, "nm");

    layer->GetOutputSlot(0).Connect(normLayer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).SetTensorInfo(desc);

    layer = net.AddActivationLayer(acDesc, "ac");

    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    armnn::IConnectableLayer* prevLayer = layer;
    layer = net.AddMultiplicationLayer("ml");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    normLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    armnn::SoftmaxDescriptor softmaxDescriptor;
    layer = net.AddSoftmaxLayer(softmaxDescriptor, "sm");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));
    layer->GetOutputSlot(0).SetTensorInfo(desc);

    prevLayer = layer;
    layer = net.AddOutputLayer(0, "ot");

    prevLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec());
    static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph().AllocateDynamicBuffers();
    BOOST_CHECK(optNet);

    // Validates workloads.
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph(), fact));
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsCpuRefPermuteLayer)
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

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateWorkloadsCpuRefMeanLayer)
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

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
    }
}

BOOST_AUTO_TEST_CASE(FP16TurboModeTestOnCpuRef)
{
    // Test to check when FP16 Turbo mode set
    // it converts the FP32 network to FP16 Network
    // add FP32ToFP16 conversion layer after the InputLayer
    // add FP16ToFP32 conversion layer after the OutputLayer
    // checks the other layers if they are supported in FP16
    // if they are not put the conversion layers before and after
    // if they are not supported in FP16 use FP32 instead
    // if there are inverse conversion layers remove them with optimization
    // at the moment FloorLayer is not supported in FP16 so it rolls back to FP32
    // and inverse conversion layers are removed by the optimizer
    armnn::Network net;

    // Defines layers.
    auto input = net.AddInputLayer(0);
    auto floor = net.AddFloorLayer();
    auto output = net.AddOutputLayer(0);

    // Connects layers.
    input->GetOutputSlot(0).Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    floor->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};

    armnn::OptimizerOptions optimizerOptions;
    optimizerOptions.m_ReduceFp32ToFp16 = true;

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);

    std::ostringstream ss;
    optimizedNet->SerializeToDot(ss);

    auto inputId = input->GetGuid();
    auto floorId = floor->GetGuid();
    auto outputId = output->GetGuid();

    std::stringstream expected;
    expected <<
             "digraph Optimized {\n"
             "    node [shape=\"record\"];\n"
             "    edge [fontsize=8 fontcolor=\"blue\" fontname=\"arial-bold\"];\n"
             "    " << inputId << " [label=\"{Input}\"];\n"
             "    " << floorId << " [label=\"{Floor}\"];\n"
             "    " << outputId << " [label=\"{Output}\"];\n"
             "    " << inputId << " -> " << floorId << " [label=< [4] >];\n"
             "    " << floorId << " -> " << outputId << " [label=< [4] >];\n"
             "}\n";

    BOOST_TEST(ss.str() == expected.str());
}

BOOST_AUTO_TEST_CASE(DebugTestOnCpuRef)
{
    armnn::Network net;

    armnn::ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;

    // Defines layers.
    auto input = net.AddInputLayer(0, "InputLayer");
    auto activation = net.AddActivationLayer(activation1Descriptor, "ActivationLayer");
    auto output = net.AddOutputLayer(0, "OutputLayer");

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

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(net, backends, runtime->GetDeviceSpec(),
                                                               optimizerOptions);

    const armnn::Graph& graph = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetGraph();
    // Tests that all layers are present in the graph.
    BOOST_TEST(graph.GetNumLayers() == 5);

    // Tests that the vertices exist and have correct names.
    BOOST_TEST(GraphHasNamedLayer(graph, "InputLayer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "DebugLayerAfterInputLayer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "ActivationLayer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "DebugLayerAfterActivationLayer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "OutputLayer"));
}

BOOST_AUTO_TEST_SUITE_END()
