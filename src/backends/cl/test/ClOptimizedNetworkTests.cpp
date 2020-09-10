//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClWorkloadFactoryHelper.hpp"

#include <Network.hpp>

#include <test/GraphUtils.hpp>

#include <cl/ClWorkloadFactory.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClOptimizedNetwork)

BOOST_AUTO_TEST_CASE(OptimizeValidateGpuDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);
    // validate workloads
    armnn::ClWorkloadFactory fact =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::GpuAcc);
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(fact));
    }
}

BOOST_AUTO_TEST_CASE(FP16TurboModeTestOnGpuAcc)
{
    // Test to check when Fp16 Turbo mode set
    // it converts the Fp32 network to Fp16 Network
    // add Fp32ToFp16 conversion layer after the InputLayer
    // add Fp16ToFp32 conversion layer after the OutputLayer
    // checks the other layers if they are supported in Fp16
    // if they are not put the conversion layers before and after
    // if they are not supported in Fp16 use Fp32 instead
    // if there are inverse conversion layers remove them with optimization
    // at the moment FloorLayer is not supported in Fp16 so it rolls back to Fp32
    // and inverse conversion layers are removed by the optimizer
    armnn::Network net;

    // Defines layers.
    auto input = net.AddInputLayer(0, "input layer");
    // ReLu1
    armnn::ActivationDescriptor activation1Descriptor;
    activation1Descriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activation1Descriptor.m_A = 1.f;
    activation1Descriptor.m_B = -1.f;
    auto activation = net.AddActivationLayer(activation1Descriptor, "activation layer");
    auto output = net.AddOutputLayer(0, "output layer");

    // Connects layers.
    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    activation->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};

    armnn::OptimizerOptions optimizerOptions;
    optimizerOptions.m_ReduceFp32ToFp16 = true;

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
            net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    const armnn::Graph& graph = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetGraph();

    // Tests that all layers are present in the graph.
    BOOST_TEST(graph.GetNumLayers() == 5);

    // Tests that the vertices exist and have correct names.
    BOOST_TEST(GraphHasNamedLayer(graph, "input layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "convert_fp32_to_fp16-0-input layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "activation layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "convert_fp16_to_fp32-0-output layer"));
    BOOST_TEST(GraphHasNamedLayer(graph, "output layer"));
}

BOOST_AUTO_TEST_CASE(FastMathEnabledTestOnGpuAcc)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    armnn::OptimizerOptions optimizerOptions;
    armnn::BackendOptions modelOptions("GpuAcc", {{"FastMathEnabled", true}});
    optimizerOptions.m_ModelOptions.push_back(modelOptions);

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
    *net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    BOOST_CHECK(optimizedNet);

    auto modelOptionsOut = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetModelOptions();

    BOOST_TEST(modelOptionsOut.size() == 1);
    BOOST_TEST(modelOptionsOut[0].GetOption(0).GetName() == "FastMathEnabled");
    BOOST_TEST(modelOptionsOut[0].GetOption(0).GetValue().AsBool() == true);
}

BOOST_AUTO_TEST_SUITE_END();
