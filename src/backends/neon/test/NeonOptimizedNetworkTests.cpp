//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <Graph.hpp>
#include <Network.hpp>

#include <neon/NeonWorkloadFactory.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonOptimizedNetwork)

BOOST_AUTO_TEST_CASE(OptimizeValidateCpuAccDeviceSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    BOOST_CHECK(optNet);
    // validate workloads
    armnn::NeonWorkloadFactory fact =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    for (auto&& layer : static_cast<armnn::OptimizedNetwork*>(optNet.get())->GetGraph())
    {
        BOOST_CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        BOOST_CHECK_NO_THROW(
            layer->CreateWorkload(fact));
    }
}

BOOST_AUTO_TEST_CASE(OptimizeValidateDeviceNonSupportLayerNoFallback)
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc and isn't allowed to fall back, so Optimize will return null.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    std::vector<std::string> errMessages;

    try
    {
        Optimize(*net, backends, runtime->GetDeviceSpec(), armnn::OptimizerOptions(), errMessages);
        BOOST_FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        // Different exceptions are thrown on different backends
    }
    BOOST_CHECK(errMessages.size() > 0);
}

BOOST_AUTO_TEST_CASE(FastMathEnabledTestOnCpuAcc)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    armnn::OptimizerOptions optimizerOptions;
    armnn::BackendOptions modelOptions("CpuAcc", {{"FastMathEnabled", true}});
    optimizerOptions.m_ModelOptions.push_back(modelOptions);

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
    *net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    BOOST_CHECK(optimizedNet);

    auto modelOptionsOut = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetModelOptions();

    BOOST_TEST(modelOptionsOut.size() == 1);
    BOOST_TEST(modelOptionsOut[0].GetOption(0).GetName() == "FastMathEnabled");
    BOOST_TEST(modelOptionsOut[0].GetOption(0).GetValue().AsBool() == true);
}

BOOST_AUTO_TEST_SUITE_END()