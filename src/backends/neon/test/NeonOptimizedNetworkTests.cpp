//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <Graph.hpp>
#include <Network.hpp>

#include <neon/NeonWorkloadFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("NeonOptimizedNetwork")
{
TEST_CASE("OptimizeValidateCpuAccDeviceSupportLayerNoFallback")
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
    CHECK(optNet);
    // validate workloads
    armnn::NeonWorkloadFactory fact =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());

    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    for (auto&& layer : graph)
    {
        CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        CHECK_NOTHROW(
            layer->CreateWorkload(fact));
    }
}

TEST_CASE("OptimizeValidateDeviceNonSupportLayerNoFallback")
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
        FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        // Different exceptions are thrown on different backends
    }
    CHECK(errMessages.size() > 0);
}

TEST_CASE("FastMathEnabledTestOnCpuAcc")
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

    CHECK(optimizedNet);

    auto modelOptionsOut = GetModelOptionsForTesting(optimizedNet.get());

    CHECK(modelOptionsOut.size() == 1);
    CHECK(modelOptionsOut[0].GetOption(0).GetName() == "FastMathEnabled");
    CHECK(modelOptionsOut[0].GetOption(0).GetValue().AsBool() == true);
}

TEST_CASE("NumberOfThreadsTestOnCpuAcc")
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input  = net->AddInputLayer(0);
    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    unsigned int numberOfThreads = 2;

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    armnn::OptimizerOptions optimizerOptions;
    armnn::BackendOptions modelOptions("CpuAcc", {{"NumberOfThreads", numberOfThreads}});
    optimizerOptions.m_ModelOptions.push_back(modelOptions);

    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(
            *net, backends, runtime->GetDeviceSpec(), optimizerOptions);

    CHECK(optimizedNet);
    std::unique_ptr<armnn::Graph> graphPtr;
    armnn::OptimizedNetworkImpl impl(std::move(graphPtr), optimizerOptions.m_ModelOptions);

    auto modelOptionsOut = impl.GetModelOptions();

    CHECK(modelOptionsOut.size() == 1);
    CHECK(modelOptionsOut[0].GetOption(0).GetName() == "NumberOfThreads");
    CHECK(modelOptionsOut[0].GetOption(0).GetValue().AsUnsignedInt() == numberOfThreads);
}

}