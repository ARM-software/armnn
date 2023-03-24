//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CommonTestUtils.hpp>

#include <Graph.hpp>
#include <Network.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("OptimizedNetwork")
{
TEST_CASE("SerializeToDot")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    //Defines layers.
    auto input = net->AddInputLayer(0);
    auto add = net->AddElementwiseBinaryLayer(armnn::BinaryOperation::Add);
    auto output = net->AddOutputLayer(0);

    // Connects layers.
    input->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorShape shape({4});
    armnn::TensorInfo info(shape, armnn::DataType::Float32);
    input->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    std::ostringstream ss;
    optimizedNet->SerializeToDot(ss);

    auto inputId = input->GetGuid();
    auto addId = add->GetGuid();
    auto outputId = output->GetGuid();

    std::stringstream expected;
    expected <<
        "digraph Optimized {\n"
        "    node [shape=\"record\"];\n"
        "    edge [fontsize=8 fontcolor=\"blue\" fontname=\"arial-bold\"];\n"
        "    " << inputId << " [label=\"{Input|Guid : " << inputId << "\\lLayerType : Input\\l"
                             "BackendID : CpuRef\\l}\"];\n"
        "    " << addId << " [label=\"{ElementwiseBinary|Guid : " << addId << "\\lLayerType : ElementwiseBinary\\l"
                           "BackendID : CpuRef\\l}\"];\n"
        "    " << outputId << " [label=\"{Output|Guid : " << outputId << "\\lLayerType : Output\\l"
                              "BackendID : CpuRef\\l}\"];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << inputId << " -> " << addId << " [label=< [4] >];\n"
        "    " << addId << " -> " << outputId << " [label=< [4] >];\n"
        "}\n";

    CHECK(ss.str() == expected.str());
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
        Optimize(*net, backends, runtime->GetDeviceSpec(), armnn::OptimizerOptionsOpaque(), errMessages);
        FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException&)
    {
        // Different exceptions are thrown on different backends
    }
    CHECK(errMessages.size() > 0);
}

TEST_CASE("OptimizeValidateDeviceNonSupportLayerWithFallback")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but it allows to fallback to CpuRef.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc, armnn::Compute::CpuRef };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    REQUIRE(optNet);

    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();

    for (auto&& layer : graph)
    {
        // If NEON is enabled, Input and Output layers are supported by CpuAcc,
        // the other layers are supported by CpuRef.
        // If NEON is not enabled, all layers are supported by CpuRef.
#if defined(ARMCOMPUTENEON_ENABLED)
        if (layer->GetType() == armnn::LayerType::Output)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#else
        CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
#endif
    }
}

TEST_CASE("OptimizeValidateWorkloadsUndefinedComputeDevice")
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
    layer = net->AddElementwiseBinaryLayer(armnn::BinaryOperation::Mul, "ml");

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

    std::vector<armnn::BackendId> backends = { armnn::Compute::Undefined };
    std::vector<std::string> errMessages;

    try
    {
        Optimize(*net, backends, runtime->GetDeviceSpec(),
                 armnn::OptimizerOptionsOpaque(), errMessages);
        FAIL("Should have thrown an exception.");
    }
    catch (const armnn::InvalidArgumentException&)
    {
        // Different exceptions are thrown on different backends
    }
    CHECK(errMessages.size() > 0);
}

TEST_CASE("OptimizeValidateWorkloadsUndefinedComputeDeviceWithFallback")
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
    layer = net->AddElementwiseBinaryLayer(armnn::BinaryOperation::Mul, "ml");

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

    std::vector<armnn::BackendId> backends = { armnn::Compute::Undefined, armnn::Compute::CpuRef };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();

    // validate workloads
    armnn::RefWorkloadFactory fact;
    for (auto&& layer : graph)
    {
        CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        CHECK_NOTHROW(
            layer->CreateWorkload(fact));
    }
}

TEST_CASE("OptimizeValidateWorkloadsDuplicateComputeDeviceWithFallback")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but it allows to fallback to CpuRef.
    armnn::NormalizationDescriptor descriptor;
    armnn::IConnectableLayer* normalize = net->AddNormalizationLayer(descriptor);

    armnn::IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(normalize->GetInputSlot(0));
    normalize->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    normalize->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc,
                                             armnn::Compute::GpuAcc,
                                             armnn::Compute::CpuRef };

    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    REQUIRE(optNet);

    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    graph.AllocateDynamicBuffers();

    for (auto&& layer : graph)
    {
        // If NEON is enabled, Input and Output layers are supported by CpuAcc,
        // the other layers are supported by CpuRef.
        // If only CL is enabled, Input and Output layers are supported by GpuAcc,
        // the other layers are supported by CpuRef.
        // If neither NEON, nor CL is enabled, all layers are supported by CpuRef.
#if defined(ARMCOMPUTENEON_ENABLED)
        if (layer->GetType() == armnn::LayerType::Input)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
        else if (layer->GetType() == armnn::LayerType::Output)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#elif defined(ARMCOMPUTECL_ENABLED)
        if (layer->GetType() == armnn::LayerType::Input)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
        else if (layer->GetType() == armnn::LayerType::Output)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::GpuAcc);
        }
        else if (layer->GetType() == armnn::LayerType::Normalization)
        {
            CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
        }
#else
        CHECK(layer->GetBackendId() == armnn::Compute::CpuRef);
#endif
    }
}

TEST_CASE("OptimizeNetworkCopy")
{
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    std::vector<armnn::NetworkId> networkIds;

    const std::string layerName("convolution2d");
    const armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 2, 2, 1 }, armnn::DataType::Float32);

    const armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    armnn::ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    armnn::ConstTensor biases(biasesInfo, biasesData);

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_DilationX   = 2;
    descriptor.m_DilationY   = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = armnn::DataLayout::NHWC;

    armnn::INetworkPtr network = armnn::INetwork::Create();
    armnn::IConnectableLayer* const inputLayer  = network->AddInputLayer(0);

    armnn::IConnectableLayer* const convLayer   = network->AddConvolution2dLayer(descriptor, layerName.c_str());
    armnn::IConnectableLayer* const outputLayer = network->AddOutputLayer(0);
    armnn::IConnectableLayer* weightsLayer = network->AddConstantLayer(weights);
    armnn::IConnectableLayer* biasLayer = network->AddConstantLayer(biases);

    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1u));

    biasLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2u));

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::vector<armnn::BackendId> preferredBackends { "CpuRef" };
    armnn::ModelOptions modelOptions;
    armnn::OptimizerOptionsOpaque optimizerOptions(false, false, false,
                                                   false, modelOptions, false);
    std::vector<std::string> errorMessages;

    // optimize the network.
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network,
                                                  preferredBackends,
                                                  runtime->GetDeviceSpec(),
                                                  optimizerOptions,
                                                  armnn::Optional<std::vector<std::string>&>(errorMessages));

    for (unsigned int i = 0; i < 2; ++i)
    {
        armnn::ModelOptions optimizedModelOptions;
        auto copy = armnn::IOptimizedNetworkPtr(new armnn::IOptimizedNetwork(*optNet.get(), optimizedModelOptions),
                                               &armnn::IOptimizedNetwork::Destroy);

        CHECK(copy);

        armnn::NetworkId netId;
        std::string errorMessage;

        CHECK(armnn::Status::Success == runtime->LoadNetwork(netId, std::move(copy), errorMessage));

        // Record the networkID for the loaded network
        networkIds.emplace_back(netId);
    }
    armnn::NetworkId optNetId;
    std::string errorMessage;

    // Load the original optNet
    CHECK(armnn::Status::Success == runtime->LoadNetwork(optNetId, std::move(optNet), errorMessage));

    std::vector<float> inputData = GenerateRandomData<float>(runtime->GetInputTensorInfo(optNetId, 0).GetNumElements());
    std::vector<float> outputData(runtime->GetOutputTensorInfo(optNetId, 0).GetNumElements());

    armnn::TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(optNetId, 0);
    inputTensorInfo.SetConstant(true);
    armnn::InputTensors inputTensors
    {
        {
            0, armnn::ConstTensor(inputTensorInfo, inputData.data())
        }
    };
    armnn::OutputTensors outputTensors
    {
        {
            0, armnn::Tensor(runtime->GetOutputTensorInfo(optNetId, 0), outputData.data())
        }
    };
    runtime->EnqueueWorkload(optNetId, inputTensors, outputTensors);
    runtime->UnloadNetwork(optNetId);

    // Record the networkID for the loaded network
    for (unsigned int i = 0; i < networkIds.size(); ++i)
    {
        armnn::NetworkId netId = networkIds[i];
        std::vector<float> copyOutputData(runtime->GetOutputTensorInfo(netId, 0).GetNumElements());

        armnn::TensorInfo inputTensorInfo2 = runtime->GetInputTensorInfo(netId, 0);
        inputTensorInfo2.SetConstant(true);
        armnn::InputTensors copyInputTensors
        {
            {
                0, armnn::ConstTensor(inputTensorInfo2, inputData.data())
            }
        };
        armnn::OutputTensors copyOutputTensors
        {
            {
                0, armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), copyOutputData.data())
            }
        };
        runtime->EnqueueWorkload(netId, copyInputTensors, copyOutputTensors);
        runtime->UnloadNetwork(netId);

        // Check results are identical to "original" version
        for (unsigned int j = 0; j < outputData.size(); ++j)
        {
            CHECK(outputData[j] == copyOutputData[j]);
        }
    }
}

}
