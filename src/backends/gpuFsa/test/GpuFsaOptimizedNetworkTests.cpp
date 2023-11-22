//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("GpuFsaOptimizedNetwork")
{

TEST_CASE("SingleConv2dSupportedOptimizedNetwork")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    TensorInfo inputInfo({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo biasesInfo({ 1 }, DataType::Float32, 0.0f, 0, true);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;
    desc.m_DataLayout = DataLayout::NHWC;

    auto inputLayer = network->AddInputLayer(0, "input");
    auto weightLayer = network->AddConstantLayer(ConstTensor(weightsInfo, nullptr), "weights");
    auto biasLayer = network->AddConstantLayer(ConstTensor(biasesInfo, nullptr), "bias");
    auto convLayer = network->AddConvolution2dLayer(desc, "conv2d");
    auto outputLayer = network->AddOutputLayer(1, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    weightLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
    weightLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));
    biasLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);

    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::vector<BackendId> backends = { "GpuFsa" };

    OptimizerOptionsOpaque optimizedOptions;
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("TwoConv2dSupportedOptimizedNetwork")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    TensorInfo inputInfo({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo intermediateInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 1, 1, 1 }, DataType::Float32);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo biasesInfo({ 1 }, DataType::Float32, 0.0f, 0, true);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;
    desc.m_DataLayout = DataLayout::NHWC;

    auto inputLayer = network->AddInputLayer(0, "input");

    auto weightLayer1 = network->AddConstantLayer(ConstTensor(weightsInfo, nullptr), "weights");
    auto biasLayer1 = network->AddConstantLayer(ConstTensor(biasesInfo, nullptr), "bias");
    auto convLayer1 = network->AddConvolution2dLayer(desc, "conv2d");

    auto weightLayer2 = network->AddConstantLayer(ConstTensor(weightsInfo, nullptr), "weights");
    auto biasLayer2 = network->AddConstantLayer(ConstTensor(biasesInfo, nullptr), "bias");
    auto convLayer2 = network->AddConvolution2dLayer(desc, "conv2d");

    auto outputLayer = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

    weightLayer1->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(1));
    weightLayer1->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    biasLayer1->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(2));
    biasLayer1->GetOutputSlot(0).SetTensorInfo(biasesInfo);

    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).SetTensorInfo(intermediateInfo);

    weightLayer2->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(1));
    weightLayer2->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    biasLayer2->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(2));
    biasLayer2->GetOutputSlot(0).SetTensorInfo(biasesInfo);

    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::vector<BackendId> backends = { "GpuFsa" };

    OptimizerOptionsOpaque optimizedOptions;
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<OutputLayer>));
}

}