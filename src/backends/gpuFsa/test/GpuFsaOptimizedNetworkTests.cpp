//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("GpuFsaOptimizedNetwork")
{

TEST_CASE("CastSupportedOptimizedNetwork")
{
    using namespace armnn;

    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const TensorShape& inputShape   = { 2, 2, 2 };
    const TensorShape& outputShape  = { 2, 2, 2 };

    TensorInfo inputTensorInfo(inputShape, DataType::Float32, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType::Float16, qScale, qOffset);

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input     = network->AddInputLayer(0, "input");
    IConnectableLayer* castLayer = network->AddCastLayer("cast");
    IConnectableLayer* output    = network->AddOutputLayer(1, "output");

    Connect(input, castLayer, inputTensorInfo, 0, 0);
    Connect(castLayer, output, outputTensorInfo, 0, 0);

    std::vector<BackendId> backends = { "GpuFsa" };

    OptimizerOptionsOpaque optimizedOptions;
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<OutputLayer>));
}

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

TEST_CASE("ElementwiseBinaryAddSupportedOptimizedNetwork")
{
    using namespace armnn;

    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const TensorShape& input1Shape  = { 2, 2, 2 };
    const TensorShape& input2Shape  = { 2, 2, 2 };
    const TensorShape& outputShape  = { 2, 2, 2 };

    TensorInfo input1TensorInfo(input1Shape, DataType::Float32, qScale, qOffset, true);
    TensorInfo input2TensorInfo(input2Shape, DataType::Float32, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType::Float32, qScale, qOffset);

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input1 = network->AddInputLayer(0, "input0");
    IConnectableLayer* input2 = network->AddInputLayer(1, "input1");

    ElementwiseBinaryDescriptor desc;
    desc.m_Operation = BinaryOperation::Add;

    IConnectableLayer* elementwiseBinaryLayer = network->AddElementwiseBinaryLayer(desc, "elementwiseBinary");
    IConnectableLayer* output = network->AddOutputLayer(2, "output");

    Connect(input1, elementwiseBinaryLayer, input1TensorInfo, 0, 0);
    Connect(input2, elementwiseBinaryLayer, input2TensorInfo, 0, 1);
    Connect(elementwiseBinaryLayer, output, outputTensorInfo, 0, 0);

    std::vector<BackendId> backends = { "GpuFsa" };

    OptimizerOptionsOpaque optimizedOptions;
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("ElementwiseBinarySubSupportedOptimizedNetwork")
{
    using namespace armnn;

    const float qScale = 1.0f;
    const int32_t qOffset = 0;

    const TensorShape& input1Shape  = { 2, 2, 2 };
    const TensorShape& input2Shape  = { 2, 2, 2 };
    const TensorShape& outputShape  = { 2, 2, 2 };

    TensorInfo input1TensorInfo(input1Shape, DataType::Float32, qScale, qOffset, true);
    TensorInfo input2TensorInfo(input2Shape, DataType::Float32, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType::Float32, qScale, qOffset);

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input1 = network->AddInputLayer(0, "input0");
    IConnectableLayer* input2 = network->AddInputLayer(1, "input1");

    ElementwiseBinaryDescriptor desc;
    desc.m_Operation = BinaryOperation::Sub;

    IConnectableLayer* elementwiseBinaryLayer = network->AddElementwiseBinaryLayer(desc, "elementwiseBinary");
    IConnectableLayer* output = network->AddOutputLayer(2, "output");

    Connect(input1, elementwiseBinaryLayer, input1TensorInfo, 0, 0);
    Connect(input2, elementwiseBinaryLayer, input2TensorInfo, 0, 1);
    Connect(elementwiseBinaryLayer, output, outputTensorInfo, 0, 0);

    std::vector<BackendId> backends = { "GpuFsa" };

    OptimizerOptionsOpaque optimizedOptions;
    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PreCompiledLayer>,
                        &IsLayerOfType<OutputLayer>));
}

}