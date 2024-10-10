//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"
#include <Network.hpp>
#include <TestUtils.hpp>
#include <doctest/doctest.h>
#include <armnn/backends/TensorHandle.hpp>
#include <Optimizer.hpp>

TEST_SUITE("Optimizer")
{
using namespace armnn;
using namespace armnn::optimizations;

// Helpers for testing
auto checkConstantFloat32 = [](const armnn::Layer *const layer)
{
    return IsLayerOfType<ConstantLayer>(layer) && (layer->GetDataType() == DataType::Float32);
};

auto checkConstantFloat16 = [](const armnn::Layer *const layer)
{
    return IsLayerOfType<ConstantLayer>(layer) && (layer->GetDataType() == DataType::Float16);
};

auto checkConstantQAsymmS8 = [](const armnn::Layer *const layer)
{
    return IsLayerOfType<ConstantLayer>(layer) && (layer->GetDataType() == DataType::QAsymmS8);
};

auto checkPadFoldedIntoConv2d = [](const Layer* const layer)
{
    const auto conv2dLayer       = static_cast<const Convolution2dLayer*>(layer);
    const auto conv2dLayerParams = conv2dLayer->GetParameters();

    return IsLayerOfType<Convolution2dLayer>(layer) &&
           (layer->GetNameStr() == "folded-pad-into-conv2d") &&
           (conv2dLayerParams.m_PadLeft == 2) &&
           (conv2dLayerParams.m_PadRight == 2) &&
           (conv2dLayerParams.m_PadTop == 2) &&
           (conv2dLayerParams.m_PadBottom == 2) &&
           (conv2dLayerParams.m_StrideX == 1) &&
           (conv2dLayerParams.m_StrideY == 1) &&
           (conv2dLayerParams.m_BiasEnabled == false) &&
           (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
};

TEST_CASE("ConvertConstFloat16DequantizeToConstFloat32")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float16, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    auto constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    auto dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    auto output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> dequantize -> output
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(output->GetInputSlot(0));


    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat16,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(ConvertConstDequantisationLayersToConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat32,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("ConvertConstFloat16DequantizeToConstFloat32PlusFusePadWithConv2d")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float16, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    auto constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    auto dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    auto output = graph.AddLayer<OutputLayer>(0, "output");

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_StrideX     = 1;
    convolution2dDescriptor.m_StrideY     = 1;
    convolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;
    auto conv2d = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "conv2d");


    auto inputLayer = graph.AddLayer<InputLayer>(0, "input");

    PadDescriptor padDescriptor({{0, 0},
                                 {2, 2},
                                 {2, 2},
                                 {0, 0}});

    const unsigned int paddedShape[]  = {1, 6, 6, 3};

    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);

    auto padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    // Connect up:
    //           input -> pad -> conv2d -> output
    // constant -> dequantize ->
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(conv2d->GetInputSlot(1));
    inputLayer->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat16,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<PadLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(ConvertConstDequantisationLayersToConstLayers()));
    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoConvolution2d()));

    // Ensure that the const and dequantize are now constant of type fp32
    // Ensure pad and conv2d are now just convolution
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat32,
                        checkPadFoldedIntoConv2d,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("ConvertConstInt8DequantizeToConstFloat32")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::QAsymmS8, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    auto constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<int8_t> constantValues(constTensorInfo.GetNumElements(), 5);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    auto dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    auto output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> dequantize -> output
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantQAsymmS8,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(ConvertConstDequantisationLayersToConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat32,
                        &IsLayerOfType<OutputLayer>));
}

// Convert TfLite Turbo Model Tests
TEST_CASE("TurboConvertConstFloat16DequantizeToConstFloat16V2")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float16, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    auto constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    auto dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    auto output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> dequantize -> output
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(output->GetInputSlot(0));


    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat16,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(TurboConvertConstDequantisationLayersToConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat16,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("TurboConvertConstFloat16DequantizeToConstFloat16PlusFusePadWithConv2dV2")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float16, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    auto constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    auto dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    auto output = graph.AddLayer<OutputLayer>(0, "output");

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_StrideX     = 1;
    convolution2dDescriptor.m_StrideY     = 1;
    convolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;
    auto conv2d = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "conv2d");


    auto inputLayer = graph.AddLayer<InputLayer>(0, "input");

    PadDescriptor padDescriptor({{0, 0},
                                 {2, 2},
                                 {2, 2},
                                 {0, 0}});

    const unsigned int paddedShape[]  = {1, 6, 6, 3};

    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);

    auto padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    // Connect up:
    //           input -> pad -> conv2d -> output
    // constant -> dequantize ->
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(conv2d->GetInputSlot(1));
    inputLayer->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(conv2d->GetInputSlot(0));
    conv2d->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat16,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<PadLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(TurboConvertConstDequantisationLayersToConstLayers()));
    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoConvolution2d()));

    // Ensure that the const and dequantize are now constant of type fp16
    // Ensure pad and conv2d are now just convolution
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat16,
                        checkPadFoldedIntoConv2d,
                        &IsLayerOfType<OutputLayer>));
}

#if defined(ARMNNREF_ENABLED)
TEST_CASE("TurboModelRecognitionAndOptimization")
{
    // This test creates a basic "TfLite Turbo Model" which we define as an FP32 Model with FP16 weights dequantized
    // into FP32 weights.
    // The test then runs the basic turbo model through our Optimize function to check that we can recognise the model
    // as a turbo model and then apply the correct optimizations:
    // ConvertFp32NetworkToFP16 and TurboCOnvertConstDequantisationLayersToConstLayers
    // We know the test is succesful by checking what layers are present after Optimization as well as checking the
    // data type of the constant weights layer.

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input = network->AddInputLayer(0, "input");

    Convolution2dDescriptor descriptor;
    descriptor.m_DataLayout  = DataLayout::NHWC;

    TensorInfo inputInfo(  { 1, 5, 5, 1 }, DataType::Float32, 1, 0, true);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float16, 1.0, 0, true);
    TensorInfo outputInfo( { 1, 3, 3, 1 }, DataType::Float32, 1, 0);

    std::vector<float> weightsData = { 4, 5, 6, 0, 0, 0, 3, 2, 1 };

    std::vector<Half> qWeightsData = armnnUtils::QuantizedVector<Half>(weightsData, 1, 0);
    ConstTensor weights(weightsInfo, qWeightsData);

    IConnectableLayer* weightsLayer = network->AddConstantLayer(weights, "Weights");
    IConnectableLayer* convolution2d = network->AddConvolution2dLayer(descriptor, "convolution2d");
    IConnectableLayer* dequantizeLayer = network->AddDequantizeLayer("dequantize");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, convolution2d, inputInfo, 0, 0);
    Connect(weightsLayer, dequantizeLayer, weightsInfo, 0, 0);
    Connect(dequantizeLayer, convolution2d, outputInfo, 0, 1);
    Connect(convolution2d, output, outputInfo, 0, 0);

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuRef};

    IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), OptimizerOptionsOpaque());
    Graph& graph = GetGraphForTesting(optNet.get());

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat32,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<OutputLayer>));

    // Now we will set FastMath capability to true which is the remaining condition needed to allow us to test if the
    // Optimize function can recognise a basic TfLite turbo model and perform the correct optimizations:
    // Backend has Fastmath capability: true
    // Backend has FP16 support: true
    // TfLite Model is a Turbo Model

    armnn::BackendOptions cpuRef("CpuRef", { { "FastMathEnabled", true } });
    armnn::ModelOptions modelOptions;
    modelOptions.push_back(cpuRef);

    armnn::OptimizerOptionsOpaque optimizerOptions(false, false, false,
                                                   false, modelOptions, false);

    optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizerOptions);
    Graph& turboGraph = GetGraphForTesting(optNet.get());

    CHECK(CheckSequence(turboGraph.cbegin(), turboGraph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat16,
                        &IsLayerOfType<ConvertFp32ToFp16Layer>,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<ConvertFp16ToFp32Layer>,
                        &IsLayerOfType<OutputLayer>));
}
#endif
}
