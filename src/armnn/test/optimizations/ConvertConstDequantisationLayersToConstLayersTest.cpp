//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
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
}
