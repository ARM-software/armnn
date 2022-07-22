//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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

TEST_CASE("FoldPadLayerIntoConvolution2dLayer")
{
    Graph              graph;
    const unsigned int inputShape[]   = {1, 2, 2, 3};
    const unsigned int paddedShape[]  = {1, 6, 6, 3};
    const unsigned int weightsShape[] = {1, 2, 3, 3};
    const unsigned int outputShape[]  = {1, 2, 1, 1};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo weightsInfo(4, weightsShape, DataType::Float32, 1.0f, 0, true);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {2, 2},
                                 {2, 2},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Convolution2dDescriptor convolution2dDescriptor;
    convolution2dDescriptor.m_BiasEnabled = false;
    convolution2dDescriptor.m_StrideX     = 1;
    convolution2dDescriptor.m_StrideY     = 1;
    convolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;

    std::vector<float> weightsVector(18);
    ConstTensor        weights(weightsInfo, weightsVector);

    ConstantLayer* weightsLayer = graph.AddLayer<ConstantLayer>("Weights");
    weightsLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(weights);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);

    Convolution2dLayer* conv2dLayer = graph.AddLayer<Convolution2dLayer>(convolution2dDescriptor, "conv2d");
    conv2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> conv2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(conv2dLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot().Connect(conv2dLayer->GetInputSlot(1));
    conv2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimpleConv2d = [](const Layer* const layer)->bool {
        const auto conv2dLayer       = static_cast<const Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<Convolution2dLayer>(layer) && (layer->GetNameStr() == "conv2d") &&
            (conv2dLayerParams.m_PadLeft == 0) && (conv2dLayerParams.m_PadRight == 0) &&
            (conv2dLayerParams.m_PadTop == 0) && (conv2dLayerParams.m_PadBottom == 0) &&
            (conv2dLayerParams.m_StrideX == 1) && (conv2dLayerParams.m_StrideY == 1) &&
            (conv2dLayerParams.m_BiasEnabled == false) && (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>,
                                                      &IsLayerOfType<ConstantLayer>,
                                                      &IsLayerOfType<PadLayer>,
                                                      checkSimpleConv2d,
                                                      &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(FoldPadIntoConvolution2d()));

    auto checkPadFoldedIntoConv2d = [](const Layer* const layer)->bool {
        const auto conv2dLayer       = static_cast<const Convolution2dLayer*>(layer);
        const auto conv2dLayerParams = conv2dLayer->GetParameters();
        return IsLayerOfType<Convolution2dLayer>(layer) && (layer->GetNameStr() == "folded-pad-into-conv2d") &&
            (conv2dLayerParams.m_PadLeft == 2) && (conv2dLayerParams.m_PadRight == 2) &&
            (conv2dLayerParams.m_PadTop == 2) && (conv2dLayerParams.m_PadBottom == 2) &&
            (conv2dLayerParams.m_StrideX == 1) && (conv2dLayerParams.m_StrideY == 1) &&
            (conv2dLayerParams.m_BiasEnabled == false) && (conv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>,
                                                      &IsLayerOfType<ConstantLayer>,
                                                      checkPadFoldedIntoConv2d,
                                                      &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoDepthwiseConvolution2dLayer")
{
    Graph              graph;
    const unsigned int inputShape[]   = {1, 2, 2, 3};
    const unsigned int paddedShape[]  = {1, 6, 6, 3};
    const unsigned int weightsShape[] = {1, 2, 3, 3};
    const unsigned int outputShape[]  = {1, 2, 1, 3};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo weightsInfo(4, weightsShape, DataType::Float32, 1.0f, 0, true);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {2, 2},
                                 {2, 2},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    DepthwiseConvolution2dDescriptor depthwiseConvolution2dDescriptor;
    depthwiseConvolution2dDescriptor.m_BiasEnabled = false;
    depthwiseConvolution2dDescriptor.m_StrideX     = 1;
    depthwiseConvolution2dDescriptor.m_StrideY     = 1;
    depthwiseConvolution2dDescriptor.m_DataLayout  = DataLayout::NHWC;

    std::vector<float> weightsVector(18);
    ConstTensor        weights(weightsInfo, weightsVector);

    auto* weightsLayer = graph.AddLayer<ConstantLayer>("weights");
    weightsLayer->GetOutputSlot().SetTensorInfo(weightsInfo);
    weightsLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(weights);

    auto* depthwiseConv2dLayer = graph.AddLayer<DepthwiseConvolution2dLayer>(depthwiseConvolution2dDescriptor,
                                                                             "depthwiseConv2d");
    depthwiseConv2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> depthwiseConv2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(depthwiseConv2dLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot().Connect(depthwiseConv2dLayer->GetInputSlot(1));
    depthwiseConv2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimpleDepthwiseConv2d = [](const Layer* const layer)->bool {
        const auto depthwiseConv2dLayer       = static_cast<const DepthwiseConvolution2dLayer*>(layer);
        const auto depthwiseConv2dLayerParams = depthwiseConv2dLayer->GetParameters();
        return IsLayerOfType<DepthwiseConvolution2dLayer>(layer) && (layer->GetNameStr() == "depthwiseConv2d") &&
            (depthwiseConv2dLayerParams.m_PadLeft == 0) && (depthwiseConv2dLayerParams.m_PadRight == 0) &&
            (depthwiseConv2dLayerParams.m_PadTop == 0) && (depthwiseConv2dLayerParams.m_PadBottom == 0) &&
            (depthwiseConv2dLayerParams.m_StrideX == 1) && (depthwiseConv2dLayerParams.m_StrideY == 1) &&
            (depthwiseConv2dLayerParams.m_BiasEnabled == false) &&
            (depthwiseConv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>,
                                                      &IsLayerOfType<ConstantLayer>,
                                                      &IsLayerOfType<PadLayer>,
                                                      checkSimpleDepthwiseConv2d,
                                                      &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoDepthwiseConvolution2d()));

    auto checkPadFoldedIntoDepthwiseConv2d = [](const Layer* const layer)->bool {
        const auto depthwiseConv2dLayer       = static_cast<const DepthwiseConvolution2dLayer*>(layer);
        const auto depthwiseConv2dLayerParams = depthwiseConv2dLayer->GetParameters();
        return IsLayerOfType<DepthwiseConvolution2dLayer>(layer) &&
            (layer->GetNameStr() == "folded-pad-into-depthwiseConv2d") &&
            (depthwiseConv2dLayerParams.m_PadLeft == 2) && (depthwiseConv2dLayerParams.m_PadRight == 2) &&
            (depthwiseConv2dLayerParams.m_PadTop == 2) && (depthwiseConv2dLayerParams.m_PadBottom == 2) &&
            (depthwiseConv2dLayerParams.m_StrideX == 1) && (depthwiseConv2dLayerParams.m_StrideY == 1) &&
            (depthwiseConv2dLayerParams.m_BiasEnabled == false) &&
            (depthwiseConv2dLayerParams.m_DataLayout == DataLayout::NHWC);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<InputLayer>,
                                                      &IsLayerOfType<ConstantLayer>,
                                                      checkPadFoldedIntoDepthwiseConv2d,
                                                      &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoPooling2dLayer")
{
    Graph              graph;
    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const Layer* const layer) {
        const auto pool2dLayer = static_cast<const Pooling2dLayer*>(layer);
        return IsLayerOfType<Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
            (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoPooling2d()));

    auto checkPadFoldedIntoPool2d = [&](const Layer* const layer) {
        if (!IsLayerOfType<Pooling2dLayer>(layer) || (layer->GetNameStr() != "folded-pad-into-pool2d"))
        {
            return false;
        }

        const auto                pool2dLayer       = static_cast<const Pooling2dLayer*>(layer);
        const Pooling2dDescriptor pool2dLayerParams = pool2dLayer->GetParameters();

        Pooling2dDescriptor pool2dLayerParamsNoPad = pool2dLayerParams;
        pool2dLayerParamsNoPad.m_PadLeft       = 0;
        pool2dLayerParamsNoPad.m_PadRight      = 0;
        pool2dLayerParamsNoPad.m_PadTop        = 0;
        pool2dLayerParamsNoPad.m_PadBottom     = 0;
        // If we fold then PaddingMethod will be set to Ignore. The original will be Exclude.
        pool2dLayerParamsNoPad.m_PaddingMethod = PaddingMethod::Exclude;

        return (pool2dLayerParamsNoPad == pooling2dDescriptor) && (pool2dLayerParams.m_PadLeft == 1) &&
            (pool2dLayerParams.m_PadRight == 1) && (pool2dLayerParams.m_PadTop == 1) &&
            (pool2dLayerParams.m_PadBottom == 1) && (pool2dLayerParams.m_PaddingMethod == PaddingMethod::IgnoreValue);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             checkPadFoldedIntoPool2d,
                             &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoPooling2d_PadWithMultipleOutputsShouldNotBeOptimized")
{
    // In this test case we'll setup a pad layer with two outputs. One goes to a polling layers and the other
    // goes to an output layer. FoldPadLayerIntoPooling2d should not optimize this graph as it uses the
    // OptimizeForExclusiveConnection method.
    Graph              graph;
    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Add the alternative branch from the pas layer to an output layer.
    Layer* secondOutput = graph.AddLayer<OutputLayer>(1, "dummy output");
    padLayer->GetOutputSlot().Connect(secondOutput->GetInputSlot(0));

    auto checkSimplePool2d = [&](const Layer* const layer) {
        const auto pool2dLayer = static_cast<const Pooling2dLayer*>(layer);
        return IsLayerOfType<Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
            (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    // Initial sequence.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>,
                             &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoPooling2d()));

    // The network should not change.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>,
                             &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoPooling2dLayer_PoolingLayerWithExcludePaddingShouldNotTakeMorePadding")
{
    // In this test setup input, Pad layer, Pooling layer that includes padding, output layer. The optimization
    // should not work as the pooling layer already includes and existing pad and specifies PaddingMethod::Exclude.
    Graph              graph;
    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType      = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth     = 3;
    pooling2dDescriptor.m_PoolHeight    = 3;
    pooling2dDescriptor.m_StrideX       = 1;
    pooling2dDescriptor.m_StrideY       = 1;
    pooling2dDescriptor.m_DataLayout    = DataLayout::NHWC;
    // Include a pad with the pooling layer. This should prevent the optimization working.
    pooling2dDescriptor.m_PadLeft       = 1;
    pooling2dDescriptor.m_PadRight      = 1;
    pooling2dDescriptor.m_PadTop        = 1;
    pooling2dDescriptor.m_PadBottom     = 1;
    pooling2dDescriptor.m_PaddingMethod = PaddingMethod::Exclude;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const Layer* const layer) {
        const auto pool2dLayer = static_cast<const Pooling2dLayer*>(layer);
        return IsLayerOfType<Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
            (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoPooling2d()));

    // The optimization should not have modified the graph.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoPooling2dLayer_MaxPoolingLayerWithLargePadValueShouldNotBeFolded")
{
    // In this test setup input, Pad layer with a large pad value, Max Pooling layer, output layer. The optimization
    // should not work as the pad value will modify the result of the max pooling layer.
    Graph              graph;
    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::Float32);
    TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
    TensorInfo outputInfo(4, outputShape, DataType::Float32);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});
    // For Max pooling of a float a pad value of 0 is more than enough to stop the fold happening.
    // Set this to -std::numeric_limits<float>::infinity() to make the fold happen.
    padDescriptor.m_PadValue = 0;

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Max;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const Layer* const layer) {
        const auto pool2dLayer = static_cast<const Pooling2dLayer*>(layer);
        return IsLayerOfType<Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
            (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoPooling2d()));

    // The optimization should not have modified the graph.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<PadLayer>,
                             checkSimplePool2d,
                             &IsLayerOfType<OutputLayer>));
}

TEST_CASE("FoldPadLayerIntoPooling2dLayer_QuantizedAveragePoolingShouldNotBeFolded")
{
    Graph              graph;
    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::QAsymmU8);
    TensorInfo paddedInfo(4, paddedShape, DataType::QAsymmU8);
    TensorInfo outputInfo(4, outputShape, DataType::QAsymmU8);

    Layer* input = graph.AddLayer<InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});

    PadLayer* padLayer = graph.AddLayer<PadLayer>(padDescriptor, "pad");
    padLayer->GetOutputSlot().SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    Pooling2dLayer* pool2dLayer = graph.AddLayer<Pooling2dLayer>(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    Layer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot().Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot().Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkSimplePool2d = [&](const Layer* const layer) {
        const auto pool2dLayer = static_cast<const Pooling2dLayer*>(layer);
        return IsLayerOfType<Pooling2dLayer>(layer) && (layer->GetNameStr() == "pool2d") &&
            (pool2dLayer->GetParameters() == pooling2dDescriptor);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PadLayer>,
                        checkSimplePool2d,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FoldPadIntoPooling2d()));

    // The optimization should not have modified the graph.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PadLayer>,
                        checkSimplePool2d,
                        &IsLayerOfType<OutputLayer>));
}

#if defined(ARMNNREF_ENABLED)
TEST_CASE("FoldPadLayerIntoPooling2dLayer_ExecuteInferenceWithAndWithoutOptimization")
{
    // The idea of this test to run a simple pad+pool2d network twice. Once
    // with FoldPadLayerIntoPooling2dLayer enabled and a second time with it
    // avoided. The output tensors of each should match.
    const unsigned int inputShape[]  = {1, 4, 4, 2};
    const unsigned int paddedShape[] = {1, 6, 6, 2};
    const unsigned int outputShape[] = {1, 4, 4, 2};
    std::vector<float> inputData({2.0f, 2.0f, 6.0f, 6.0f,
                                  4.0f, 4.0f, 8.0f, 8.0f,
                                  10.0f, 12.0f, 14.0f, 16.0f,
                                  10.0f, 12.0f, 16.0f, 14.0f,

                                  18.0f, 20.0f, 24.0f, 22.0f,
                                  20.0f, 18.0f, 22.0f, 24.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,
                                 });
    try
    {
        // Create a network of input, pad, pooling 2D, output.
        INetworkPtr network = INetwork::Create();

        IConnectableLayer* inputLayer = network->AddInputLayer(0);
        TensorInfo inputInfo(4, inputShape, DataType::Float32);
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        PadDescriptor padDescriptor({{0, 0},
                                     {1, 1},
                                     {1, 1},
                                     {0, 0}});
        IConnectableLayer* padLayer = network->AddPadLayer(padDescriptor, "Pad");
        TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
        padLayer->GetOutputSlot(0).SetTensorInfo(paddedInfo);

        Pooling2dDescriptor pooling2dDescriptor;
        pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
        pooling2dDescriptor.m_PoolWidth  = 3;
        pooling2dDescriptor.m_PoolHeight = 3;
        pooling2dDescriptor.m_StrideX    = 1;
        pooling2dDescriptor.m_StrideY    = 1;
        pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;
        IConnectableLayer* pool2dLayer = network->AddPooling2dLayer(pooling2dDescriptor, "Pool2D");
        TensorInfo outputInfo(4, outputShape, DataType::Float32);
        pool2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        IConnectableLayer* outputLayer = network->AddOutputLayer(0);

        // Connect layers
        inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
        padLayer->GetOutputSlot(0).Connect(pool2dLayer->GetInputSlot(0));
        pool2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        // Create ArmNN runtime
        IRuntimePtr          run              = IRuntime::Create(IRuntime::CreationOptions());    // default options
        // Optimise the network
        IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        // Load network into runtime
        NetworkId            networkIdentifier;
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);

        TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
        inputTensorInfo.SetConstant(true);
        InputTensors inputTensors{{0, ConstTensor(inputTensorInfo, inputData.data())}};

        // Set the initial values of the data to different values to the golden data just in case the inference fails.
        std::vector<float> optimizedData(32, -std::numeric_limits<float>::infinity());
        OutputTensors      outputTensors{{0, Tensor(outputInfo, optimizedData.data())}};
        // Execute network
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        // Unload it.
        run->UnloadNetwork(networkIdentifier);

        // In this second case the pad will have two outputs, one connected to the pooling layer the second connected to
        // a second output layer. This will prevent the FoldPadLayerIntoPooling2dLayer optimization from working.
        // A previous test, FoldPadLayerIntoPooling2d_PadWithMultipleOutputsShouldNotBeOptimized, has proved that doing
        // this will avoid the optimization.
        IConnectableLayer* dummyOutputLayer = network->AddOutputLayer(1);
        padLayer->GetOutputSlot(0).Connect(dummyOutputLayer->GetInputSlot(0));

        // Optimize and load and execute it a second time.
        optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);
        std::vector<float> goldenData(32, 0.0f);
        std::vector<float> padOutputData(72, 0.0f);
        OutputTensors      goldenTensors{{0, Tensor(outputInfo, goldenData.data())},
                                         {1, Tensor(paddedInfo, padOutputData.data())}};
        run->EnqueueWorkload(networkIdentifier, inputTensors, goldenTensors);

        // Now we can compare goldenData against optimizedData. They should be the same.
        CHECK(std::equal(goldenData.begin(), goldenData.end(), optimizedData.begin()));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        ARMNN_ASSERT_MSG(false, e.what());
    }
}

TEST_CASE("FoldPadLayerIntoConv2dLayer_ExecuteInferenceWithAndWithoutOptimization")
{
    // The idea of this test to run a simple pad+conv2d network twice. Once
    // with FoldPadLayerIntoConv2dLayer enabled and a second time with it
    // avoided. The output tensors of each should match.
    const unsigned int inputShape[]   = {1, 4, 4, 3}; // NHWCin
    const unsigned int paddedShape[]  = {1, 6, 6, 3};
    const unsigned int weightsShape[] = {4, 2, 2, 3}; // CoutHWCin
    const unsigned int outputShape[]  = {1, 5, 5, 4}; // NHWCout

    std::vector<float> inputData({2.0f, 2.0f, 6.0f, 6.0f,
                                  4.0f, 4.0f, 8.0f, 8.0f,
                                  10.0f, 12.0f, 14.0f, 16.0f,
                                  10.0f, 12.0f, 16.0f, 14.0f,

                                  18.0f, 20.0f, 24.0f, 22.0f,
                                  20.0f, 18.0f, 22.0f, 24.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,

                                  2.0f, 2.0f, 6.0f, 6.0f,
                                  4.0f, 4.0f, 8.0f, 8.0f,
                                  10.0f, 12.0f, 14.0f, 16.0f,
                                  10.0f, 12.0f, 16.0f, 14.0f,
                                 });
    try
    {
        // Create a network of input, pad, pooling 2D, output.
        INetworkPtr network = INetwork::Create();

        IConnectableLayer* inputLayer = network->AddInputLayer(0);
        TensorInfo inputInfo(4, inputShape, DataType::Float32);
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        PadDescriptor padDescriptor({{0, 0},
                                     {1, 1},
                                     {1, 1},
                                     {0, 0}});
        IConnectableLayer* padLayer = network->AddPadLayer(padDescriptor, "Pad");
        TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
        padLayer->GetOutputSlot(0).SetTensorInfo(paddedInfo);

        Convolution2dDescriptor convDescriptor;
        convDescriptor.m_DataLayout  = DataLayout::NHWC;
        convDescriptor.m_StrideX     = 1;
        convDescriptor.m_StrideY     = 1;
        convDescriptor.m_BiasEnabled = true;

        std::vector<float>    weightsData  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
        TensorInfo            weightsInfo(4, weightsShape, DataType::Float32, 1.0f, 0, true);
        ConstTensor           weights(weightsInfo, weightsData);
        std::vector<float>    biasVector   = {5, 6, 7, 8};
        TensorInfo            biasInfo({4}, DataType::Float32, 1.0f, 0, true);
        ConstTensor           bias(biasInfo, biasVector);

        IConnectableLayer* conv2dLayer = network->AddConvolution2dLayer(convDescriptor, "Conv2D");

        TensorInfo outputInfo(4, outputShape, DataType::Float32);
        conv2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        IConnectableLayer* outputLayer = network->AddOutputLayer(0);

        // Connect layers
        inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
        padLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(0));
        conv2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        auto weightsLayer = network->AddConstantLayer(weights, "Weights");
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
        weightsLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(1));

        auto biasLayer = network->AddConstantLayer(bias, "Bias");
        biasLayer->GetOutputSlot(0).SetTensorInfo(bias.GetInfo());
        biasLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(2));

        // Create ArmNN runtime
        IRuntimePtr          run              = IRuntime::Create(IRuntime::CreationOptions());    // default options
        // Optimise the network
        IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        // Load network into runtime
        NetworkId            networkIdentifier;
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);

        TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
        inputTensorInfo.SetConstant(true);
        InputTensors inputTensors{{0, ConstTensor(inputTensorInfo, inputData.data())}};

        // Set the initial values of the data to different values to the golden data just in case the inference fails.
        std::vector<float> optimizedData(100, -std::numeric_limits<float>::infinity());
        OutputTensors      outputTensors{{0, Tensor(outputInfo, optimizedData.data())}};
        // Execute network
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        // Unload it.
        run->UnloadNetwork(networkIdentifier);

        // In this second case the pad will have two outputs, one connected to the conv layer the second connected to
        // a second output layer. This will prevent the FoldPadLayerIntoConv2dLayer optimization from working.
        // A previous test, FoldPadLayerIntoConv2d_PadWithMultipleOutputsShouldNotBeOptimized, has proved that doing
        // this will avoid the optimization.
        IConnectableLayer* dummyOutputLayer = network->AddOutputLayer(1);
        padLayer->GetOutputSlot(0).Connect(dummyOutputLayer->GetInputSlot(0));

        // Optimize and load and execute it a second time.
        optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);
        std::vector<float> goldenData(100, 0.0f);
        std::vector<float> padOutputData(108, 0.0f);
        OutputTensors      goldenTensors{{0, Tensor(outputInfo, goldenData.data())},
                                         {1, Tensor(paddedInfo, padOutputData.data())}};
        run->EnqueueWorkload(networkIdentifier, inputTensors, goldenTensors);

        // Now we can compare goldenData against optimizedData. They should be the same.
        CHECK(std::equal(goldenData.begin(), goldenData.end(), optimizedData.begin()));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        ARMNN_ASSERT_MSG(false, e.what());
    }
}

TEST_CASE("FoldPadLayerIntoDepthwiseConv2dLayer_ExecuteInferenceWithAndWithoutOptimization")
{
    // The idea of this test to run a simple pad+depthwiseconv2d network twice. Once
    // with FoldPadLayerIntoDeptwiseConv2dLayer enabled and a second time with it
    // avoided. The output tensors of each should match.
    const unsigned int inputShape[]   = {1, 4, 4, 3}; // NHWCin
    const unsigned int paddedShape[]  = {1, 6, 6, 3};
    const unsigned int weightsShape[] = {1, 2, 2, 12};  // 1HWCout
    const unsigned int outputShape[]  = {1, 5, 5, 12}; // NHWCout

    std::vector<float> inputData({2.0f, 2.0f, 6.0f, 6.0f,
                                  4.0f, 4.0f, 8.0f, 8.0f,
                                  10.0f, 12.0f, 14.0f, 16.0f,
                                  10.0f, 12.0f, 16.0f, 14.0f,

                                  18.0f, 20.0f, 24.0f, 22.0f,
                                  20.0f, 18.0f, 22.0f, 24.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,
                                  26.0f, 28.0f, 0.0f, 0.0f,

                                  2.0f, 2.0f, 6.0f, 6.0f,
                                  4.0f, 4.0f, 8.0f, 8.0f,
                                  10.0f, 12.0f, 14.0f, 16.0f,
                                  10.0f, 12.0f, 16.0f, 14.0f,
                                 });
    try
    {
        // Create a network of input, pad, pooling 2D, output.
        INetworkPtr network = INetwork::Create();

        IConnectableLayer* inputLayer = network->AddInputLayer(0);
        TensorInfo inputInfo(4, inputShape, DataType::Float32);
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);

        PadDescriptor padDescriptor({{0, 0},
                                     {1, 1},
                                     {1, 1},
                                     {0, 0}});
        IConnectableLayer* padLayer = network->AddPadLayer(padDescriptor, "Pad");
        TensorInfo paddedInfo(4, paddedShape, DataType::Float32);
        padLayer->GetOutputSlot(0).SetTensorInfo(paddedInfo);

        DepthwiseConvolution2dDescriptor convDescriptor;
        convDescriptor.m_DataLayout  = DataLayout::NHWC;
        convDescriptor.m_StrideX     = 1;
        convDescriptor.m_StrideY     = 1;
        convDescriptor.m_BiasEnabled = true;

        std::vector<float>    weightsData  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                              31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
        TensorInfo            weightsInfo(4, weightsShape, DataType::Float32, 0.0f, 0, true);
        ConstTensor           weights(weightsInfo, weightsData);
        std::vector<float>    biasVector   = {5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8};
        TensorInfo            biasInfo({12}, DataType::Float32, 0.0f, 0, true);
        ConstTensor           bias(biasInfo, biasVector);

        IConnectableLayer* conv2dLayer = network->AddDepthwiseConvolution2dLayer(convDescriptor,
                                                                                 "DepthwiseConv2D");

        TensorInfo outputInfo(4, outputShape, DataType::Float32);
        conv2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

        IConnectableLayer* outputLayer = network->AddOutputLayer(0);

        // Connect layers
        inputLayer->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
        padLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(0));
        conv2dLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

        auto weightsLayer = network->AddConstantLayer(weights, "Weights");
        weightsLayer->GetOutputSlot(0).SetTensorInfo(weights.GetInfo());
        weightsLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(1));

        auto biasLayer = network->AddConstantLayer(bias, "Bias");
        biasLayer->GetOutputSlot(0).SetTensorInfo(bias.GetInfo());
        biasLayer->GetOutputSlot(0).Connect(conv2dLayer->GetInputSlot(2));

        // Create ArmNN runtime
        IRuntimePtr          run              = IRuntime::Create(IRuntime::CreationOptions());    // default options
        // Optimise the network
        IOptimizedNetworkPtr optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        // Load network into runtime
        NetworkId            networkIdentifier;
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);

        TensorInfo inputTensorInfo = run->GetInputTensorInfo(networkIdentifier, 0);
        inputTensorInfo.SetConstant(true);
        InputTensors inputTensors{{0, ConstTensor(inputTensorInfo, inputData.data())}};

        // Set the initial values of the data to different values to the golden data just in case the inference fails.
        std::vector<float> optimizedData(300, -std::numeric_limits<float>::infinity());
        OutputTensors      outputTensors{{0, Tensor(outputInfo, optimizedData.data())}};
        // Execute network
        run->EnqueueWorkload(networkIdentifier, inputTensors, outputTensors);
        // Unload it.
        run->UnloadNetwork(networkIdentifier);

        // In this second case the pad will have two outputs, one connected to the conv layer the second connected to
        // a second output layer. This will prevent the FoldPadLayerIntoDepthwiseConv2dLayer optimization from working.
        // A previous test, FoldPadLayerIntoDepthwiseConv2d_PadWithMultipleOutputsShouldNotBeOptimized, has proved that
        // doing this will avoid the optimization.
        IConnectableLayer* dummyOutputLayer = network->AddOutputLayer(1);
        padLayer->GetOutputSlot(0).Connect(dummyOutputLayer->GetInputSlot(0));

        // Optimize and load and execute it a second time.
        optimizedNetwork = Optimize(*network, {Compute::CpuRef}, run->GetDeviceSpec());
        CHECK(run->LoadNetwork(networkIdentifier, std::move(optimizedNetwork)) == Status::Success);
        std::vector<float> goldenData(300, 0.0f);
        std::vector<float> padOutputData(108, 0.0f);
        OutputTensors      goldenTensors{{0, Tensor(outputInfo, goldenData.data())},
                                         {1, Tensor(paddedInfo, padOutputData.data())}};
        run->EnqueueWorkload(networkIdentifier, inputTensors, goldenTensors);

        // Now we can compare goldenData against optimizedData. They should be the same.
        CHECK(std::equal(goldenData.begin(), goldenData.end(), optimizedData.begin()));
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        ARMNN_ASSERT_MSG(false, e.what());
    }
}
#endif

}