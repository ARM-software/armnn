//
// Copyright © 2017, 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <CommonTestUtils.hpp>
#include "MockBackendId.hpp"

#include <Graph.hpp>
#include <Network.hpp>

#include <armnn/BackendRegistry.hpp>
#include <armnnTestUtils/MockBackend.hpp>

#include <doctest/doctest.h>
#include <unordered_map>

using namespace armnn;

namespace
{

// The expected number of layers, input and output slots in a subgraph after a test
struct ExpectedSubgraphSize
{
    size_t m_NumInputSlots  = 0;
    size_t m_NumOutputSlots = 0;
    size_t m_NumLayers      = 0;
};

// Keep the layers organized by layer name
using LayerNameToLayerMap = std::unordered_map<std::string, Layer*>;

// Used to convert input and output slots from reference type (as stored in graphs) to
// pointer type (as stored in subgraphs)
template <typename SlotType>
SlotType* ConvertReferenceTypeToPointerType(const SlotType& input)
{
    return const_cast<SlotType*>(&input);
}

// Used to convert input and output slots from reference type (as stored in graphs) to
// pointer type (as stored in subgraphs), array version
template <typename SlotType>
std::vector<SlotType*> ConvertReferenceTypeToPointerType(const std::vector<SlotType>& input)
{
    std::vector<SlotType*> output;
    std::transform(input.begin(),
                   input.end(),
                   std::back_inserter(output),
                   [](const SlotType& inputItem)
    {
        return ConvertReferenceTypeToPointerType(inputItem);
    });

    return output;
}

// Convert from vector of Slots* (Input/Output) to vector of ISlots* (IInput/IOutput)
template <typename SlotType, typename ResultSlotType>
std::vector<ResultSlotType*> ConvertSlotsToISlots(const std::vector<SlotType*> input)
{
    std::vector<ResultSlotType*> output;
    for (auto slot : input)
    {
        output.push_back(PolymorphicDowncast<ResultSlotType*>(slot));
    }
    return output;
}

// Convenience function to add an input layer to a graph
Layer* AddInputLayer(Graph& graph,
                     const std::string& layerName,
                     const TensorInfo& inputInfo,
                     LayerBindingId inputId = 0)
{
    Layer* const inputLayer = graph.AddLayer<InputLayer>(inputId, layerName.c_str());
    CHECK(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    return inputLayer;
}

// Convenience function to add an output layer to a graph
Layer* AddOutputLayer(Graph& graph,
                      const std::string& layerName)
{
    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, layerName.c_str());
    CHECK(outputLayer);
    return outputLayer;
}

// Convenience function to add a convolution layer to a graph
Convolution2dLayer* AddConvolutionLayer(Graph& graph,
                                        LayerNameToLayerMap& layersInGraph,
                                        const Convolution2dDescriptor& convolutionDescriptor,
                                        const std::string& layerName,
                                        const TensorInfo& outputInfo)
{
    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, layerName.c_str());
    CHECK(convLayer);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(convLayer->GetName(), convLayer));
    return convLayer;
}

// Convenience function to add a constant layer to a graph
ConstantLayer* AddConstantLayer(Graph& graph,
                                LayerNameToLayerMap& layersInGraph,
                                const std::string& layerName,
                                const ConstTensor& constTensor,
                                const TensorInfo& outputInfo)
{
    ConstantLayer* const constantLayer = graph.AddLayer<ConstantLayer>(layerName.c_str());
    CHECK(constantLayer);
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(constantLayer->GetName(), constantLayer));
    return constantLayer;
}

// Convenience function to add a pooling layer to a graph
Pooling2dLayer* AddPoolingLayer(Graph& graph,
                                LayerNameToLayerMap& layersInGraph,
                                const Pooling2dDescriptor& poolingDescriptor,
                                const std::string& layerName,
                                const TensorInfo& outputInfo)
{
    Pooling2dLayer* const poolingLayer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, layerName.c_str());
    CHECK(poolingLayer);
    poolingLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(poolingLayer->GetName(), poolingLayer));
    return poolingLayer;
}

// Convenience function to add an addition layer to a graph
AdditionLayer* AddAdditionaLayer(Graph& graph,
                                 LayerNameToLayerMap& layersInGraph,
                                 const std::string& layerName,
                                 const TensorInfo& outputInfo)
{
    AdditionLayer* const additionLayer = graph.AddLayer<AdditionLayer>(layerName.c_str());
    CHECK(additionLayer);
    additionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(additionLayer->GetName(), additionLayer));
    return additionLayer;
}

// Convenience function to check that the given substitution matches the specified expected values
void CheckSubstitution(const OptimizationViews::SubstitutionPair& substitution,
                       const ExpectedSubgraphSize& expectedSubstitutableSubgraphSize,
                       const ExpectedSubgraphSize& expectedReplacementSubgraphSize,
                       const SubgraphView::IInputSlots& expectedSubstitutableInputSlots,
                       const SubgraphView::IOutputSlots& expectedSubstitutableOutputSlots,
                       const SubgraphView::IConnectableLayers& expectedSubstitutableLayers)
{
    const SubgraphView& substitutableSubgraph = substitution.m_SubstitutableSubgraph;
    const SubgraphView::IInputSlots& substitutableSubgraphInputSlots = substitutableSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& substitutableSubgraphOutputSlots = substitutableSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& substitutableSubgraphLayers =
            substitutableSubgraph.GetIConnectableLayers();

    const SubgraphView& replacementSubgraph                          = substitution.m_ReplacementSubgraph;
    const SubgraphView::IInputSlots& replacementSubgraphInputSlots   = replacementSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& replacementSubgraphOutputSlots = replacementSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& replacementSubgraphLayers = replacementSubgraph.GetIConnectableLayers();

    CHECK(substitutableSubgraphInputSlots.size()  == expectedSubstitutableSubgraphSize.m_NumInputSlots);
    CHECK(substitutableSubgraphOutputSlots.size() == expectedSubstitutableSubgraphSize.m_NumOutputSlots);
    CHECK(substitutableSubgraphLayers.size()      == expectedSubstitutableSubgraphSize.m_NumLayers);

    CHECK(AreEqual(substitutableSubgraphInputSlots,  expectedSubstitutableInputSlots));
    CHECK(AreEqual(substitutableSubgraphOutputSlots, expectedSubstitutableOutputSlots));
    CHECK(AreEqual(substitutableSubgraphLayers,      expectedSubstitutableLayers));

    CHECK(replacementSubgraphInputSlots.size()  == expectedReplacementSubgraphSize.m_NumInputSlots);
    CHECK(replacementSubgraphOutputSlots.size() == expectedReplacementSubgraphSize.m_NumOutputSlots);
    CHECK(replacementSubgraphLayers.size()      == expectedReplacementSubgraphSize.m_NumLayers);

    CHECK(!AreEqual(replacementSubgraphInputSlots,  expectedSubstitutableInputSlots));
    CHECK(!AreEqual(replacementSubgraphOutputSlots, expectedSubstitutableOutputSlots));
    CHECK(!AreEqual(replacementSubgraphLayers,      expectedSubstitutableLayers));

    CHECK(std::all_of(replacementSubgraphLayers.begin(),
                           replacementSubgraphLayers.end(),
                           [](const IConnectableLayer* layer)
    {
        return layer->GetType() == LayerType::PreCompiled;
    }));
}

// Convenience function to check that the given failed subgraph matches the specified expected values
void CheckFailedSubgraph(const SubgraphView& failedSubgraph,
                         const ExpectedSubgraphSize& expectedFailedSubgraphSize,
                         const SubgraphView::IInputSlots& expectedFailedInputSlots,
                         const SubgraphView::IOutputSlots& expectedFailedOutputSlots,
                         const SubgraphView::IConnectableLayers& expectedFailedLayers)
{
    const SubgraphView::IInputSlots&  failedSubgraphInputSlots  = failedSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& failedSubgraphOutputSlots = failedSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& failedSubgraphLayers = failedSubgraph.GetIConnectableLayers();

    CHECK(failedSubgraphInputSlots.size()  == expectedFailedSubgraphSize.m_NumInputSlots);
    CHECK(failedSubgraphOutputSlots.size() == expectedFailedSubgraphSize.m_NumOutputSlots);
    CHECK(failedSubgraphLayers.size()      == expectedFailedSubgraphSize.m_NumLayers);

    CHECK(AreEqual(failedSubgraphInputSlots,  expectedFailedInputSlots));
    CHECK(AreEqual(failedSubgraphOutputSlots, expectedFailedOutputSlots));
    CHECK(AreEqual(failedSubgraphLayers,      expectedFailedLayers));
}

// Convenience function to check that the given untouched subgraph matches the specified expected values
void CheckUntouchedSubgraph(const SubgraphView& untouchedSubgraph,
                            const ExpectedSubgraphSize& expectedUntouchedSubgraphSize,
                            const SubgraphView::IInputSlots& expectedUntouchedInputSlots,
                            const SubgraphView::IOutputSlots& expectedUntouchedOutputSlots,
                            const SubgraphView::IConnectableLayers& expectedUntouchedLayers)
{
    const SubgraphView::IInputSlots& untouchedSubgraphInputSlots = untouchedSubgraph.GetIInputSlots();
    const SubgraphView::IOutputSlots& untouchedSubgraphOutputSlots = untouchedSubgraph.GetIOutputSlots();
    const SubgraphView::IConnectableLayers& untouchedSubgraphLayers = untouchedSubgraph.GetIConnectableLayers();

    CHECK(untouchedSubgraphInputSlots.size()  == expectedUntouchedSubgraphSize.m_NumInputSlots);
    CHECK(untouchedSubgraphOutputSlots.size() == expectedUntouchedSubgraphSize.m_NumOutputSlots);
    CHECK(untouchedSubgraphLayers.size()      == expectedUntouchedSubgraphSize.m_NumLayers);

    CHECK(AreEqual(untouchedSubgraphInputSlots,  expectedUntouchedInputSlots));
    CHECK(AreEqual(untouchedSubgraphOutputSlots, expectedUntouchedOutputSlots));
    CHECK(AreEqual(untouchedSubgraphLayers,      expectedUntouchedLayers));
}

// Creates a subgraph containing only a single unsupported layer (only convolutions are unsupported by the mock backend)
SubgraphView::SubgraphViewPtr BuildFullyUnsupportedSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    Pooling2dDescriptor poolingDescriptor;
    poolingDescriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    poolingDescriptor.m_PoolWidth     = 2;
    poolingDescriptor.m_PoolHeight    = 2;
    poolingDescriptor.m_StrideX       = 2;
    poolingDescriptor.m_StrideY       = 2;
    poolingDescriptor.m_PadLeft       = 1;
    poolingDescriptor.m_PadRight      = 1;
    poolingDescriptor.m_PadTop        = 1;
    poolingDescriptor.m_PadBottom     = 1;
    poolingDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    poolingDescriptor.m_DataLayout    = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Pooling2dLayer* const poolingLayer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                         "pooling layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(poolingLayer->GetInputSlot(0));
    poolingLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(poolingLayer),
                                  CreateOutputsFrom({poolingLayer}),
                                  {poolingLayer});
}

// Creates a subgraph containing only unsupported layers (only convolutions are unsupported by the mock backend)
SubgraphView::SubgraphViewPtr BuildFullyUnsupportedSubgraph2(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);

    Pooling2dDescriptor poolingDescriptor;
    poolingDescriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    poolingDescriptor.m_PoolWidth     = 2;
    poolingDescriptor.m_PoolHeight    = 2;
    poolingDescriptor.m_StrideX       = 2;
    poolingDescriptor.m_StrideY       = 2;
    poolingDescriptor.m_PadLeft       = 1;
    poolingDescriptor.m_PadRight      = 1;
    poolingDescriptor.m_PadTop        = 1;
    poolingDescriptor.m_PadBottom     = 1;
    poolingDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    poolingDescriptor.m_DataLayout    = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Pooling2dLayer* const pooling1Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling1 layer", outputInfo);
    Pooling2dLayer* const pooling2Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling2 layer", outputInfo);
    Pooling2dLayer* const pooling3Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling3 layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(pooling1Layer->GetInputSlot(0));
    pooling1Layer->GetOutputSlot(0).Connect(pooling2Layer->GetInputSlot(0));
    pooling2Layer->GetOutputSlot(0).Connect(pooling3Layer->GetInputSlot(0));
    pooling3Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(pooling1Layer),
                                  CreateOutputsFrom({pooling3Layer}),
                                  {pooling1Layer,
                                   pooling2Layer,
                                   pooling3Layer});
}

// Creates a simple subgraph with only one convolution layer, supported by the mock backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo({ 1, 1, 1, 16 }, DataType::Signed32, 0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const convLayer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                              "conv layer", outputInfo);

  ConstantLayer* const weightsLayer =
      AddConstantLayer(graph, layersInGraph, "Weights Layer", constWeightsTensor, weightInfo);
  ConstantLayer* const biasLayer = AddConstantLayer(graph, layersInGraph, "Bias Layer", constBiasTensor, biasInfo);

    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
    biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<unsigned int> ignoreSlots = {1, 2};
    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(convLayer, ignoreSlots),
                                  CreateOutputsFrom({convLayer}),
                                  {convLayer, weightsLayer, biasLayer});
}

// Creates a subgraph with five convolutions layers, all supported by the mock backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph2(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", outputInfo);
    ConstantLayer* const weightsLayer1 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 1", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer1 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 1", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer", outputInfo);
    ConstantLayer* const weightsLayer2 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 2", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer2 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 2", constBiasTensor, biasInfo);


    Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv3 layer", outputInfo);
    ConstantLayer* const weightsLayer3 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 3", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer3 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 3", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv4Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv4 layer", outputInfo);
    ConstantLayer* const weightsLayer4 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 4", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer4 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 4", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv5Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv5 layer", outputInfo);
    ConstantLayer* const weightsLayer5 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 5", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer5 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 5", constBiasTensor, biasInfo);


    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    weightsLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(1));
    biasLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(2));

    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    weightsLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(1));
    biasLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(2));

    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    weightsLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(1));
    biasLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(2));

    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    weightsLayer4->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(1));
    biasLayer4->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(2));

    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    weightsLayer5->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(1));
    biasLayer5->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(2));

    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    std::vector<unsigned int> ignoreSlots = {1, 2};
    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(conv1Layer, ignoreSlots),
                                  CreateOutputsFrom({ conv5Layer }),
                                  { weightsLayer1,
                                    biasLayer1,
                                    conv1Layer,
                                    weightsLayer2,
                                    biasLayer2,
                                    conv2Layer,
                                    weightsLayer3,
                                    biasLayer3,
                                    conv3Layer,
                                    weightsLayer4,
                                    biasLayer4,
                                    conv4Layer,
                                    weightsLayer5,
                                    biasLayer5,
                                    conv5Layer });
}

// Creates a subgraph with both supported and unsupported layers
// (only convolutions are unsupported by the mock backend)
SubgraphView::SubgraphViewPtr BuildPartiallySupportedSubgraph(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    Pooling2dDescriptor poolingDescriptor;
    poolingDescriptor.m_PoolType      = armnn::PoolingAlgorithm::Average;
    poolingDescriptor.m_PoolWidth     = 2;
    poolingDescriptor.m_PoolHeight    = 2;
    poolingDescriptor.m_StrideX       = 2;
    poolingDescriptor.m_StrideY       = 2;
    poolingDescriptor.m_PadLeft       = 1;
    poolingDescriptor.m_PadRight      = 1;
    poolingDescriptor.m_PadTop        = 1;
    poolingDescriptor.m_PadBottom     = 1;
    poolingDescriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;
    poolingDescriptor.m_DataLayout    = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    ConstantLayer* const weightsLayer1 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 1", constWeightsTensor, weightInfo);

    ConstantLayer* const biasLayer1 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 1", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", outputInfo);
    Pooling2dLayer* const pooling1Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling1 layer", outputInfo);
    Pooling2dLayer* const pooling2Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling2 layer", outputInfo);

    ConstantLayer* const weightsLayer2 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 2", constWeightsTensor, weightInfo);

    ConstantLayer* const biasLayer2 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 2", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer", outputInfo);
    Pooling2dLayer* const pooling3Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling3 layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    weightsLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(1));
    biasLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(2));
    conv1Layer->GetOutputSlot(0).Connect(pooling1Layer->GetInputSlot(0));
    pooling1Layer->GetOutputSlot(0).Connect(pooling2Layer->GetInputSlot(0));
    pooling2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    weightsLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(1));
    biasLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(2));
    conv2Layer->GetOutputSlot(0).Connect(pooling3Layer->GetInputSlot(0));
    pooling3Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<unsigned int> ignoreSlots = {1, 2};
    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(conv1Layer, ignoreSlots),
                                  CreateOutputsFrom({pooling3Layer}),
                                  {weightsLayer1,
                                   biasLayer1,
                                   conv1Layer,
                                   pooling1Layer,
                                   pooling2Layer,
                                   weightsLayer2,
                                   biasLayer2,
                                   conv2Layer,
                                   pooling3Layer});
}

// Creates a subgraph with only unoptimizable layers ("unoptimizable" is added to the layer's name)
SubgraphView::SubgraphViewPtr BuildFullyUnoptimizableSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);
    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);

    ConstantLayer* const weightsLayer =
        AddConstantLayer(graph, layersInGraph, "Weights Layer unoptimizable", constWeightsTensor, weightInfo);

    ConstantLayer* const biasLayer =
        AddConstantLayer(graph, layersInGraph, "Bias Layer unoptimizable", constBiasTensor, biasInfo);

    Convolution2dLayer* const convLayer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv layer unoptimizable", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
    biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<unsigned int> ignoreSlots = {1, 2};
    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(convLayer, ignoreSlots),
                                  CreateOutputsFrom({convLayer}),
                                  {convLayer, weightsLayer, biasLayer});
}

// Creates a subgraph with some unoptimizable layers ("unoptimizable" is added to the layer's name)
SubgraphView::SubgraphViewPtr BuildPartiallyOptimizableSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);

    ConstantLayer* const weightsLayer1 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 1", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer1 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 1", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer2 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 2 unoptimizable", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer2 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 2 unoptimizable", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer3 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 3", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer3 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 3", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer4 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 4 unoptimizable", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer4 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 4 unoptimizable", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer5 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 5", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer5 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 5", constBiasTensor, biasInfo);

  Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                             "conv1 layer", outputInfo);
  Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                             "conv2 layer unoptimizable", outputInfo);
  Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                             "conv3 layer", outputInfo);
  Convolution2dLayer* const conv4Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                             "conv4 layer unoptimizable", outputInfo);
  Convolution2dLayer* const conv5Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                             "conv5 layer", outputInfo);

  Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    weightsLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(1));
    biasLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(2));

    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    weightsLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(1));
    biasLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(2));

    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    weightsLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(1));
    biasLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(2));

    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    weightsLayer4->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(1));
    biasLayer4->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(2));

    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    weightsLayer5->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(1));
    biasLayer5->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(2));

    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    std::vector<unsigned int> ignoreSlots = {1, 2};
    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom(conv1Layer, ignoreSlots),
                                  CreateOutputsFrom({conv5Layer}),
                                  {weightsLayer1,
                                    biasLayer1,
                                    conv1Layer,
                                    weightsLayer2,
                                    biasLayer2,
                                    conv2Layer,
                                    weightsLayer3,
                                    biasLayer3,
                                    conv3Layer,
                                    weightsLayer4,
                                    biasLayer4,
                                    conv4Layer,
                                    weightsLayer5,
                                    biasLayer5,
                                    conv5Layer});
}

// Creates a subgraph with some input unoptimizable layers ("unoptimizable" is added to the layer's name),
// this is meant to test input slots coming from different layers
SubgraphView::SubgraphViewPtr BuildPartiallyOptimizableSubgraph2(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    weightInfo.SetConstant(true);
    biasInfo.SetConstant(true);

    std::vector<float> weightsVector(64);
    ConstTensor constWeightsTensor(weightInfo, weightsVector);

    std::vector<float> biasVector(16);
    ConstTensor constBiasTensor(biasInfo, biasVector);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const input1Layer = AddInputLayer(graph, "input1 layer", inputInfo, 0);
    Layer* const input2Layer = AddInputLayer(graph, "input2 layer", inputInfo, 1);

    ConstantLayer* const weightsLayer1 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 1", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer1 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 1", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer2 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 2 unoptimizable", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer2 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 2 unoptimizable", constBiasTensor, biasInfo);
    ConstantLayer* const weightsLayer3 =
        AddConstantLayer(graph, layersInGraph, "Weights Layer 3", constWeightsTensor, weightInfo);
    ConstantLayer* const biasLayer3 =
        AddConstantLayer(graph, layersInGraph, "Bias Layer 3", constBiasTensor, biasInfo);

    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", outputInfo);
    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer unoptimizable", outputInfo);
    Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv3 layer", outputInfo);
    AdditionLayer* const addLayer = AddAdditionaLayer(graph, layersInGraph, "add layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    input1Layer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    weightsLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(1));
    biasLayer1->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(2));
    conv1Layer->GetOutputSlot(0).Connect(addLayer->GetInputSlot(0));

    input2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    weightsLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(1));
    biasLayer2->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(2));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    weightsLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(1));
    biasLayer3->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(2));
    conv3Layer->GetOutputSlot(0).Connect(addLayer->GetInputSlot(1));

    addLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    std::vector<unsigned int> ignoreSlots = {1, 2};
    return CreateSubgraphViewFrom(CreateInputsFrom({conv1Layer,
                                                    conv2Layer}, ignoreSlots),
                                  CreateOutputsFrom({addLayer}),
                                  { weightsLayer1,
                                    biasLayer1,
                                    weightsLayer2,
                                    biasLayer2,
                                    weightsLayer3,
                                    biasLayer3,
                                    conv1Layer,
                                    conv2Layer,
                                    conv3Layer,
                                    addLayer });
}

// The input subgraph contains only a single unsupported layer (only convolutions are unsupported by the mock backend)
void FullyUnsupporteSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyUnsupportedSubgraph1(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 1);

    CHECK(Contains(layersInGraph, "pooling layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // =======================================================================
    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole original one
    //  - No untouched subgraphs
    // =======================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    CHECK(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    CHECK(failedSubgraphs.size() == 1);

    CheckFailedSubgraph(failedSubgraphs.at(0),
                        { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                        subgraphInputSlots,
                        subgraphOutputSlots,
                        subgraphLayers);

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contains only unsupported layers (only convolutions are unsupported by the mock backend)
void FullyUnsupporteSubgraphTestImpl2()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyUnsupportedSubgraph2(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 3);

    CHECK(Contains(layersInGraph, "pooling1 layer"));
    CHECK(Contains(layersInGraph, "pooling2 layer"));
    CHECK(Contains(layersInGraph, "pooling3 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // =======================================================================
    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole original one
    //  - No untouched subgraphs
    // =======================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    CHECK(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    CHECK(failedSubgraphs.size() == 1);

    std::list<IConnectableLayer*> expectedFailedLayers{ layersInGraph.at("pooling1 layer"),
                                            layersInGraph.at("pooling2 layer"),
                                            layersInGraph.at("pooling3 layer") };

    const SubgraphView& failedSubgraph = failedSubgraphs.at(0);

    CheckFailedSubgraph(failedSubgraph,
                        { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                        subgraphInputSlots,
                        subgraphOutputSlots,
                        subgraphLayers);

    const SubgraphView::IConnectableLayers& failedSubgraphLayers = failedSubgraph.GetIConnectableLayers();

    CHECK_EQ(failedSubgraphLayers.front() + 0, expectedFailedLayers.front() + 0);
    CHECK_EQ(failedSubgraphLayers.front() + 1, expectedFailedLayers.front() + 1);
    CHECK_EQ(failedSubgraphLayers.front() + 2, expectedFailedLayers.front() + 2);

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());
}

// A simple case with only one layer (convolution) to optimize, supported by the mock backend
void FullyOptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph1(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 3);

    CHECK(Contains(layersInGraph, "conv layer"));
    CHECK(Contains(layersInGraph, "Weights Layer"));
    CHECK(Contains(layersInGraph, "Bias Layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ===========================================================================================
    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs
    // ===========================================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 1);

    CheckSubstitution(substitutions.at(0),
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), 1 },
                      subgraphInputSlots,
                      subgraphOutputSlots,
                      subgraphLayers);

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    CHECK(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());
}

// A case with five layers (all convolutions) to optimize, all supported by the mock backend
void FullyOptimizableSubgraphTestImpl2()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph2(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphPtr->GetIConnectableLayers().size() == 15);

    CHECK(Contains(layersInGraph, "conv1 layer"));
    CHECK(Contains(layersInGraph, "conv2 layer"));
    CHECK(Contains(layersInGraph, "conv3 layer"));
    CHECK(Contains(layersInGraph, "conv4 layer"));
    CHECK(Contains(layersInGraph, "conv5 layer"));
    CHECK(Contains(layersInGraph, "Weights Layer 1"));
    CHECK(Contains(layersInGraph, "Weights Layer 2"));
    CHECK(Contains(layersInGraph, "Weights Layer 3"));
    CHECK(Contains(layersInGraph, "Weights Layer 4"));
    CHECK(Contains(layersInGraph, "Weights Layer 5"));
    CHECK(Contains(layersInGraph, "Bias Layer 1"));
    CHECK(Contains(layersInGraph, "Bias Layer 2"));
    CHECK(Contains(layersInGraph, "Bias Layer 3"));
    CHECK(Contains(layersInGraph, "Bias Layer 4"));
    CHECK(Contains(layersInGraph, "Bias Layer 5"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ===========================================================================================
    // The expected results are:
    //  - Exactly one substitution, mapping the whole input subgraph to a new replacement subgraph
    //  - No failed subgraphs
    //  - No untouched subgraphs
    // ===========================================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 1);

    std::list<IConnectableLayer*> expectedSubstitutableLayers{
                                                   layersInGraph.at("Weights Layer 1"),
                                                   layersInGraph.at("Weights Layer 2"),
                                                   layersInGraph.at("Weights Layer 3"),
                                                   layersInGraph.at("Weights Layer 4"),
                                                   layersInGraph.at("Weights Layer 5"),
                                                   layersInGraph.at("Bias Layer 1"),
                                                   layersInGraph.at("Bias Layer 2"),
                                                   layersInGraph.at("Bias Layer 3"),
                                                   layersInGraph.at("Bias Layer 4"),
                                                   layersInGraph.at("Bias Layer 5"),
                                                   layersInGraph.at("conv1 layer"),
                                                   layersInGraph.at("conv2 layer"),
                                                   layersInGraph.at("conv3 layer"),
                                                   layersInGraph.at("conv4 layer"),
                                                   layersInGraph.at("conv5 layer")};

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    CheckSubstitution(
        substitution,
        {subgraphInputSlots.size(), subgraphOutputSlots.size(),
         subgraphLayers.size()},
        {subgraphInputSlots.size(), subgraphOutputSlots.size(), 1},
        subgraphInputSlots, subgraphOutputSlots, expectedSubstitutableLayers);

    const SubgraphView::IConnectableLayers& substitutableSubgraphLayers =
            substitution.m_SubstitutableSubgraph.GetIConnectableLayers();

    CHECK_EQ(substitutableSubgraphLayers.front() + 0, expectedSubstitutableLayers.front() + 0);
    CHECK_EQ(substitutableSubgraphLayers.front() + 1, expectedSubstitutableLayers.front() + 1);
    CHECK_EQ(substitutableSubgraphLayers.front() + 2, expectedSubstitutableLayers.front() + 2);
    CHECK_EQ(substitutableSubgraphLayers.front() + 3, expectedSubstitutableLayers.front() + 3);
    CHECK_EQ(substitutableSubgraphLayers.front() + 4, expectedSubstitutableLayers.front() + 4);

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    CHECK(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contaions both supported and unsupported layers
// (but only convolutions are unsupported by the mock backend)
void PartiallySupportedSubgraphTestImpl()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildPartiallySupportedSubgraph(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 9);

    CHECK(Contains(layersInGraph, "Weights Layer 1"));
    CHECK(Contains(layersInGraph, "Bias Layer 1"));
    CHECK(Contains(layersInGraph, "conv1 layer"));
    CHECK(Contains(layersInGraph, "pooling1 layer"));
    CHECK(Contains(layersInGraph, "pooling2 layer"));
    CHECK(Contains(layersInGraph, "Weights Layer 2"));
    CHECK(Contains(layersInGraph, "Bias Layer 2"));
    CHECK(Contains(layersInGraph, "conv2 layer"));
    CHECK(Contains(layersInGraph, "pooling3 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ========================================================================
    // The expected results are:
    //  - Exactly two substitution, corresponding to the supported layers
    //  - Exactly two failed subgraphs, corresponding to the unsupported layers
    //  - No untouched subgraphs
    // ========================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    OptimizationViews::Substitutions substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 2);
    // Sort into a consistent order
    std::sort(substitutions.begin(), substitutions.end(), [](auto s1, auto s2) {
        return strcmp(s1.m_SubstitutableSubgraph.GetIConnectableLayers().front()->GetName(),
                      s2.m_SubstitutableSubgraph.GetIConnectableLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedSubstitutableSubgraphSizes{ { 1, 1, 3 },
                                                                          { 1, 1, 3 } };
    std::vector<ExpectedSubgraphSize> expectedReplacementSubgraphSizes{ { 1, 1, 1 },
                                                                        { 1, 1, 1 } };
    std::vector<SubgraphView::IInputSlots> expectedSubstitutableInputSlots
    {
            ConvertSlotsToISlots<InputSlot, IInputSlot>(
                {ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlot(0))}),
            ConvertSlotsToISlots<InputSlot, IInputSlot>(
                {ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer")->GetInputSlot(0))})
    };

    std::vector<SubgraphView::IOutputSlots> expectedSubstitutableOutputSlots
    {
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
                ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetOutputSlots())),
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
                ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer")->GetOutputSlots()))
    };
    std::vector<SubgraphView::IConnectableLayers> expectedSubstitutableLayers
    {
        { layersInGraph.at("Weights Layer 1"), layersInGraph.at("Bias Layer 1"), layersInGraph.at("conv1 layer") },
        { layersInGraph.at("Weights Layer 2"), layersInGraph.at("Bias Layer 2"), layersInGraph.at("conv2 layer") }
    };

    for (size_t substitutionIndex = 0; substitutionIndex < substitutions.size(); substitutionIndex++)
    {
        CheckSubstitution(substitutions.at(substitutionIndex),
                          expectedSubstitutableSubgraphSizes.at(substitutionIndex),
                          expectedReplacementSubgraphSizes.at(substitutionIndex),
                          expectedSubstitutableInputSlots.at(substitutionIndex),
                          expectedSubstitutableOutputSlots.at(substitutionIndex),
                          expectedSubstitutableLayers.at(substitutionIndex));
    }

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    OptimizationViews::Subgraphs failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    CHECK(failedSubgraphs.size() == 2);
    // Sort into a consistent order
    std::sort(failedSubgraphs.begin(), failedSubgraphs.end(), [](auto s1, auto s2) {
        return strcmp(s1.GetIConnectableLayers().front()->GetName(),
                      s2.GetIConnectableLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedFailedSubgraphSizes{ { 1, 1, 2 },
                                                                   { 1, 1, 1 } };
    std::vector<SubgraphView::IInputSlots> expectedFailedInputSlots
    {
        ConvertSlotsToISlots<InputSlot, IInputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling1 layer")->GetInputSlots())),
        ConvertSlotsToISlots<InputSlot, IInputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling3 layer")->GetInputSlots()))
    };
    std::vector<SubgraphView::IOutputSlots> expectedFailedOutputSlots
    {
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling2 layer")->GetOutputSlots())),
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling3 layer")->GetOutputSlots()))
    };
    std::vector<SubgraphView::IConnectableLayers> expectedFailedLayers
    {
        { layersInGraph.at("pooling1 layer"),
          layersInGraph.at("pooling2 layer") },
        { layersInGraph.at("pooling3 layer") }
    };

    for (size_t failedIndex = 0; failedIndex < failedSubgraphs.size(); failedIndex++)
    {
        CheckFailedSubgraph(failedSubgraphs.at(failedIndex),
                            expectedFailedSubgraphSizes.at(failedIndex),
                            expectedFailedInputSlots.at(failedIndex),
                            expectedFailedOutputSlots.at(failedIndex),
                            expectedFailedLayers.at(failedIndex));
    }

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    CHECK(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contains only unoptimizable layers ("unoptimizable" is added to the layer's name)
void FullyUnoptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyUnoptimizableSubgraph1(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 3);

    CHECK(Contains(layersInGraph, "conv layer unoptimizable"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ============================================================================
    // The expected results are:
    //  - No substitutions
    //  - No failed subgraphs
    //  - Exactly one untouched subgraph, corresponding to the whole input subgraph
    // ============================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    CHECK(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    CHECK(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    const OptimizationViews::Subgraphs& untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    CHECK(untouchedSubgraphs.size() == 1);

    CheckUntouchedSubgraph(untouchedSubgraphs.at(0),
                           {subgraphInputSlots.size(),
                            subgraphOutputSlots.size(), subgraphLayers.size()},
                           subgraphInputSlots, subgraphOutputSlots,
                           subgraphLayers);
}

// The input subgraph contains some unoptimizable layers ("unoptimizable" is added to the layer's name)
void PartiallyOptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildPartiallyOptimizableSubgraph1(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 1);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 15);

    CHECK(Contains(layersInGraph, "conv1 layer"));
    CHECK(Contains(layersInGraph, "conv2 layer unoptimizable"));
    CHECK(Contains(layersInGraph, "conv3 layer"));
    CHECK(Contains(layersInGraph, "conv4 layer unoptimizable"));
    CHECK(Contains(layersInGraph, "conv5 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ===============================================================================
    // The expected results are:
    //  - Exactly three substitutions, corresponding to the optimizable layers
    //  - No failed subgraphs
    //  - Exactly two untouched subgraphs, corresponding to the non-optimizable layers
    // ===============================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    OptimizationViews::Substitutions substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 3);
    // Sort into a consistent order
    std::sort(substitutions.begin(), substitutions.end(),
        [](auto s1, auto s2)
        { return strcmp(s1.m_SubstitutableSubgraph.GetIConnectableLayers().front()->GetName(),
                        s2.m_SubstitutableSubgraph.GetIConnectableLayers().front()->GetName()) < 0; });

    std::vector<ExpectedSubgraphSize> expectedSubstitutableSubgraphSizes{ { 1, 1, 3 },
                                                                          { 1, 1, 3 },
                                                                          { 1, 1, 3 } };
    std::vector<ExpectedSubgraphSize> expectedReplacementSubgraphSizes{ { 1, 1, 1 },
                                                                        { 1, 1, 1 },
                                                                        { 1, 1, 1 } };
    std::vector<SubgraphView::IInputSlots> expectedSubstitutableInputSlots
    {
        ConvertSlotsToISlots<InputSlot, IInputSlot>(
            {ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlot(0))}),
        ConvertSlotsToISlots<InputSlot, IInputSlot>(
        {ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetInputSlot(0))}),
        ConvertSlotsToISlots<InputSlot, IInputSlot>(
        {ConvertReferenceTypeToPointerType(layersInGraph.at("conv5 layer")->GetInputSlot(0))})
    };
    std::vector<SubgraphView::IOutputSlots> expectedSubstitutableOutputSlots
    {
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetOutputSlots())),
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetOutputSlots())),
        ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv5 layer")->GetOutputSlots()))
    };
    std::vector<SubgraphView::IConnectableLayers> expectedSubstitutableLayers
    {
        { layersInGraph.at("Weights Layer 1"), layersInGraph.at("Bias Layer 1"), layersInGraph.at("conv1 layer") },
        { layersInGraph.at("Weights Layer 3"), layersInGraph.at("Bias Layer 3"), layersInGraph.at("conv3 layer") },
        { layersInGraph.at("Weights Layer 5"), layersInGraph.at("Bias Layer 5"), layersInGraph.at("conv5 layer") }
    };

    for (size_t substitutionIndex = 0; substitutionIndex < substitutions.size(); substitutionIndex++)
    {
        CheckSubstitution(substitutions.at(substitutionIndex),
                          expectedSubstitutableSubgraphSizes.at(substitutionIndex),
                          expectedReplacementSubgraphSizes.at(substitutionIndex),
                          expectedSubstitutableInputSlots.at(substitutionIndex),
                          expectedSubstitutableOutputSlots.at(substitutionIndex),
                          expectedSubstitutableLayers.at(substitutionIndex));
    }

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    CHECK(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    OptimizationViews::Subgraphs untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    CHECK(untouchedSubgraphs.size() == 2);
    // Sort into a consistent order
    std::sort(untouchedSubgraphs.begin(), untouchedSubgraphs.end(), [](auto s1, auto s2) {
        return strcmp(s1.GetIConnectableLayers().front()->GetName(),
                      s2.GetIConnectableLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedUntouchedSubgraphSizes{ { 1, 1, 3 },
                                                                      { 1, 1, 3 } };
    std::vector<SubgraphView::IInputSlots> expectedUntouchedInputSlots{
        ConvertSlotsToISlots<InputSlot,
                             IInputSlot>({ConvertReferenceTypeToPointerType(
            layersInGraph.at("conv2 layer unoptimizable")->GetInputSlot(0))}),
        ConvertSlotsToISlots<InputSlot,
                             IInputSlot>({ConvertReferenceTypeToPointerType(
            layersInGraph.at("conv4 layer unoptimizable")->GetInputSlot(0))})};

    std::vector<SubgraphView::IOutputSlots> expectedUntouchedOutputSlots
        {
            ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
                ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetOutputSlots())),
            ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
                ConvertReferenceTypeToPointerType(layersInGraph.at("conv4 layer unoptimizable")->GetOutputSlots()))
        };

    std::vector<SubgraphView::IConnectableLayers> expectedUntouchedLayers
        {
            { layersInGraph.at("Weights Layer 2 unoptimizable"),
              layersInGraph.at("Bias Layer 2 unoptimizable"),
              layersInGraph.at("conv2 layer unoptimizable") },
            { layersInGraph.at("Weights Layer 4 unoptimizable"),
              layersInGraph.at("Bias Layer 4 unoptimizable"),
              layersInGraph.at("conv4 layer unoptimizable") }
        };

    for (size_t untouchedIndex = 0; untouchedIndex < untouchedSubgraphs.size(); untouchedIndex++)
    {
        CheckUntouchedSubgraph(untouchedSubgraphs.at(untouchedIndex),
                               expectedUntouchedSubgraphSizes.at(untouchedIndex),
                               expectedUntouchedInputSlots.at(untouchedIndex),
                               expectedUntouchedOutputSlots.at(untouchedIndex),
                               expectedUntouchedLayers.at(untouchedIndex));
    }
}

// The input subgraph contains some unoptimizable layers ("unoptimizable" is added to the layer's name),
// this is meant to test input slots coming from different layers
void PartiallyOptimizableSubgraphTestImpl2()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a partially optimizable subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildPartiallyOptimizableSubgraph2(graph, layersInGraph);
    CHECK((subgraphPtr != nullptr));

    const SubgraphView::IInputSlots& subgraphInputSlots = subgraphPtr->GetIInputSlots();
    const SubgraphView::IOutputSlots& subgraphOutputSlots = subgraphPtr->GetIOutputSlots();
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraphPtr->GetIConnectableLayers();

    CHECK(subgraphInputSlots.size()  == 2);
    CHECK(subgraphOutputSlots.size() == 1);
    CHECK(subgraphLayers.size()      == 10);

    CHECK(Contains(layersInGraph, "conv1 layer"));
    CHECK(Contains(layersInGraph, "conv2 layer unoptimizable"));
    CHECK(Contains(layersInGraph, "conv3 layer"));
    CHECK(Contains(layersInGraph, "add layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    CHECK((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    CHECK_NOTHROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ==============================================================================
    // The expected results are:
    //  - Exactly one substitution, corresponding to the optimizable layers
    //  - No failed subgraphs
    //  - Exactly two untouched subgraphs, corresponding to the non-optimizable layer
    // ==============================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    const OptimizationViews::Substitutions& substitutions = optimizationViews.GetSubstitutions();
    CHECK(substitutions.size() == 1);

    ExpectedSubgraphSize expectedSubstitutableSubgraphSizes{ 2, 1, 7 };
    ExpectedSubgraphSize expectedReplacementSubgraphSizes{ 2, 1, 1 };

    SubgraphView::IInputSlots expectedSubstitutableInputSlots
    {
            ConvertSlotsToISlots<InputSlot, IInputSlot>({
                    ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlots()[0])})[0],
            ConvertSlotsToISlots<InputSlot, IInputSlot>({
                    ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetInputSlots()[0])})[0]
    };

    SubgraphView::IOutputSlots expectedSubstitutableOutputSlots
    {
            ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
                    ConvertReferenceTypeToPointerType(layersInGraph.at("add layer")->GetOutputSlots()))
    };

    SubgraphView::IConnectableLayers expectedSubstitutableLayers
    {
        layersInGraph.at("Weights Layer 1"),
        layersInGraph.at("Weights Layer 3"),
        layersInGraph.at("Bias Layer 1"),
        layersInGraph.at("Bias Layer 3"),
        layersInGraph.at("conv1 layer"),
        layersInGraph.at("conv3 layer"),
        layersInGraph.at("add layer")
    };

    CheckSubstitution(substitutions[0],
                      expectedSubstitutableSubgraphSizes,
                      expectedReplacementSubgraphSizes,
                      expectedSubstitutableInputSlots,
                      expectedSubstitutableOutputSlots,
                      expectedSubstitutableLayers);

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    CHECK(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    const OptimizationViews::Subgraphs& untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    CHECK(untouchedSubgraphs.size() == 1);

    std::vector<ExpectedSubgraphSize> expectedUntouchedSubgraphSizes{ { 1, 1, 3 } };
    std::vector<SubgraphView::IInputSlots> expectedUntouchedInputSlots
    {
        ConvertSlotsToISlots<InputSlot,
                             IInputSlot>({ConvertReferenceTypeToPointerType(
            layersInGraph.at("conv2 layer unoptimizable")->GetInputSlot(0))})};
    std::vector<SubgraphView::IOutputSlots> expectedUntouchedOutputSlots
    {
            ConvertSlotsToISlots<OutputSlot, IOutputSlot>(
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetOutputSlots()))
    };
    std::vector<SubgraphView::IConnectableLayers> expectedUntouchedLayers
    {
        { layersInGraph.at("conv2 layer unoptimizable"), layersInGraph.at("Weights Layer 2 unoptimizable"),
        layersInGraph.at("Bias Layer 2 unoptimizable") }
    };

    for (size_t untouchedIndex = 0; untouchedIndex < untouchedSubgraphs.size(); untouchedIndex++)
    {
        CheckUntouchedSubgraph(untouchedSubgraphs.at(untouchedIndex),
                               expectedUntouchedSubgraphSizes.at(untouchedIndex),
                               expectedUntouchedInputSlots.at(untouchedIndex),
                               expectedUntouchedOutputSlots.at(untouchedIndex),
                               expectedUntouchedLayers.at(untouchedIndex));
    }
}

} // Anonymous namespace

TEST_SUITE("OptimizeSubGraph")
{
TEST_CASE("FullyUnsupportedSubgraph1")     { FullyUnsupporteSubgraphTestImpl1();      }
TEST_CASE("FullyUnsupportedSubgraph2")     { FullyUnsupporteSubgraphTestImpl2();      }
TEST_CASE("FullyOptimizableSubgraph1")     { FullyOptimizableSubgraphTestImpl1();     }
TEST_CASE("FullyOptimizableSubgraph2")     { FullyOptimizableSubgraphTestImpl2();     }
TEST_CASE("PartiallySupportedSubgraph")    { PartiallySupportedSubgraphTestImpl();    }
TEST_CASE("FullyUnoptimizableSubgraph")    { FullyUnoptimizableSubgraphTestImpl1();   }
TEST_CASE("PartiallyOptimizableSubgraph1") { PartiallyOptimizableSubgraphTestImpl1(); }
TEST_CASE("PartiallyOptimizableSubgraph2") { PartiallyOptimizableSubgraphTestImpl2(); }

}
