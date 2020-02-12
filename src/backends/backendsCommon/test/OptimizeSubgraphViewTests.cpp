//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "CommonTestUtils.hpp"
#include "MockBackend.hpp"
#include "MockBackendId.hpp"

#include <Graph.hpp>
#include <Network.hpp>

#include <armnn/BackendRegistry.hpp>

#include <boost/test/unit_test.hpp>

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

// Convenience function to add an input layer to a graph
Layer* AddInputLayer(Graph& graph,
                     const std::string& layerName,
                     const TensorInfo& inputInfo,
                     LayerBindingId inputId = 0)
{
    Layer* const inputLayer = graph.AddLayer<InputLayer>(inputId, layerName.c_str());
    BOOST_TEST(inputLayer);
    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    return inputLayer;
}

// Convenience function to add an output layer to a graph
Layer* AddOutputLayer(Graph& graph,
                      const std::string& layerName)
{
    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, layerName.c_str());
    BOOST_TEST(outputLayer);
    return outputLayer;
}

// Convenience function to add a convolution layer to a graph
Convolution2dLayer* AddConvolutionLayer(Graph& graph,
                                        LayerNameToLayerMap& layersInGraph,
                                        const Convolution2dDescriptor& convolutionDescriptor,
                                        const std::string& layerName,
                                        const TensorInfo& weightInfo,
                                        const TensorInfo& biasInfo,
                                        const TensorInfo& outputInfo)
{
    Convolution2dLayer* const convLayer = graph.AddLayer<Convolution2dLayer>(convolutionDescriptor, layerName.c_str());
    BOOST_TEST(convLayer);
    SetWeightAndBias(convLayer, weightInfo, biasInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(convLayer->GetName(), convLayer));
    return convLayer;
}

// Convenience function to add a pooling layer to a graph
Pooling2dLayer* AddPoolingLayer(Graph& graph,
                                LayerNameToLayerMap& layersInGraph,
                                const Pooling2dDescriptor& poolingDescriptor,
                                const std::string& layerName,
                                const TensorInfo& outputInfo)
{
    Pooling2dLayer* const poolingLayer = graph.AddLayer<Pooling2dLayer>(poolingDescriptor, layerName.c_str());
    BOOST_TEST(poolingLayer);
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
    BOOST_TEST(additionLayer);
    additionLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    layersInGraph.insert(std::make_pair(additionLayer->GetName(), additionLayer));
    return additionLayer;
}

// Convenience function to check that the given substitution matches the specified expected values
void CheckSubstitution(const OptimizationViews::SubstitutionPair& substitution,
                       const ExpectedSubgraphSize& expectedSubstitutableSubgraphSize,
                       const ExpectedSubgraphSize& expectedReplacementSubgraphSize,
                       const SubgraphView::InputSlots& expectedSubstitutableInputSlots,
                       const SubgraphView::OutputSlots& expectedSubstitutableOutputSlots,
                       const SubgraphView::Layers& expectedSubstitutableLayers)
{
    const SubgraphView&              substitutableSubgraph            = substitution.m_SubstitutableSubgraph;
    const SubgraphView::InputSlots&  substitutableSubgraphInputSlots  = substitutableSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& substitutableSubgraphOutputSlots = substitutableSubgraph.GetOutputSlots();
    const SubgraphView::Layers&      substitutableSubgraphLayers      = substitutableSubgraph.GetLayers();

    const SubgraphView&              replacementSubgraph            = substitution.m_ReplacementSubgraph;
    const SubgraphView::InputSlots&  replacementSubgraphInputSlots  = replacementSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& replacementSubgraphOutputSlots = replacementSubgraph.GetOutputSlots();
    const SubgraphView::Layers&      replacementSubgraphLayers      = replacementSubgraph.GetLayers();

    BOOST_TEST(substitutableSubgraphInputSlots.size()  == expectedSubstitutableSubgraphSize.m_NumInputSlots);
    BOOST_TEST(substitutableSubgraphOutputSlots.size() == expectedSubstitutableSubgraphSize.m_NumOutputSlots);
    BOOST_TEST(substitutableSubgraphLayers.size()      == expectedSubstitutableSubgraphSize.m_NumLayers);

    BOOST_TEST(AreEqual(substitutableSubgraphInputSlots,  expectedSubstitutableInputSlots));
    BOOST_TEST(AreEqual(substitutableSubgraphOutputSlots, expectedSubstitutableOutputSlots));
    BOOST_TEST(AreEqual(substitutableSubgraphLayers,      expectedSubstitutableLayers));

    BOOST_TEST(replacementSubgraphInputSlots.size()  == expectedReplacementSubgraphSize.m_NumInputSlots);
    BOOST_TEST(replacementSubgraphOutputSlots.size() == expectedReplacementSubgraphSize.m_NumOutputSlots);
    BOOST_TEST(replacementSubgraphLayers.size()      == expectedReplacementSubgraphSize.m_NumLayers);

    BOOST_TEST(!AreEqual(replacementSubgraphInputSlots,  expectedSubstitutableInputSlots));
    BOOST_TEST(!AreEqual(replacementSubgraphOutputSlots, expectedSubstitutableOutputSlots));
    BOOST_TEST(!AreEqual(replacementSubgraphLayers,      expectedSubstitutableLayers));

    BOOST_TEST(std::all_of(replacementSubgraphLayers.begin(),
                           replacementSubgraphLayers.end(),
                           [](const Layer* layer)
    {
        return layer->GetType() == LayerType::PreCompiled;
    }));
}

// Convenience function to check that the given failed subgraph matches the specified expected values
void CheckFailedSubgraph(const SubgraphView& failedSubgraph,
                         const ExpectedSubgraphSize& expectedFailedSubgraphSize,
                         const SubgraphView::InputSlots& expectedFailedInputSlots,
                         const SubgraphView::OutputSlots& expectedFailedOutputSlots,
                         const SubgraphView::Layers& expectedFailedLayers)
{
    const SubgraphView::InputSlots&  failedSubgraphInputSlots  = failedSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& failedSubgraphOutputSlots = failedSubgraph.GetOutputSlots();
    const SubgraphView::Layers&      failedSubgraphLayers      = failedSubgraph.GetLayers();

    BOOST_TEST(failedSubgraphInputSlots.size()  == expectedFailedSubgraphSize.m_NumInputSlots);
    BOOST_TEST(failedSubgraphOutputSlots.size() == expectedFailedSubgraphSize.m_NumOutputSlots);
    BOOST_TEST(failedSubgraphLayers.size()      == expectedFailedSubgraphSize.m_NumLayers);

    BOOST_TEST(AreEqual(failedSubgraphInputSlots,  expectedFailedInputSlots));
    BOOST_TEST(AreEqual(failedSubgraphOutputSlots, expectedFailedOutputSlots));
    BOOST_TEST(AreEqual(failedSubgraphLayers,      expectedFailedLayers));
}

// Convenience function to check that the given untouched subgraph matches the specified expected values
void CheckUntouchedSubgraph(const SubgraphView& untouchedSubgraph,
                            const ExpectedSubgraphSize& expectedUntouchedSubgraphSize,
                            const SubgraphView::InputSlots& expectedUntouchedInputSlots,
                            const SubgraphView::OutputSlots& expectedUntouchedOutputSlots,
                            const SubgraphView::Layers& expectedUntouchedLayers)
{
    const SubgraphView::InputSlots&  untouchedSubgraphInputSlots  = untouchedSubgraph.GetInputSlots();
    const SubgraphView::OutputSlots& untouchedSubgraphOutputSlots = untouchedSubgraph.GetOutputSlots();
    const SubgraphView::Layers&      untouchedSubgraphLayers      = untouchedSubgraph.GetLayers();

    BOOST_TEST(untouchedSubgraphInputSlots.size()  == expectedUntouchedSubgraphSize.m_NumInputSlots);
    BOOST_TEST(untouchedSubgraphOutputSlots.size() == expectedUntouchedSubgraphSize.m_NumOutputSlots);
    BOOST_TEST(untouchedSubgraphLayers.size()      == expectedUntouchedSubgraphSize.m_NumLayers);

    BOOST_TEST(AreEqual(untouchedSubgraphInputSlots,  expectedUntouchedInputSlots));
    BOOST_TEST(AreEqual(untouchedSubgraphOutputSlots, expectedUntouchedOutputSlots));
    BOOST_TEST(AreEqual(untouchedSubgraphLayers,      expectedUntouchedLayers));
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
    return CreateSubgraphViewFrom(CreateInputsFrom({poolingLayer}),
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
    return CreateSubgraphViewFrom(CreateInputsFrom({pooling1Layer}),
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
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const convLayer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                              "conv layer", weightInfo, biasInfo, outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({convLayer}),
                                  CreateOutputsFrom({convLayer}),
                                  {convLayer});
}

// Creates a subgraph with five convolutions layers, all supported by the mock backend
SubgraphView::SubgraphViewPtr BuildFullyOptimizableSubgraph2(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv3 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv4Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv4 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv5Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv5 layer", weightInfo, biasInfo, outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({conv1Layer}),
                                  CreateOutputsFrom({conv5Layer}),
                                  {conv1Layer,
                                   conv2Layer,
                                   conv3Layer,
                                   conv4Layer,
                                   conv5Layer});
}

// Creates a subgraph with both supported and unsupported layers
// (only convolutions are unsupported by the mock backend)
SubgraphView::SubgraphViewPtr BuildPartiallySupportedSubgraph(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

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
    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", weightInfo, biasInfo, outputInfo);
    Pooling2dLayer* const pooling1Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling1 layer", outputInfo);
    Pooling2dLayer* const pooling2Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling2 layer", outputInfo);
    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer", weightInfo, biasInfo, outputInfo);
    Pooling2dLayer* const pooling3Layer = AddPoolingLayer(graph, layersInGraph, poolingDescriptor,
                                                          "pooling3 layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(pooling1Layer->GetInputSlot(0));
    pooling1Layer->GetOutputSlot(0).Connect(pooling2Layer->GetInputSlot(0));
    pooling2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(pooling3Layer->GetInputSlot(0));
    pooling3Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({conv1Layer}),
                                  CreateOutputsFrom({pooling3Layer}),
                                  {conv1Layer,
                                   pooling1Layer,
                                   pooling2Layer,
                                   conv2Layer,
                                   pooling3Layer});
}

// Creates a subgraph with only unoptimizable layers ("unoptimizable" is added to the layer's name)
SubgraphView::SubgraphViewPtr BuildFullyUnoptimizableSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const convLayer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv layer unoptimizable", weightInfo, biasInfo,
                                                               outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({convLayer}),
                                  CreateOutputsFrom({convLayer}),
                                  {convLayer});
}

// Creates a subgraph with some unoptimizable layers ("unoptimizable" is added to the layer's name)
SubgraphView::SubgraphViewPtr BuildPartiallyOptimizableSubgraph1(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const inputLayer = AddInputLayer(graph, "input layer", inputInfo);
    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer unoptimizable", weightInfo, biasInfo,
                                                               outputInfo);
    Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv3 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv4Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv4 layer unoptimizable", weightInfo, biasInfo,
                                                               outputInfo);
    Convolution2dLayer* const conv5Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv5 layer", weightInfo, biasInfo, outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    inputLayer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    conv3Layer->GetOutputSlot(0).Connect(conv4Layer->GetInputSlot(0));
    conv4Layer->GetOutputSlot(0).Connect(conv5Layer->GetInputSlot(0));
    conv5Layer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({conv1Layer}),
                                  CreateOutputsFrom({conv5Layer}),
                                  {conv1Layer,
                                   conv2Layer,
                                   conv3Layer,
                                   conv4Layer,
                                   conv5Layer});
}

// Creates a subgraph with some input unoptimizable layers ("unoptimizable" is added to the layer's name),
// this is meant to test input slots coming from different layers
SubgraphView::SubgraphViewPtr BuildPartiallyOptimizableSubgraph2(Graph& graph, LayerNameToLayerMap& layersInGraph)
{
    const TensorInfo inputInfo ({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo outputInfo({  1, 16, 16, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo weightInfo({ 16,  1,  1, 16 }, DataType::QAsymmU8, 0.9f, 0);
    const TensorInfo biasInfo  ({  1,  1,  1, 16 }, DataType::Signed32,        0.9f, 0);

    Convolution2dDescriptor convolutionDescriptor;
    convolutionDescriptor.m_StrideX     = 1;
    convolutionDescriptor.m_StrideY     = 1;
    convolutionDescriptor.m_BiasEnabled = true;
    convolutionDescriptor.m_DataLayout  = DataLayout::NHWC;

    // Construct the graph
    Layer* const input1Layer = AddInputLayer(graph, "input1 layer", inputInfo, 0);
    Layer* const input2Layer = AddInputLayer(graph, "input2 layer", inputInfo, 1);
    Convolution2dLayer* const conv1Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv1 layer", weightInfo, biasInfo, outputInfo);
    Convolution2dLayer* const conv2Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv2 layer unoptimizable", weightInfo, biasInfo,
                                                               outputInfo);
    Convolution2dLayer* const conv3Layer = AddConvolutionLayer(graph, layersInGraph, convolutionDescriptor,
                                                               "conv3 layer", weightInfo, biasInfo, outputInfo);
    AdditionLayer* const addLayer = AddAdditionaLayer(graph, layersInGraph, "add layer", outputInfo);
    Layer* const outputLayer = AddOutputLayer(graph, "output layer");

    // Connect the network
    input1Layer->GetOutputSlot(0).Connect(conv1Layer->GetInputSlot(0));
    input2Layer->GetOutputSlot(0).Connect(conv2Layer->GetInputSlot(0));
    conv1Layer->GetOutputSlot(0).Connect(addLayer->GetInputSlot(0));
    conv2Layer->GetOutputSlot(0).Connect(conv3Layer->GetInputSlot(0));
    conv3Layer->GetOutputSlot(0).Connect(addLayer->GetInputSlot(1));
    addLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Create the subgraph view for the whole network
    return CreateSubgraphViewFrom(CreateInputsFrom({conv1Layer,
                                                    conv2Layer}),
                                  CreateOutputsFrom({addLayer}),
                                  {conv1Layer,
                                   conv2Layer,
                                   conv3Layer,
                                   addLayer});
}

// The input subgraph contains only a single unsupported layer (only convolutions are unsupported by the mock backend)
void FullyUnsupporteSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyUnsupportedSubgraph1(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 1);

    BOOST_TEST(Contains(layersInGraph, "pooling layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // =======================================================================
    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole original one
    //  - No untouched subgraphs
    // =======================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    BOOST_TEST(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    BOOST_TEST(failedSubgraphs.size() == 1);

    CheckFailedSubgraph(failedSubgraphs.at(0),
                        { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                        subgraphInputSlots,
                        subgraphOutputSlots,
                        subgraphLayers);

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contains only unsupported layers (only convolutions are unsupported by the mock backend)
void FullyUnsupporteSubgraphTestImpl2()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create an unsupported subgraph
    SubgraphView::SubgraphViewPtr subgraphPtr = BuildFullyUnsupportedSubgraph2(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 3);

    BOOST_TEST(Contains(layersInGraph, "pooling1 layer"));
    BOOST_TEST(Contains(layersInGraph, "pooling2 layer"));
    BOOST_TEST(Contains(layersInGraph, "pooling3 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly, but no optimization is performed
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // =======================================================================
    // The expected results are:
    //  - No substitutions
    //  - Exactly one failed subgraph, corresponding to the whole original one
    //  - No untouched subgraphs
    // =======================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    BOOST_TEST(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    const OptimizationViews::Subgraphs& failedSubgraphs = optimizationViews.GetFailedSubgraphs();
    BOOST_TEST(failedSubgraphs.size() == 1);

    std::vector<Layer*> expectedFailedLayers{ layersInGraph.at("pooling1 layer"),
                                              layersInGraph.at("pooling2 layer"),
                                              layersInGraph.at("pooling3 layer") };

    const SubgraphView& failedSubgraph = failedSubgraphs.at(0);

    CheckFailedSubgraph(failedSubgraph,
                        { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                        subgraphInputSlots,
                        subgraphOutputSlots,
                        subgraphLayers);

    const SubgraphView::Layers& failedSubgraphLayers = failedSubgraph.GetLayers();

    BOOST_TEST(failedSubgraphLayers.front() + 0, expectedFailedLayers.at(0));
    BOOST_TEST(failedSubgraphLayers.front() + 1, expectedFailedLayers.at(1));
    BOOST_TEST(failedSubgraphLayers.front() + 2, expectedFailedLayers.at(2));

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());
}

// A simple case with only one layer (convolution) to optimize, supported by the mock backend
void FullyOptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph1(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 1);

    BOOST_TEST(Contains(layersInGraph, "conv layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

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
    BOOST_TEST(substitutions.size() == 1);

    CheckSubstitution(substitutions.at(0),
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), 1 },
                      subgraphInputSlots,
                      subgraphOutputSlots,
                      subgraphLayers);

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());
}

// A case with five layers (all convolutions) to optimize, all supported by the mock backend
void FullyOptimizableSubgraphTestImpl2()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyOptimizableSubgraph2(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphPtr->GetInputSlots().size()  == 1);
    BOOST_TEST(subgraphPtr->GetOutputSlots().size() == 1);
    BOOST_TEST(subgraphPtr->GetLayers().size()      == 5);

    BOOST_TEST(Contains(layersInGraph, "conv1 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv2 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv3 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv4 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv5 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

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
    BOOST_TEST(substitutions.size() == 1);

    std::list<Layer*> expectedSubstitutableLayers{ layersInGraph.at("conv1 layer"),
                                                   layersInGraph.at("conv2 layer"),
                                                   layersInGraph.at("conv3 layer"),
                                                   layersInGraph.at("conv4 layer"),
                                                   layersInGraph.at("conv5 layer") };

    const OptimizationViews::SubstitutionPair& substitution = substitutions.at(0);

    CheckSubstitution(substitution,
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                      { subgraphInputSlots.size(), subgraphOutputSlots.size(), 1 },
                      subgraphInputSlots,
                      subgraphOutputSlots,
                      expectedSubstitutableLayers);

    const SubgraphView::Layers& substitutableSubgraphLayers = substitution.m_SubstitutableSubgraph.GetLayers();

    BOOST_TEST(substitutableSubgraphLayers.front() + 0, expectedSubstitutableLayers.front() + 0);
    BOOST_TEST(substitutableSubgraphLayers.front() + 1, expectedSubstitutableLayers.front() + 1);
    BOOST_TEST(substitutableSubgraphLayers.front() + 2, expectedSubstitutableLayers.front() + 2);
    BOOST_TEST(substitutableSubgraphLayers.front() + 3, expectedSubstitutableLayers.front() + 3);
    BOOST_TEST(substitutableSubgraphLayers.front() + 4, expectedSubstitutableLayers.front() + 4);

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contaions both supported and unsupported layers
// (but only convolutions are unsupported by the mock backend)
void PartiallySupportedSubgraphTestImpl()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildPartiallySupportedSubgraph(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 5);

    BOOST_TEST(Contains(layersInGraph, "conv1 layer"));
    BOOST_TEST(Contains(layersInGraph, "pooling1 layer"));
    BOOST_TEST(Contains(layersInGraph, "pooling2 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv2 layer"));
    BOOST_TEST(Contains(layersInGraph, "pooling3 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

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
    BOOST_TEST(substitutions.size() == 2);
    // Sort into a consistent order
    std::sort(substitutions.begin(), substitutions.end(), [](auto s1, auto s2) {
        return strcmp(s1.m_SubstitutableSubgraph.GetLayers().front()->GetName(),
                      s2.m_SubstitutableSubgraph.GetLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedSubstitutableSubgraphSizes{ { 1, 1, 1 },
                                                                          { 1, 1, 1 } };
    std::vector<ExpectedSubgraphSize> expectedReplacementSubgraphSizes{ { 1, 1, 1 },
                                                                        { 1, 1, 1 } };
    std::vector<SubgraphView::InputSlots> expectedSubstitutableInputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer")->GetInputSlots())
    };
    std::vector<SubgraphView::OutputSlots> expectedSubstitutableOutputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetOutputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer")->GetOutputSlots())
    };
    std::vector<SubgraphView::Layers> expectedSubstitutableLayers
    {
        { layersInGraph.at("conv1 layer") },
        { layersInGraph.at("conv2 layer") }
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
    BOOST_TEST(failedSubgraphs.size() == 2);
    // Sort into a consistent order
    std::sort(failedSubgraphs.begin(), failedSubgraphs.end(), [](auto s1, auto s2) {
        return strcmp(s1.GetLayers().front()->GetName(), s2.GetLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedFailedSubgraphSizes{ { 1, 1, 2 },
                                                                   { 1, 1, 1 } };
    std::vector<SubgraphView::InputSlots> expectedFailedInputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling1 layer")->GetInputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling3 layer")->GetInputSlots())
    };
    std::vector<SubgraphView::OutputSlots> expectedFailedOutputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling2 layer")->GetOutputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("pooling3 layer")->GetOutputSlots())
    };
    std::vector<SubgraphView::Layers> expectedFailedLayers
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

    BOOST_TEST(optimizationViews.GetUntouchedSubgraphs().empty());
}

// The input subgraph contains only unoptimizable layers ("unoptimizable" is added to the layer's name)
void FullyUnoptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildFullyUnoptimizableSubgraph1(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 1);

    BOOST_TEST(Contains(layersInGraph, "conv layer unoptimizable"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

    // ============================================================================
    // The expected results are:
    //  - No substitutions
    //  - No failed subgraphs
    //  - Exactly one untouched subgraph, corresponding to the whole input subgraph
    // ============================================================================

    // -----------------------
    // Check the substitutions
    // -----------------------

    BOOST_TEST(optimizationViews.GetSubstitutions().empty());

    // --------------------------
    // Check the failed subgraphs
    // --------------------------

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    const OptimizationViews::Subgraphs& untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    BOOST_TEST(untouchedSubgraphs.size() == 1);

    CheckUntouchedSubgraph(untouchedSubgraphs.at(0),
                           { subgraphInputSlots.size(), subgraphOutputSlots.size(), subgraphLayers.size() },
                           subgraphInputSlots,
                           subgraphOutputSlots,
                           subgraphLayers);
}

// The input subgraph contains some unoptimizable layers ("unoptimizable" is added to the layer's name)
void PartiallyOptimizableSubgraphTestImpl1()
{
    Graph graph;
    LayerNameToLayerMap layersInGraph;

    // Create a fully optimizable subgraph
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildPartiallyOptimizableSubgraph1(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 1);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 5);

    BOOST_TEST(Contains(layersInGraph, "conv1 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv2 layer unoptimizable"));
    BOOST_TEST(Contains(layersInGraph, "conv3 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv4 layer unoptimizable"));
    BOOST_TEST(Contains(layersInGraph, "conv5 layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

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
    BOOST_TEST(substitutions.size() == 3);
    // Sort into a consistent order
    std::sort(substitutions.begin(), substitutions.end(),
        [](auto s1, auto s2) { return strcmp(s1.m_SubstitutableSubgraph.GetLayers().front()->GetName(),
                                             s2.m_SubstitutableSubgraph.GetLayers().front()->GetName()) < 0; });

    std::vector<ExpectedSubgraphSize> expectedSubstitutableSubgraphSizes{ { 1, 1, 1 },
                                                                          { 1, 1, 1 },
                                                                          { 1, 1, 1 } };
    std::vector<ExpectedSubgraphSize> expectedReplacementSubgraphSizes{ { 1, 1, 1 },
                                                                        { 1, 1, 1 },
                                                                        { 1, 1, 1 } };
    std::vector<SubgraphView::InputSlots> expectedSubstitutableInputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetInputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv5 layer")->GetInputSlots())
    };
    std::vector<SubgraphView::OutputSlots> expectedSubstitutableOutputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetOutputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetOutputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv5 layer")->GetOutputSlots())
    };
    std::vector<SubgraphView::Layers> expectedSubstitutableLayers
    {
        { layersInGraph.at("conv1 layer") },
        { layersInGraph.at("conv3 layer") },
        { layersInGraph.at("conv5 layer") }
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

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    OptimizationViews::Subgraphs untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    BOOST_TEST(untouchedSubgraphs.size() == 2);
    // Sort into a consistent order
    std::sort(untouchedSubgraphs.begin(), untouchedSubgraphs.end(), [](auto s1, auto s2) {
        return strcmp(s1.GetLayers().front()->GetName(), s2.GetLayers().front()->GetName()) < 0;
    });

    std::vector<ExpectedSubgraphSize> expectedUntouchedSubgraphSizes{ { 1, 1, 1 },
                                                                      { 1, 1, 1 } };
    std::vector<SubgraphView::InputSlots> expectedUntouchedInputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetInputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv4 layer unoptimizable")->GetInputSlots())
    };
    std::vector<SubgraphView::OutputSlots> expectedUntouchedOutputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetOutputSlots()),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv4 layer unoptimizable")->GetOutputSlots())
    };
    std::vector<SubgraphView::Layers> expectedUntouchedLayers
    {
        { layersInGraph.at("conv2 layer unoptimizable") },
        { layersInGraph.at("conv4 layer unoptimizable") }
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
    SubgraphViewSelector::SubgraphViewPtr subgraphPtr = BuildPartiallyOptimizableSubgraph2(graph, layersInGraph);
    BOOST_TEST((subgraphPtr != nullptr));

    const SubgraphView::InputSlots&  subgraphInputSlots  = subgraphPtr->GetInputSlots();
    const SubgraphView::OutputSlots& subgraphOutputSlots = subgraphPtr->GetOutputSlots();
    const SubgraphView::Layers&      subgraphLayers      = subgraphPtr->GetLayers();

    BOOST_TEST(subgraphInputSlots.size()  == 2);
    BOOST_TEST(subgraphOutputSlots.size() == 1);
    BOOST_TEST(subgraphLayers.size()      == 4);

    BOOST_TEST(Contains(layersInGraph, "conv1 layer"));
    BOOST_TEST(Contains(layersInGraph, "conv2 layer unoptimizable"));
    BOOST_TEST(Contains(layersInGraph, "conv3 layer"));
    BOOST_TEST(Contains(layersInGraph, "add layer"));

    // Create a mock backend object
    MockBackendInitialiser initialiser; // Register the Mock Backend
    auto backendObjPtr = CreateBackendObject(MockBackendId());
    BOOST_TEST((backendObjPtr != nullptr));

    // Optimize the subgraph
    OptimizationViews optimizationViews;

    // Check that the optimization is carried out correctly
    BOOST_CHECK_NO_THROW(optimizationViews = backendObjPtr->OptimizeSubgraphView(*subgraphPtr));

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
    BOOST_TEST(substitutions.size() == 1);

    ExpectedSubgraphSize expectedSubstitutableSubgraphSizes{ 2, 1, 3 };
    ExpectedSubgraphSize expectedReplacementSubgraphSizes{ 2, 1, 1 };

    SubgraphView::InputSlots expectedSubstitutableInputSlots = {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv1 layer")->GetInputSlots()[0]),
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv3 layer")->GetInputSlots()[0])
    };
    SubgraphView::OutputSlots expectedSubstitutableOutputSlots =
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("add layer")->GetOutputSlots()[0])
    };
    SubgraphView::Layers expectedSubstitutableLayers
    {
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

    BOOST_TEST(optimizationViews.GetFailedSubgraphs().empty());

    // -----------------------------
    // Check the untouched subgraphs
    // -----------------------------

    const OptimizationViews::Subgraphs& untouchedSubgraphs = optimizationViews.GetUntouchedSubgraphs();
    BOOST_TEST(untouchedSubgraphs.size() == 1);

    std::vector<ExpectedSubgraphSize> expectedUntouchedSubgraphSizes{ { 1, 1, 1 } };
    std::vector<SubgraphView::InputSlots> expectedUntouchedInputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetInputSlots())
    };
    std::vector<SubgraphView::OutputSlots> expectedUntouchedOutputSlots
    {
        ConvertReferenceTypeToPointerType(layersInGraph.at("conv2 layer unoptimizable")->GetOutputSlots())
    };
    std::vector<SubgraphView::Layers> expectedUntouchedLayers
    {
        { layersInGraph.at("conv2 layer unoptimizable") }
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

BOOST_AUTO_TEST_SUITE(OptimizeSubGraph)

BOOST_AUTO_TEST_CASE(FullyUnsupportedSubgraph1)     { FullyUnsupporteSubgraphTestImpl1();      }
BOOST_AUTO_TEST_CASE(FullyUnsupportedSubgraph2)     { FullyUnsupporteSubgraphTestImpl2();      }
BOOST_AUTO_TEST_CASE(FullyOptimizableSubgraph1)     { FullyOptimizableSubgraphTestImpl1();     }
BOOST_AUTO_TEST_CASE(FullyOptimizableSubgraph2)     { FullyOptimizableSubgraphTestImpl2();     }
BOOST_AUTO_TEST_CASE(PartiallySupportedSubgraph)    { PartiallySupportedSubgraphTestImpl();    }
BOOST_AUTO_TEST_CASE(FullyUnoptimizableSubgraph)    { FullyUnoptimizableSubgraphTestImpl1();   }
BOOST_AUTO_TEST_CASE(PartiallyOptimizableSubgraph1) { PartiallyOptimizableSubgraphTestImpl1(); }
BOOST_AUTO_TEST_CASE(PartiallyOptimizableSubgraph2) { PartiallyOptimizableSubgraphTestImpl2(); }

BOOST_AUTO_TEST_SUITE_END()
