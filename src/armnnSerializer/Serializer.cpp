//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Serializer.hpp"

#include "SerializerUtils.hpp"

#include <armnn/ArmNN.hpp>

#include <iostream>

#include <boost/numeric/conversion/cast.hpp>

#include <flatbuffers/util.h>

using namespace armnn;
namespace fb = flatbuffers;
namespace serializer = armnnSerializer;

namespace armnnSerializer
{

serializer::ActivationFunction GetFlatBufferActivationFunction(armnn::ActivationFunction function)
{
    switch (function)
    {
        case armnn::ActivationFunction::Sigmoid:
            return serializer::ActivationFunction::ActivationFunction_Sigmoid;
        case armnn::ActivationFunction::TanH:
            return serializer::ActivationFunction::ActivationFunction_TanH;
        case armnn::ActivationFunction::Linear:
            return serializer::ActivationFunction::ActivationFunction_Linear;
        case armnn::ActivationFunction::ReLu:
            return serializer::ActivationFunction::ActivationFunction_ReLu;
        case armnn::ActivationFunction::BoundedReLu:
            return serializer::ActivationFunction::ActivationFunction_BoundedReLu;
        case armnn::ActivationFunction::LeakyReLu:
            return serializer::ActivationFunction::ActivationFunction_LeakyReLu;
        case armnn::ActivationFunction::Abs:
            return serializer::ActivationFunction::ActivationFunction_Abs;
        case armnn::ActivationFunction::Sqrt:
            return serializer::ActivationFunction::ActivationFunction_Sqrt;
        case armnn::ActivationFunction::Square:
            return serializer::ActivationFunction::ActivationFunction_Square;
        default:
            return serializer::ActivationFunction::ActivationFunction_Sigmoid;
    }
}

uint32_t SerializerVisitor::GetSerializedId(unsigned int guid)
{
    std::pair<unsigned int, uint32_t> guidPair(guid, m_layerId);

    if (m_guidMap.empty())
    {
        m_guidMap.insert(guidPair);
    }
    else if (m_guidMap.find(guid) == m_guidMap.end())
    {
        guidPair.second = ++m_layerId;
        m_guidMap.insert(guidPair);
        return m_layerId;
    }
    return m_guidMap[guid];
}

// Build FlatBuffer for Input Layer
void SerializerVisitor::VisitInputLayer(const armnn::IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferInputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Input);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferInputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                flatBufferInputBaseLayer,
                                                                                id);
    // Push layer Guid to outputIds.
    m_inputIds.push_back(GetSerializedId(layer->GetGuid()));

    // Create the FlatBuffer InputLayer
    auto flatBufferInputLayer = serializer::CreateInputLayer(m_flatBufferBuilder, flatBufferInputBindableBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferInputLayer.o, serializer::Layer::Layer_InputLayer);
}

// Build FlatBuffer for Output Layer
void SerializerVisitor::VisitOutputLayer(const armnn::IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferOutputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Output);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferOutputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                 flatBufferOutputBaseLayer,
                                                                                 id);
    // Push layer Guid to outputIds.
    m_outputIds.push_back(GetSerializedId(layer->GetGuid()));

    // Create the FlatBuffer OutputLayer
    auto flatBufferOutputLayer = serializer::CreateOutputLayer(m_flatBufferBuilder, flatBufferOutputBindableBaseLayer);
    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferOutputLayer.o, serializer::Layer::Layer_OutputLayer);
}

// Build FlatBuffer for Activation Layer
void SerializerVisitor::VisitActivationLayer(const armnn::IConnectableLayer* layer,
                                             const armnn::ActivationDescriptor& descriptor,
                                             const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Activation);

    // Create the FlatBuffer ActivationDescriptor
    auto flatBufferDescriptor = CreateActivationDescriptor(m_flatBufferBuilder,
                                                           GetFlatBufferActivationFunction(descriptor.m_Function),
                                                           descriptor.m_A,
                                                           descriptor.m_B);

    // Create the FlatBuffer ActivationLayer
    auto flatBufferAdditionLayer = CreateActivationLayer(m_flatBufferBuilder,
                                                         flatBufferBaseLayer,
                                                         flatBufferDescriptor);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferAdditionLayer.o, serializer::Layer::Layer_ActivationLayer);
}

// Build FlatBuffer for Addition Layer
void SerializerVisitor::VisitAdditionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferAdditionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Addition);

    // Create the FlatBuffer AdditionLayer
    auto flatBufferAdditionLayer = serializer::CreateAdditionLayer(m_flatBufferBuilder, flatBufferAdditionBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferAdditionLayer.o, serializer::Layer::Layer_AdditionLayer);
}

// Build FlatBuffer for BatchToSpaceNd Layer
void SerializerVisitor::VisitBatchToSpaceNdLayer(const armnn::IConnectableLayer* layer,
                                                 const armnn::BatchToSpaceNdDescriptor& descriptor,
                                                 const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_BatchToSpaceNd);

    std::vector<unsigned int> crops;
    crops.reserve(descriptor.m_Crops.size() * 2);
    for (auto& crop : descriptor.m_Crops)
    {
        crops.push_back(crop.first);
        crops.push_back(crop.second);
    }

    auto flatBufferDescriptor =
        CreateBatchToSpaceNdDescriptor(m_flatBufferBuilder,
                                       m_flatBufferBuilder.CreateVector(descriptor.m_BlockShape),
                                       m_flatBufferBuilder.CreateVector(crops),
                                       GetFlatBufferDataLayout(descriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateBatchToSpaceNdLayer(m_flatBufferBuilder,
                                                                 flatBufferBaseLayer,
                                                                 flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_BatchToSpaceNdLayer);
}

void SerializerVisitor::VisitBatchNormalizationLayer(const armnn::IConnectableLayer* layer,
                                                     const armnn::BatchNormalizationDescriptor& batchNormDescriptor,
                                                     const armnn::ConstTensor& mean,
                                                     const armnn::ConstTensor& variance,
                                                     const armnn::ConstTensor& beta,
                                                     const armnn::ConstTensor& gamma,
                                                     const char* name)
{
    auto fbBatchNormalizationBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_BatchNormalization);
    auto fbBatchNormalizationDescriptor = serializer::CreateBatchNormalizationDescriptor(
                                                  m_flatBufferBuilder,
                                                  batchNormDescriptor.m_Eps,
                                                  GetFlatBufferDataLayout(batchNormDescriptor.m_DataLayout));

    auto fbMeanConstTensorInfo     = CreateConstTensorInfo(mean);
    auto fbVarianceConstTensorInfo = CreateConstTensorInfo(variance);
    auto fbBetaConstTensorInfo     = CreateConstTensorInfo(beta);
    auto fbGammaConstTensorInfo    = CreateConstTensorInfo(gamma);
    auto fbBatchNormalizationLayer = serializer::CreateBatchNormalizationLayer(m_flatBufferBuilder,
                                                                               fbBatchNormalizationBaseLayer,
                                                                               fbBatchNormalizationDescriptor,
                                                                               fbMeanConstTensorInfo,
                                                                               fbVarianceConstTensorInfo,
                                                                               fbBetaConstTensorInfo,
                                                                               fbGammaConstTensorInfo);

    CreateAnyLayer(fbBatchNormalizationLayer.o, serializer::Layer::Layer_BatchNormalizationLayer);
}

// Build FlatBuffer for Constant Layer
void SerializerVisitor::VisitConstantLayer(const armnn::IConnectableLayer* layer,
                                           const armnn::ConstTensor& input,
                                           const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferConstantBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Constant);

    auto flatBufferConstTensorInfo = CreateConstTensorInfo(input);

    // Create the FlatBuffer ConstantLayer
    auto flatBufferLayer = CreateConstantLayer(m_flatBufferBuilder,
                                               flatBufferConstantBaseLayer,
                                               flatBufferConstTensorInfo);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ConstantLayer);
}

// Build FlatBuffer for Convolution2dLayer
void SerializerVisitor::VisitConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                                const armnn::Convolution2dDescriptor& descriptor,
                                                const armnn::ConstTensor& weights,
                                                const armnn::Optional<armnn::ConstTensor>& biases,
                                                const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Convolution2d);

    auto flatBufferDescriptor = CreateConvolution2dDescriptor(m_flatBufferBuilder,
                                                              descriptor.m_PadLeft,
                                                              descriptor.m_PadRight,
                                                              descriptor.m_PadTop,
                                                              descriptor.m_PadBottom,
                                                              descriptor.m_StrideX,
                                                              descriptor.m_StrideY,
                                                              descriptor.m_DilationX,
                                                              descriptor.m_DilationY,
                                                              descriptor.m_BiasEnabled,
                                                              GetFlatBufferDataLayout(descriptor.m_DataLayout));
    auto flatBufferWeightsConstTensorInfo = CreateConstTensorInfo(weights);
    flatbuffers::Offset<serializer::ConstTensor> flatBufferBiasesConstTensorInfo;

    if (biases.has_value())
    {
        flatBufferBiasesConstTensorInfo = CreateConstTensorInfo(biases.value());
    }

    // Create the FlatBuffer Convolution2dLayer
    auto flatBufferLayer = CreateConvolution2dLayer(m_flatBufferBuilder,
                                                    flatBufferBaseLayer,
                                                    flatBufferDescriptor,
                                                    flatBufferWeightsConstTensorInfo,
                                                    flatBufferBiasesConstTensorInfo);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_Convolution2dLayer);
}

void SerializerVisitor::VisitDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                                         const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                                         const armnn::ConstTensor& weights,
                                                         const armnn::Optional<armnn::ConstTensor>& biases,
                                                         const char* name)
{
    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_DepthwiseConvolution2d);
    auto fbDescriptor = CreateDepthwiseConvolution2dDescriptor(m_flatBufferBuilder,
                                                               descriptor.m_PadLeft,
                                                               descriptor.m_PadRight,
                                                               descriptor.m_PadTop,
                                                               descriptor.m_PadBottom,
                                                               descriptor.m_StrideX,
                                                               descriptor.m_StrideY,
                                                               descriptor.m_DilationX,
                                                               descriptor.m_DilationY,
                                                               descriptor.m_BiasEnabled,
                                                               GetFlatBufferDataLayout(descriptor.m_DataLayout));

    flatbuffers::Offset<serializer::ConstTensor> fbWeightsConstTensorInfo = CreateConstTensorInfo(weights);
    flatbuffers::Offset<serializer::ConstTensor> fbBiasesConstTensorInfo;
    if (biases.has_value())
    {
        fbBiasesConstTensorInfo = CreateConstTensorInfo(biases.value());
    }

    auto flatBufferLayer = CreateDepthwiseConvolution2dLayer(m_flatBufferBuilder,
                                                             fbBaseLayer,
                                                             fbDescriptor,
                                                             fbWeightsConstTensorInfo,
                                                             fbBiasesConstTensorInfo);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_DepthwiseConvolution2dLayer);
}

void SerializerVisitor::VisitDequantizeLayer(const armnn::IConnectableLayer* layer,
                                             const char* name)
{
    auto fbDequantizeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Dequantize);
    auto fbDequantizeLayer     = serializer::CreateDequantizeLayer(m_flatBufferBuilder, fbDequantizeBaseLayer);

    CreateAnyLayer(fbDequantizeLayer.o, serializer::Layer::Layer_DequantizeLayer);
}

void SerializerVisitor::VisitDetectionPostProcessLayer(const armnn::IConnectableLayer* layer,
                                                       const armnn::DetectionPostProcessDescriptor& descriptor,
                                                       const armnn::ConstTensor& anchors,
                                                       const char* name)
{
    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_DetectionPostProcess);
    auto fbDescriptor = CreateDetectionPostProcessDescriptor(m_flatBufferBuilder,
                                                             descriptor.m_MaxDetections,
                                                             descriptor.m_MaxClassesPerDetection,
                                                             descriptor.m_DetectionsPerClass,
                                                             descriptor.m_NmsScoreThreshold,
                                                             descriptor.m_NmsIouThreshold,
                                                             descriptor.m_NumClasses,
                                                             descriptor.m_UseRegularNms,
                                                             descriptor.m_ScaleX,
                                                             descriptor.m_ScaleY,
                                                             descriptor.m_ScaleW,
                                                             descriptor.m_ScaleH);

    flatbuffers::Offset<serializer::ConstTensor> fbAnchorsConstTensorInfo = CreateConstTensorInfo(anchors);

    auto flatBufferLayer = CreateDetectionPostProcessLayer(m_flatBufferBuilder,
                                                           fbBaseLayer,
                                                           fbDescriptor,
                                                           fbAnchorsConstTensorInfo);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_DetectionPostProcessLayer);
}

void SerializerVisitor::VisitDivisionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbDivisionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Division);
    auto fbDivisionLayer     = serializer::CreateDivisionLayer(m_flatBufferBuilder, fbDivisionBaseLayer);

    CreateAnyLayer(fbDivisionLayer.o, serializer::Layer::Layer_DivisionLayer);
}

void SerializerVisitor::VisitEqualLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Equal);
    auto fbEqualLayer = serializer::CreateEqualLayer(m_flatBufferBuilder, fbBaseLayer);

    CreateAnyLayer(fbEqualLayer.o, serializer::Layer::Layer_EqualLayer);
}

void SerializerVisitor::VisitFloorLayer(const armnn::IConnectableLayer *layer, const char *name)
{
    auto flatBufferFloorBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Floor);
    auto flatBufferFloorLayer = serializer::CreateFloorLayer(m_flatBufferBuilder, flatBufferFloorBaseLayer);

    CreateAnyLayer(flatBufferFloorLayer.o, serializer::Layer::Layer_FloorLayer);
}

void SerializerVisitor::VisitGatherLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbGatherBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Gather);
    auto flatBufferLayer   = serializer::CreateGatherLayer(m_flatBufferBuilder, fbGatherBaseLayer);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_GatherLayer);
}

void SerializerVisitor::VisitGreaterLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbGreaterBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Greater);
    auto fbGreaterLayer = serializer::CreateGreaterLayer(m_flatBufferBuilder, fbGreaterBaseLayer);

    CreateAnyLayer(fbGreaterLayer.o, serializer::Layer::Layer_GreaterLayer);
}

void SerializerVisitor::VisitL2NormalizationLayer(const armnn::IConnectableLayer* layer,
                                                  const armnn::L2NormalizationDescriptor& l2NormalizationDescriptor,
                                                  const char* name)
{
    // Create FlatBuffer BaseLayer
    auto fbBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_L2Normalization);

    // Create the FlatBuffer L2Normalization Descriptor
    auto fbDescriptor = serializer::CreateL2NormalizationDescriptor(
            m_flatBufferBuilder,
            GetFlatBufferDataLayout(l2NormalizationDescriptor.m_DataLayout),
            l2NormalizationDescriptor.m_Eps);

    // Create FlatBuffer layer
    auto fbLayer = serializer::CreateL2NormalizationLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_L2NormalizationLayer);
}

void SerializerVisitor::VisitLstmLayer(const armnn::IConnectableLayer* layer, const armnn::LstmDescriptor& descriptor,
                                       const armnn::LstmInputParams& params, const char* name)
{
    auto fbLstmBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Lstm);

    auto fbLstmDescriptor = serializer::CreateLstmDescriptor(
        m_flatBufferBuilder,
        descriptor.m_ActivationFunc,
        descriptor.m_ClippingThresCell,
        descriptor.m_ClippingThresProj,
        descriptor.m_CifgEnabled,
        descriptor.m_PeepholeEnabled,
        descriptor.m_ProjectionEnabled,
        descriptor.m_LayerNormEnabled);

    // Get mandatory input parameters
    auto inputToForgetWeights = CreateConstTensorInfo(*params.m_InputToForgetWeights);
    auto inputToCellWeights = CreateConstTensorInfo(*params.m_InputToCellWeights);
    auto inputToOutputWeights = CreateConstTensorInfo(*params.m_InputToOutputWeights);
    auto recurrentToForgetWeights = CreateConstTensorInfo(*params.m_RecurrentToForgetWeights);
    auto recurrentToCellWeights = CreateConstTensorInfo(*params.m_RecurrentToCellWeights);
    auto recurrentToOutputWeights = CreateConstTensorInfo(*params.m_RecurrentToOutputWeights);
    auto forgetGateBias = CreateConstTensorInfo(*params.m_ForgetGateBias);
    auto cellBias = CreateConstTensorInfo(*params.m_CellBias);
    auto outputGateBias = CreateConstTensorInfo(*params.m_OutputGateBias);

    //Define optional parameters, these will be set depending on configuration in Lstm descriptor
    flatbuffers::Offset<serializer::ConstTensor> inputToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> recurrentToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> inputGateBias;
    flatbuffers::Offset<serializer::ConstTensor> projectionWeights;
    flatbuffers::Offset<serializer::ConstTensor> projectionBias;
    flatbuffers::Offset<serializer::ConstTensor> cellToForgetWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellToOutputWeights;
    flatbuffers::Offset<serializer::ConstTensor> inputLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> forgetLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> outputLayerNormWeights;

    if (!descriptor.m_CifgEnabled)
    {
        inputToInputWeights = CreateConstTensorInfo(*params.m_InputToInputWeights);
        recurrentToInputWeights = CreateConstTensorInfo(*params.m_RecurrentToInputWeights);
        cellToInputWeights = CreateConstTensorInfo(*params.m_CellToInputWeights);
        inputGateBias = CreateConstTensorInfo(*params.m_InputGateBias);
    }

    if (descriptor.m_ProjectionEnabled)
    {
        projectionWeights = CreateConstTensorInfo(*params.m_ProjectionWeights);
        projectionBias = CreateConstTensorInfo(*params.m_ProjectionBias);
    }

    if (descriptor.m_PeepholeEnabled)
    {
        cellToForgetWeights = CreateConstTensorInfo(*params.m_CellToForgetWeights);
        cellToOutputWeights = CreateConstTensorInfo(*params.m_CellToOutputWeights);
    }

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            inputLayerNormWeights = CreateConstTensorInfo((*params.m_InputLayerNormWeights));
        }
        forgetLayerNormWeights = CreateConstTensorInfo(*params.m_ForgetLayerNormWeights);
        cellLayerNormWeights   = CreateConstTensorInfo(*params.m_CellLayerNormWeights);
        outputLayerNormWeights = CreateConstTensorInfo(*params.m_OutputLayerNormWeights);
    }

    auto fbLstmParams = serializer::CreateLstmInputParams(
        m_flatBufferBuilder,
        inputToForgetWeights,
        inputToCellWeights,
        inputToOutputWeights,
        recurrentToForgetWeights,
        recurrentToCellWeights,
        recurrentToOutputWeights,
        forgetGateBias,
        cellBias,
        outputGateBias,
        inputToInputWeights,
        recurrentToInputWeights,
        cellToInputWeights,
        inputGateBias,
        projectionWeights,
        projectionBias,
        cellToForgetWeights,
        cellToOutputWeights,
        inputLayerNormWeights,
        forgetLayerNormWeights,
        cellLayerNormWeights,
        outputLayerNormWeights);

    auto fbLstmLayer = serializer::CreateLstmLayer(
        m_flatBufferBuilder,
        fbLstmBaseLayer,
        fbLstmDescriptor,
        fbLstmParams);

    CreateAnyLayer(fbLstmLayer.o, serializer::Layer::Layer_LstmLayer);
}

void SerializerVisitor::VisitMaximumLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbMaximumBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Maximum);
    auto fbMaximumLayer     = serializer::CreateMaximumLayer(m_flatBufferBuilder, fbMaximumBaseLayer);

    CreateAnyLayer(fbMaximumLayer.o, serializer::Layer::Layer_MaximumLayer);
}

void SerializerVisitor::VisitMeanLayer(const armnn::IConnectableLayer* layer,
                                       const armnn::MeanDescriptor& descriptor,
                                       const char* name)
{
    auto fbMeanBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Mean);
    auto fbMeanDescriptor = serializer::CreateMeanDescriptor(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(descriptor.m_Axis),
                                                             descriptor.m_KeepDims);

    auto fbMeanLayer = serializer::CreateMeanLayer(m_flatBufferBuilder,
                                                   fbMeanBaseLayer,
                                                   fbMeanDescriptor);

    CreateAnyLayer(fbMeanLayer.o, serializer::Layer::Layer_MeanLayer);
}

void SerializerVisitor::VisitMinimumLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbMinimumBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Minimum);
    auto fbMinimumLayer     = serializer::CreateMinimumLayer(m_flatBufferBuilder, fbMinimumBaseLayer);

    CreateAnyLayer(fbMinimumLayer.o, serializer::Layer::Layer_MinimumLayer);
}

void SerializerVisitor::VisitMergeLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbMergeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Merge);
    auto fbMergeLayer     = serializer::CreateMergeLayer(m_flatBufferBuilder, fbMergeBaseLayer);

    CreateAnyLayer(fbMergeLayer.o, serializer::Layer::Layer_MergeLayer);
}

void SerializerVisitor::VisitMergerLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::MergerDescriptor& mergerDescriptor,
                                         const char* name)
{
    VisitConcatLayer(layer, mergerDescriptor, name);
}

void SerializerVisitor::VisitConcatLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::ConcatDescriptor& concatDescriptor,
                                         const char* name)
{
    auto flatBufferConcatBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Concat);

    std::vector<flatbuffers::Offset<UintVector>> views;
    for (unsigned int v = 0; v < concatDescriptor.GetNumViews(); ++v)
    {
        const uint32_t* origin = concatDescriptor.GetViewOrigin(v);
        std::vector<uint32_t> origins;
        for (unsigned int d = 0; d < concatDescriptor.GetNumDimensions(); ++d)
        {
            origins.push_back(origin[d]);
        }
        auto view = m_flatBufferBuilder.CreateVector(origins);
        auto uintVector = CreateUintVector(m_flatBufferBuilder, view);
        views.push_back(uintVector);
    }

    auto flatBufferConcatDescriptor = CreateOriginsDescriptor(m_flatBufferBuilder,
                                                              concatDescriptor.GetConcatAxis(),
                                                              concatDescriptor.GetNumViews(),
                                                              concatDescriptor.GetNumDimensions(),
                                                              m_flatBufferBuilder.CreateVector(views));

    auto flatBufferLayer = CreateConcatLayer(m_flatBufferBuilder,
                                             flatBufferConcatBaseLayer,
                                             flatBufferConcatDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ConcatLayer);
}

void SerializerVisitor::VisitMultiplicationLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbMultiplicationBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Multiplication);
    auto fbMultiplicationLayer     = serializer::CreateMultiplicationLayer(m_flatBufferBuilder,
                                                                           fbMultiplicationBaseLayer);

    CreateAnyLayer(fbMultiplicationLayer.o, serializer::Layer::Layer_MultiplicationLayer);
}

void SerializerVisitor::VisitPadLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::PadDescriptor& padDescriptor,
                                      const char* name)
{
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Pad);

    std::vector<unsigned int> padList;
    for (auto& p: padDescriptor.m_PadList)
    {
        padList.push_back(p.first);
        padList.push_back(p.second);
    }

    auto flatBufferPadDesc = serializer::CreatePadDescriptor(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(padList),
                                                             padDescriptor.m_PadValue);

    auto flatBufferPadLayer = serializer::CreatePadLayer(m_flatBufferBuilder,
                                                         flatBufferBaseLayer,
                                                         flatBufferPadDesc);

    CreateAnyLayer(flatBufferPadLayer.o, serializer::Layer::Layer_PadLayer);
}

void SerializerVisitor::VisitPermuteLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::PermuteDescriptor& permuteDescriptor,
                                          const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferPermuteBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Permute);

    std::vector<unsigned int> dimMappings;
    for (unsigned int i=0; i<permuteDescriptor.m_DimMappings.GetSize(); ++i)
    {
        dimMappings.push_back(permuteDescriptor.m_DimMappings[i]);
    }

    auto flatBufferPermuteDesc = serializer::CreatePermuteDescriptor(m_flatBufferBuilder,
                                                                     m_flatBufferBuilder.CreateVector(dimMappings));

    // Create the FlatBuffer PermuteLayer
    auto flatBufferPermuteLayer = serializer::CreatePermuteLayer(m_flatBufferBuilder,
                                                                 flatBufferPermuteBaseLayer,
                                                                 flatBufferPermuteDesc);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferPermuteLayer.o, serializer::Layer::Layer_PermuteLayer);
}

// Build FlatBuffer for Reshape Layer
void SerializerVisitor::VisitReshapeLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::ReshapeDescriptor& reshapeDescriptor,
                                          const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferReshapeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Reshape);

    std::vector<unsigned int> targetShape;
    for (unsigned int i =0; i < reshapeDescriptor.m_TargetShape.GetNumDimensions(); i++)
    {
        targetShape.push_back(reshapeDescriptor.m_TargetShape[i]);
    }

    auto flatBufferReshapeDesc = serializer::CreateReshapeDescriptor(m_flatBufferBuilder,
                                                                     m_flatBufferBuilder.CreateVector(targetShape));

    // Create the FlatBuffer ReshapeLayer
    auto flatBufferReshapeLayer = serializer::CreateReshapeLayer(m_flatBufferBuilder, flatBufferReshapeBaseLayer,
                                                                 flatBufferReshapeDesc);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferReshapeLayer.o, serializer::Layer::Layer_ReshapeLayer);
}

void SerializerVisitor::VisitResizeBilinearLayer(const armnn::IConnectableLayer* layer,
                                                 const armnn::ResizeBilinearDescriptor& resizeDescriptor,
                                                 const char* name)
{
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_ResizeBilinear);

    auto flatBufferDescriptor =
        CreateResizeBilinearDescriptor(m_flatBufferBuilder,
                                       resizeDescriptor.m_TargetWidth,
                                       resizeDescriptor.m_TargetHeight,
                                       GetFlatBufferDataLayout(resizeDescriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateResizeBilinearLayer(m_flatBufferBuilder,
                                                                 flatBufferBaseLayer,
                                                                 flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ResizeBilinearLayer);
}

void SerializerVisitor::VisitResizeLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::ResizeDescriptor& resizeDescriptor,
                                         const char* name)
{
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Resize);

    auto flatBufferDescriptor =
            CreateResizeDescriptor(m_flatBufferBuilder,
                                   resizeDescriptor.m_TargetHeight,
                                   resizeDescriptor.m_TargetWidth,
                                   GetFlatBufferResizeMethod(resizeDescriptor.m_Method),
                                   GetFlatBufferDataLayout(resizeDescriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateResizeLayer(m_flatBufferBuilder,
                                                         flatBufferBaseLayer,
                                                         flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ResizeLayer);
}

void SerializerVisitor::VisitRsqrtLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbRsqrtBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Rsqrt);
    auto fbRsqrtLayer     = serializer::CreateRsqrtLayer(m_flatBufferBuilder, fbRsqrtBaseLayer);

    CreateAnyLayer(fbRsqrtLayer.o, serializer::Layer::Layer_RsqrtLayer);
}

// Build FlatBuffer for Softmax Layer
void SerializerVisitor::VisitSoftmaxLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::SoftmaxDescriptor& softmaxDescriptor,
                                          const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferSoftmaxBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Softmax);

    // Create the FlatBuffer SoftmaxDescriptor
    auto flatBufferSoftmaxDesc =
        serializer::CreateSoftmaxDescriptor(m_flatBufferBuilder, softmaxDescriptor.m_Beta);

    // Create the FlatBuffer SoftmaxLayer
    auto flatBufferSoftmaxLayer =
        serializer::CreateSoftmaxLayer(m_flatBufferBuilder,
                                       flatBufferSoftmaxBaseLayer,
                                       flatBufferSoftmaxDesc);

    CreateAnyLayer(flatBufferSoftmaxLayer.o, serializer::Layer::Layer_SoftmaxLayer);
}

void SerializerVisitor::VisitPooling2dLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::Pooling2dDescriptor& pooling2dDescriptor,
                                            const char* name)
{
    auto fbPooling2dBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Pooling2d);
    auto fbPooling2dDescriptor = serializer::CreatePooling2dDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferPoolingAlgorithm(pooling2dDescriptor.m_PoolType),
        pooling2dDescriptor.m_PadLeft,
        pooling2dDescriptor.m_PadRight,
        pooling2dDescriptor.m_PadTop,
        pooling2dDescriptor.m_PadBottom,
        pooling2dDescriptor.m_PoolWidth,
        pooling2dDescriptor.m_PoolHeight,
        pooling2dDescriptor.m_StrideX,
        pooling2dDescriptor.m_StrideY,
        GetFlatBufferOutputShapeRounding(pooling2dDescriptor.m_OutputShapeRounding),
        GetFlatBufferPaddingMethod(pooling2dDescriptor.m_PaddingMethod),
        GetFlatBufferDataLayout(pooling2dDescriptor.m_DataLayout));

    auto fbPooling2dLayer = serializer::CreatePooling2dLayer(m_flatBufferBuilder,
                                                             fbPooling2dBaseLayer,
                                                             fbPooling2dDescriptor);

    CreateAnyLayer(fbPooling2dLayer.o, serializer::Layer::Layer_Pooling2dLayer);
}

void SerializerVisitor::VisitPreluLayer(const armnn::IConnectableLayer* layer,
                                        const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferPreluBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Prelu);

    // Create the FlatBuffer AdditionLayer
    auto flatBufferPreluLayer = serializer::CreatePreluLayer(m_flatBufferBuilder, flatBufferPreluBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferPreluLayer.o, serializer::Layer::Layer_PreluLayer);
}

void SerializerVisitor::VisitQuantizeLayer(const armnn::IConnectableLayer *layer, const char *name)
{
    auto fbQuantizeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Quantize);
    auto fbQuantizeLayer = serializer::CreateQuantizeLayer(m_flatBufferBuilder,
                                                           fbQuantizeBaseLayer);
    CreateAnyLayer(fbQuantizeLayer.o, serializer::Layer::Layer_QuantizeLayer);
}

// Build FlatBuffer for FullyConnected Layer
void SerializerVisitor::VisitFullyConnectedLayer(const armnn::IConnectableLayer* layer,
                                                 const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                 const armnn::ConstTensor& weights,
                                                 const armnn::Optional<armnn::ConstTensor>& biases,
                                                 const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_FullyConnected);

    // Create FlatBuffer FullyConnectedDescriptor
    auto flatBufferDescriptor =
        serializer::CreateFullyConnectedDescriptor(m_flatBufferBuilder,
                                                   fullyConnectedDescriptor.m_BiasEnabled,
                                                   fullyConnectedDescriptor.m_TransposeWeightMatrix);

    // Create FlatBuffer weights data
    auto flatBufferWeights = CreateConstTensorInfo(weights);

    // Create FlatBuffer bias data
    flatbuffers::Offset<serializer::ConstTensor> flatBufferBiases;
    if (fullyConnectedDescriptor.m_BiasEnabled)
    {
        flatBufferBiases = CreateConstTensorInfo(biases.value());
    }

    // Create FlatBuffer FullyConnectedLayer
    auto flatBufferLayer = serializer::CreateFullyConnectedLayer(m_flatBufferBuilder,
                                                                 flatBufferBaseLayer,
                                                                 flatBufferDescriptor,
                                                                 flatBufferWeights,
                                                                 flatBufferBiases);

    // Add created FullyConnectedLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_FullyConnectedLayer);
}

// Build FlatBuffer for SpaceToBatchNd Layer
void SerializerVisitor::VisitSpaceToBatchNdLayer(const armnn::IConnectableLayer* layer,
                                                 const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                 const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_SpaceToBatchNd);

    std::vector<unsigned int> padList;
    padList.reserve(spaceToBatchNdDescriptor.m_PadList.size()*2);
    for (auto& pad : spaceToBatchNdDescriptor.m_PadList)
    {
        padList.push_back(pad.first);
        padList.push_back(pad.second);
    }

    auto flatBufferDescriptor =
        CreateSpaceToBatchNdDescriptor(m_flatBufferBuilder,
                                       m_flatBufferBuilder.CreateVector(spaceToBatchNdDescriptor.m_BlockShape),
                                       m_flatBufferBuilder.CreateVector(padList),
                                       GetFlatBufferDataLayout(spaceToBatchNdDescriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateSpaceToBatchNdLayer(m_flatBufferBuilder,
                                                                 flatBufferBaseLayer,
                                                                 flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_SpaceToBatchNdLayer);
}

// Build FlatBuffer for SpaceToDepthLayer
void SerializerVisitor::VisitSpaceToDepthLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                               const char* name)
{
    auto flatBufferBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_SpaceToDepth);
    auto flatBufferDescriptor =
        CreateSpaceToDepthDescriptor(m_flatBufferBuilder,
                                     spaceToDepthDescriptor.m_BlockSize,
                                     GetFlatBufferDataLayout(spaceToDepthDescriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateSpaceToDepthLayer(m_flatBufferBuilder,
                                                               flatBufferBaseLayer,
                                                               flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_SpaceToDepthLayer);
}

// Build FlatBuffer for Splitter Layer
void SerializerVisitor::VisitSplitterLayer(const armnn::IConnectableLayer* layer,
                                           const armnn::ViewsDescriptor& viewsDescriptor,
                                           const char* name)
{
    // Create FlatBuffer ViewOrigins
    std::vector<flatbuffers::Offset<UintVector>> flatBufferViewOrigins;
    flatBufferViewOrigins.reserve(viewsDescriptor.GetNumViews());

    for(unsigned int vIdx = 0; vIdx < viewsDescriptor.GetNumViews(); ++vIdx)
    {
        std::vector<uint32_t> viewOrigin;
        viewOrigin.reserve(viewsDescriptor.GetNumDimensions());

        // Copy vector
        for(unsigned int dIdx = 0; dIdx < viewsDescriptor.GetNumDimensions(); ++dIdx)
        {
            viewOrigin.push_back(viewsDescriptor.GetViewOrigin(vIdx)[dIdx]);
        }

        flatBufferViewOrigins.push_back(CreateUintVector(m_flatBufferBuilder,
                                                         m_flatBufferBuilder.CreateVector(viewOrigin)));
    }

    // Create FlatBuffer OriginsDescriptor
    auto flatBufferOriginDescriptor = CreateOriginsDescriptor(m_flatBufferBuilder,
                                                              viewsDescriptor.GetOrigins().GetConcatAxis(),
                                                              viewsDescriptor.GetOrigins().GetNumViews(),
                                                              viewsDescriptor.GetOrigins().GetNumDimensions(),
                                                              m_flatBufferBuilder.CreateVector(flatBufferViewOrigins));

    // Create FlatBuffer ViewOrigins
    std::vector<flatbuffers::Offset<UintVector>> flatBufferViewSizes;
    flatBufferViewSizes.reserve(viewsDescriptor.GetNumViews());

    for(unsigned int vIdx = 0; vIdx < viewsDescriptor.GetNumViews(); ++vIdx)
    {
        std::vector<uint32_t> viewSize;
        viewSize.reserve(viewsDescriptor.GetNumDimensions());

        // Copy vector
        for(unsigned int dIdx = 0; dIdx < viewsDescriptor.GetNumDimensions(); ++dIdx)
        {
            viewSize.push_back(viewsDescriptor.GetViewSizes(vIdx)[dIdx]);
        }

        flatBufferViewSizes.push_back(CreateUintVector(m_flatBufferBuilder,
                                                       m_flatBufferBuilder.CreateVector(viewSize)));
    }

    // Create FlatBuffer ViewsDescriptor
    auto flatBufferViewsDescriptor = CreateViewsDescriptor(m_flatBufferBuilder,
                                                           flatBufferOriginDescriptor,
                                                           m_flatBufferBuilder.CreateVector(flatBufferViewSizes));

    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Splitter);

    auto flatBufferSplitterLayer = serializer::CreateSplitterLayer(m_flatBufferBuilder,
                                                                   flatBufferBaseLayer,
                                                                   flatBufferViewsDescriptor);

    CreateAnyLayer(flatBufferSplitterLayer.o, serializer::Layer::Layer_SplitterLayer);
}

void SerializerVisitor::VisitNormalizationLayer(const armnn::IConnectableLayer* layer,
                                                const armnn::NormalizationDescriptor& descriptor,
                                                const char* name)
{
    auto fbNormalizationBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Normalization);

    auto fbNormalizationDescriptor = serializer::CreateNormalizationDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferNormalizationAlgorithmChannel(descriptor.m_NormChannelType),
        GetFlatBufferNormalizationAlgorithmMethod(descriptor.m_NormMethodType),
        descriptor.m_NormSize,
        descriptor.m_Alpha,
        descriptor.m_Beta,
        descriptor.m_K,
        GetFlatBufferDataLayout(descriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateNormalizationLayer(m_flatBufferBuilder,
                                                                fbNormalizationBaseLayer,
                                                                fbNormalizationDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_NormalizationLayer);
}

void SerializerVisitor::VisitStackLayer(const armnn::IConnectableLayer* layer,
                                        const armnn::StackDescriptor& stackDescriptor,
                                        const char* name)
{
    auto stackBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Stack);

    std::vector<unsigned int> inputShape;
    for (unsigned int i =0; i < stackDescriptor.m_InputShape.GetNumDimensions(); i++)
    {
        inputShape.push_back(stackDescriptor.m_InputShape[i]);
    }

    auto flatBufferStackDescriptor = CreateStackDescriptor(m_flatBufferBuilder,
                                                           stackDescriptor.m_Axis,
                                                           stackDescriptor.m_NumInputs,
                                                           m_flatBufferBuilder.CreateVector(inputShape));

    auto stackLayer = serializer::CreateStackLayer(m_flatBufferBuilder, stackBaseLayer, flatBufferStackDescriptor);
    CreateAnyLayer(stackLayer.o, serializer::Layer::Layer_StackLayer);
}

void SerializerVisitor::VisitStridedSliceLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::StridedSliceDescriptor& stridedSliceDescriptor,
                                               const char* name)
{
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_StridedSlice);

    auto flatBufferDescriptor =
        CreateStridedSliceDescriptor(m_flatBufferBuilder,
                                     m_flatBufferBuilder.CreateVector(stridedSliceDescriptor.m_Begin),
                                     m_flatBufferBuilder.CreateVector(stridedSliceDescriptor.m_End),
                                     m_flatBufferBuilder.CreateVector(stridedSliceDescriptor.m_Stride),
                                     stridedSliceDescriptor.m_BeginMask,
                                     stridedSliceDescriptor.m_EndMask,
                                     stridedSliceDescriptor.m_ShrinkAxisMask,
                                     stridedSliceDescriptor.m_EllipsisMask,
                                     stridedSliceDescriptor.m_NewAxisMask,
                                     GetFlatBufferDataLayout(stridedSliceDescriptor.m_DataLayout));

    auto flatBufferLayer = serializer::CreateStridedSliceLayer(m_flatBufferBuilder,
                                                               flatBufferBaseLayer,
                                                               flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_StridedSliceLayer);
}

void SerializerVisitor::VisitSubtractionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbSubtractionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Subtraction);
    auto fbSubtractionLayer = serializer::CreateSubtractionLayer(m_flatBufferBuilder, fbSubtractionBaseLayer);

    CreateAnyLayer(fbSubtractionLayer.o, serializer::Layer::Layer_SubtractionLayer);
}

void SerializerVisitor::VisitSwitchLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    auto fbSwitchBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Switch);
    auto fbSwitchLayer = serializer::CreateSwitchLayer(m_flatBufferBuilder, fbSwitchBaseLayer);

    CreateAnyLayer(fbSwitchLayer.o, serializer::Layer::Layer_SwitchLayer);
}

void SerializerVisitor::VisitTransposeConvolution2dLayer(
    const armnn::IConnectableLayer* layer,
    const armnn::TransposeConvolution2dDescriptor& descriptor,
    const armnn::ConstTensor& weights,
    const armnn::Optional<armnn::ConstTensor>& biases,
    const char* name)
{
    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Convolution2d);
    auto fbDescriptor = CreateTransposeConvolution2dDescriptor(m_flatBufferBuilder,
                                                               descriptor.m_PadLeft,
                                                               descriptor.m_PadRight,
                                                               descriptor.m_PadTop,
                                                               descriptor.m_PadBottom,
                                                               descriptor.m_StrideX,
                                                               descriptor.m_StrideY,
                                                               descriptor.m_BiasEnabled,
                                                               GetFlatBufferDataLayout(descriptor.m_DataLayout));

    // weights & biases
    auto fbWeightsConstTensorInfo = CreateConstTensorInfo(weights);
    flatbuffers::Offset<serializer::ConstTensor> fbBiasesConstTensorInfo;
    if (biases.has_value())
    {
        fbBiasesConstTensorInfo = CreateConstTensorInfo(biases.value());
    }

    auto fbLayer = CreateTransposeConvolution2dLayer(m_flatBufferBuilder,
                                                     fbBaseLayer,
                                                     fbDescriptor,
                                                     fbWeightsConstTensorInfo,
                                                     fbBiasesConstTensorInfo);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_TransposeConvolution2dLayer);
}

void SerializerVisitor::VisitQuantizedLstmLayer(const armnn::IConnectableLayer* layer,
                                                const armnn::QuantizedLstmInputParams& params,
                                                const char* name)
{
    auto fbQuantizedLstmBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_QuantizedLstm);

    // Get input parameters
    auto inputToInputWeights = CreateConstTensorInfo(params.GetInputToInputWeights());
    auto inputToForgetWeights = CreateConstTensorInfo(params.GetInputToForgetWeights());
    auto inputToCellWeights = CreateConstTensorInfo(params.GetInputToCellWeights());
    auto inputToOutputWeights = CreateConstTensorInfo(params.GetInputToOutputWeights());

    auto recurrentToInputWeights = CreateConstTensorInfo(params.GetRecurrentToInputWeights());
    auto recurrentToForgetWeights = CreateConstTensorInfo(params.GetRecurrentToForgetWeights());
    auto recurrentToCellWeights = CreateConstTensorInfo(params.GetRecurrentToCellWeights());
    auto recurrentToOutputWeights = CreateConstTensorInfo(params.GetRecurrentToOutputWeights());

    auto inputGateBias = CreateConstTensorInfo(params.GetInputGateBias());
    auto forgetGateBias = CreateConstTensorInfo(params.GetForgetGateBias());
    auto cellBias = CreateConstTensorInfo(params.GetCellBias());
    auto outputGateBias = CreateConstTensorInfo(params.GetOutputGateBias());

    auto fbQuantizedLstmParams = serializer::CreateQuantizedLstmInputParams(
        m_flatBufferBuilder,
        inputToInputWeights,
        inputToForgetWeights,
        inputToCellWeights,
        inputToOutputWeights,
        recurrentToInputWeights,
        recurrentToForgetWeights,
        recurrentToCellWeights,
        recurrentToOutputWeights,
        inputGateBias,
        forgetGateBias,
        cellBias,
        outputGateBias);

    auto fbQuantizedLstmLayer = serializer::CreateQuantizedLstmLayer(
        m_flatBufferBuilder,
        fbQuantizedLstmBaseLayer,
        fbQuantizedLstmParams);

    CreateAnyLayer(fbQuantizedLstmLayer.o, serializer::Layer::Layer_QuantizedLstmLayer);
}

fb::Offset<serializer::LayerBase> SerializerVisitor::CreateLayerBase(const IConnectableLayer* layer,
                                                                     const serializer::LayerType layerType)
{
    uint32_t fbIndex = GetSerializedId(layer->GetGuid());

    std::vector<fb::Offset<serializer::InputSlot>> inputSlots = CreateInputSlots(layer);
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots = CreateOutputSlots(layer);

    return serializer::CreateLayerBase(m_flatBufferBuilder,
                                       fbIndex,
                                       m_flatBufferBuilder.CreateString(layer->GetName()),
                                       layerType,
                                       m_flatBufferBuilder.CreateVector(inputSlots),
                                       m_flatBufferBuilder.CreateVector(outputSlots));
}

void SerializerVisitor::CreateAnyLayer(const flatbuffers::Offset<void>& layer, const serializer::Layer serializerLayer)
{
    auto anyLayer = armnnSerializer::CreateAnyLayer(m_flatBufferBuilder, serializerLayer, layer);
    m_serializedLayers.push_back(anyLayer);
}

template <typename T>
flatbuffers::Offset<flatbuffers::Vector<T>> SerializerVisitor::CreateDataVector(const void* memory, unsigned int size)
{
    const T* buffer = reinterpret_cast<const T*>(memory);
    std::vector<T> vector(buffer, buffer + (size / sizeof(T)));
    auto fbVector = m_flatBufferBuilder.CreateVector(vector);
    return fbVector;
}

flatbuffers::Offset<serializer::ConstTensor>
    SerializerVisitor::CreateConstTensorInfo(const armnn::ConstTensor& constTensor)
{
    armnn::TensorInfo tensorInfo = constTensor.GetInfo();

    // Get the dimensions
    std::vector<unsigned int> shape;

    for(unsigned int dim = 0; dim < tensorInfo.GetShape().GetNumDimensions(); ++dim)
    {
        shape.push_back(tensorInfo.GetShape()[dim]);
    }

    // Create FlatBuffer TensorInfo
    auto flatBufferTensorInfo = serializer::CreateTensorInfo(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(shape),
                                                             GetFlatBufferDataType(tensorInfo.GetDataType()),
                                                             tensorInfo.GetQuantizationScale(),
                                                             tensorInfo.GetQuantizationOffset());
    flatbuffers::Offset<void> fbPayload;

    switch (tensorInfo.GetDataType())
    {
        case armnn::DataType::Float32:
        case armnn::DataType::Signed32:
        {
            auto fbVector = CreateDataVector<int32_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::IntData> flatBuffersData = serializer::CreateIntData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
            break;
        }
        case armnn::DataType::Float16:
        {
            auto fbVector = CreateDataVector<int16_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::ShortData> flatBuffersData = serializer::CreateShortData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
            break;
        }
        case armnn::DataType::QuantisedSymm16:
        {
            auto fbVector = CreateDataVector<int16_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::ShortData> flatBuffersData = serializer::CreateShortData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
            break;
        }
        case armnn::DataType::QuantisedAsymm8:
        case armnn::DataType::Boolean:
        default:
        {
            auto fbVector = CreateDataVector<int8_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::ByteData> flatBuffersData = serializer::CreateByteData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
        }
    }
    flatbuffers::Offset<serializer::ConstTensor> flatBufferConstTensor = serializer::CreateConstTensor(
            m_flatBufferBuilder,
            flatBufferTensorInfo,
            GetFlatBufferConstTensorData(tensorInfo.GetDataType()),
            fbPayload);
    return flatBufferConstTensor;
}

std::vector<fb::Offset<serializer::InputSlot>>
    SerializerVisitor::CreateInputSlots(const armnn::IConnectableLayer* layer)
{
    std::vector<fb::Offset<serializer::InputSlot>> inputSlots;

    // Get the InputSlots
    for (unsigned int slotIndex = 0; slotIndex<layer->GetNumInputSlots(); ++slotIndex)
    {
        const IInputSlot& inputSlot = layer->GetInputSlot(slotIndex);

        // Get the Connection for the InputSlot
        const IOutputSlot* connection = inputSlot.GetConnection();

        // Create FlatBuffer Connection
        serializer::Connection conn(GetSerializedId(inputSlot.GetConnection()->GetOwningLayerGuid()),
                                    connection->CalculateIndexOnOwner());
        // Create FlatBuffer InputSlot
        inputSlots.push_back(serializer::CreateInputSlot(m_flatBufferBuilder, slotIndex, &conn));
    }
    return inputSlots;
}

std::vector<fb::Offset<serializer::OutputSlot>>
    SerializerVisitor::CreateOutputSlots(const armnn::IConnectableLayer* layer)
{
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots;

    // Get the OutputSlots
    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        const IOutputSlot& outputSlot = layer->GetOutputSlot(slotIndex);
        const armnn::TensorInfo& tensorInfo = outputSlot.GetTensorInfo();

        // Get the dimensions
        std::vector<unsigned int> shape;
        for(unsigned int dim = 0; dim < tensorInfo.GetShape().GetNumDimensions(); ++dim)
        {
            shape.push_back(tensorInfo.GetShape()[dim]);
        }

        // Create FlatBuffer TensorInfo
        auto flatBufferTensorInfo = serializer::CreateTensorInfo(m_flatBufferBuilder,
                                                                 m_flatBufferBuilder.CreateVector(shape),
                                                                 GetFlatBufferDataType(tensorInfo.GetDataType()),
                                                                 tensorInfo.GetQuantizationScale(),
                                                                 tensorInfo.GetQuantizationOffset());

        // Create FlatBuffer Outputslot
        outputSlots.push_back(serializer::CreateOutputSlot(m_flatBufferBuilder,
                                                           slotIndex,
                                                           flatBufferTensorInfo));
    }
    return outputSlots;
}


ISerializer* ISerializer::CreateRaw()
{
    return new Serializer();
}

ISerializerPtr ISerializer::Create()
{
    return ISerializerPtr(CreateRaw(), &ISerializer::Destroy);
}

void ISerializer::Destroy(ISerializer* serializer)
{
    delete serializer;
}

void Serializer::Serialize(const INetwork& inNetwork)
{
    // Iterate through to network
    inNetwork.Accept(m_SerializerVisitor);
    flatbuffers::FlatBufferBuilder& fbBuilder = m_SerializerVisitor.GetFlatBufferBuilder();

    // Create FlatBuffer SerializedGraph
    auto serializedGraph = serializer::CreateSerializedGraph(
        fbBuilder,
        fbBuilder.CreateVector(m_SerializerVisitor.GetSerializedLayers()),
        fbBuilder.CreateVector(m_SerializerVisitor.GetInputIds()),
        fbBuilder.CreateVector(m_SerializerVisitor.GetOutputIds()));

    // Serialize the graph
    fbBuilder.Finish(serializedGraph);
}

bool Serializer::SaveSerializedToStream(std::ostream& stream)
{
    flatbuffers::FlatBufferBuilder& fbBuilder = m_SerializerVisitor.GetFlatBufferBuilder();

    auto bytesToWrite = boost::numeric_cast<std::streamsize>(fbBuilder.GetSize());
    stream.write(reinterpret_cast<const char*>(fbBuilder.GetBufferPointer()), bytesToWrite);
    return !stream.bad();
}

} // namespace armnnSerializer
