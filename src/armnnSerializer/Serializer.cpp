//
// Copyright © 2017,2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "Serializer.hpp"
#include "SerializerUtils.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <fmt/format.h>
#include <iostream>

using namespace armnn;
namespace fb = flatbuffers;
namespace serializer = armnnSerializer;

namespace armnnSerializer
{

ISerializer::ISerializer() : pSerializerImpl(new SerializerImpl())
{
}

ISerializer::~ISerializer() = default;

ISerializer* ISerializer::CreateRaw()
{
    return new ISerializer();
}

ISerializerPtr ISerializer::Create()
{
    return ISerializerPtr(CreateRaw(), &ISerializer::Destroy);
}

void ISerializer::Destroy(ISerializer* serializer)
{
    delete serializer;
}

void ISerializer::Serialize(const armnn::INetwork& inNetwork)
{
    pSerializerImpl->Serialize(inNetwork);
}

bool ISerializer::SaveSerializedToStream(std::ostream& stream)
{
    return pSerializerImpl->SaveSerializedToStream(stream);
}

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
        case armnn::ActivationFunction::Elu:
            return serializer::ActivationFunction::ActivationFunction_Elu;
        case armnn::ActivationFunction::HardSwish:
            return serializer::ActivationFunction::ActivationFunction_HardSwish;
        default:
            return serializer::ActivationFunction::ActivationFunction_Sigmoid;
    }
}

serializer::ArgMinMaxFunction GetFlatBufferArgMinMaxFunction(armnn::ArgMinMaxFunction function)
{
    switch (function)
    {
        case armnn::ArgMinMaxFunction::Max:
            return serializer::ArgMinMaxFunction::ArgMinMaxFunction_Max;
        case armnn::ArgMinMaxFunction::Min:
        default:
            return serializer::ArgMinMaxFunction::ArgMinMaxFunction_Min;
    }
}

uint32_t SerializerStrategy::GetSerializedId(LayerGuid guid)
{
    if (m_guidMap.empty())
    {
        m_guidMap.insert(std::make_pair(guid, m_layerId));
    }
    else if (m_guidMap.find(guid) == m_guidMap.end())
    {
        ++m_layerId;
        m_guidMap.insert(std::make_pair(guid, m_layerId));

        return m_layerId;
    }
    return m_guidMap[guid];
}

// Build FlatBuffer for Input Layer
void SerializerStrategy::SerializeInputLayer(const armnn::IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferInputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Input);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferInputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                flatBufferInputBaseLayer,
                                                                                id);
    // Push layer binding id to outputIds.
    m_inputIds.push_back(id);

    // Create the FlatBuffer InputLayer
    auto flatBufferInputLayer = serializer::CreateInputLayer(m_flatBufferBuilder, flatBufferInputBindableBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferInputLayer.o, serializer::Layer::Layer_InputLayer);
}

// Build FlatBuffer for Output Layer
void SerializerStrategy::SerializeOutputLayer(const armnn::IConnectableLayer* layer,
                                              LayerBindingId id, const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferOutputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Output);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferOutputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                 flatBufferOutputBaseLayer,
                                                                                 id);
    // Push layer binding id to outputIds.
    m_outputIds.push_back(id);

    // Create the FlatBuffer OutputLayer
    auto flatBufferOutputLayer = serializer::CreateOutputLayer(m_flatBufferBuilder, flatBufferOutputBindableBaseLayer);
    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferOutputLayer.o, serializer::Layer::Layer_OutputLayer);
}

// Build FlatBuffer for Activation Layer
void SerializerStrategy::SerializeActivationLayer(const armnn::IConnectableLayer* layer,
                                                  const armnn::ActivationDescriptor& descriptor,
                                                  const char* name)
{
    IgnoreUnused(name);

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
void SerializerStrategy::SerializeAdditionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferAdditionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Addition);

    // Create the FlatBuffer AdditionLayer
    auto flatBufferAdditionLayer = serializer::CreateAdditionLayer(m_flatBufferBuilder, flatBufferAdditionBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferAdditionLayer.o, serializer::Layer::Layer_AdditionLayer);
}

// Build FlatBuffer for ArgMinMax Layer
void SerializerStrategy::SerializeArgMinMaxLayer(const armnn::IConnectableLayer *layer,
                                                 const armnn::ArgMinMaxDescriptor& descriptor,
                                                 const char *name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_ArgMinMax);

    // Create FlatBuffer Descriptor
    auto flatBufferDescriptor = CreateArgMinMaxDescriptor(m_flatBufferBuilder,
                                                          GetFlatBufferArgMinMaxFunction(descriptor.m_Function),
                                                          descriptor.m_Axis);

    // Create FlatBuffer ArgMinMaxLayer
    auto flatBufferLayer = CreateArgMinMaxLayer(m_flatBufferBuilder,
                                                flatBufferBaseLayer,
                                                flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ArgMinMaxLayer);
}

void SerializerStrategy::SerializeBatchMatMulLayer(const armnn::IConnectableLayer* layer,
                                                   const armnn::BatchMatMulDescriptor& descriptor,
                                                   const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_BatchMatMul);

    // Create the FlatBuffer BatchMatMulDescriptor
    auto flatBufferDescriptor = CreateBatchMatMulDescriptor(m_flatBufferBuilder,
                                                            descriptor.m_TransposeX,
                                                            descriptor.m_TransposeY,
                                                            descriptor.m_AdjointX,
                                                            descriptor.m_AdjointY,
                                                            GetFlatBufferDataLayout(descriptor.m_DataLayoutX),
                                                            GetFlatBufferDataLayout(descriptor.m_DataLayoutY));

    // Create the FlatBuffer BatchMatMulLayer
    auto flatBufferBatchMatMulLayer = CreateBatchMatMulLayer(m_flatBufferBuilder,
                                                             flatBufferBaseLayer,
                                                             flatBufferDescriptor);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferBatchMatMulLayer.o, serializer::Layer::Layer_BatchMatMulLayer);
}

// Build FlatBuffer for BatchToSpaceNd Layer
void SerializerStrategy::SerializeBatchToSpaceNdLayer(const armnn::IConnectableLayer* layer,
                                                      const armnn::BatchToSpaceNdDescriptor& descriptor,
                                                      const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeBatchNormalizationLayer(
        const armnn::IConnectableLayer* layer,
        const armnn::BatchNormalizationDescriptor& batchNormDescriptor,
        const std::vector<armnn::ConstTensor>& constants,
        const char* name)
{
    IgnoreUnused(name);

    const armnn::ConstTensor& mean     = constants[0];
    const armnn::ConstTensor& variance = constants[1];
    const armnn::ConstTensor& beta     = constants[2];
    const armnn::ConstTensor& gamma    = constants[3];

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

void SerializerStrategy::SerializeCastLayer(const armnn::IConnectableLayer* layer,
                                            const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Cast);
    auto fbCastLayer = serializer::CreateCastLayer(m_flatBufferBuilder, fbBaseLayer);
    CreateAnyLayer(fbCastLayer.o, serializer::Layer::Layer_CastLayer);
}

void SerializerStrategy::SerializeChannelShuffleLayer(const armnn::IConnectableLayer* layer,
                                                      const armnn::ChannelShuffleDescriptor& descriptor,
                                                      const char* name)
{
    IgnoreUnused(name);
    auto fbDescriptor = CreateChannelShuffleDescriptor(m_flatBufferBuilder,
                                                       descriptor.m_Axis,
                                                       descriptor.m_NumGroups);
    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_ChannelShuffle);
    auto fbChannelShuffleLayer = serializer::CreateChannelShuffleLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);
    CreateAnyLayer(fbChannelShuffleLayer.o, serializer::Layer::Layer_ChannelShuffleLayer);
}

void SerializerStrategy::SerializeComparisonLayer(const armnn::IConnectableLayer* layer,
                                             const armnn::ComparisonDescriptor& descriptor,
                                             const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Comparison);
    auto fbDescriptor = serializer::CreateComparisonDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferComparisonOperation(descriptor.m_Operation));

    auto fbLayer = serializer::CreateComparisonLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);
    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_ComparisonLayer);
}

// Build FlatBuffer for Constant Layer
void SerializerStrategy::SerializeConstantLayer(const armnn::IConnectableLayer* layer,
                                                const std::vector<armnn::ConstTensor>& constants,
                                                const char* name)
{
    IgnoreUnused(name);

    armnn::ConstTensor input = constants[0];

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
void SerializerStrategy::SerializeConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                                     const armnn::Convolution2dDescriptor& descriptor,
                                                     const char* name)
{
    IgnoreUnused(name);

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

    // Create the FlatBuffer Convolution2dLayer
    auto flatBufferLayer = CreateConvolution2dLayer(m_flatBufferBuilder,
                                                    flatBufferBaseLayer,
                                                    flatBufferDescriptor);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_Convolution2dLayer);
}

// Build FlatBuffer for Convolution3dLayer
void SerializerStrategy::SerializeConvolution3dLayer(const armnn::IConnectableLayer* layer,
                                                     const armnn::Convolution3dDescriptor& descriptor,
                                                     const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Convolution3d);

    auto flatBufferDescriptor = CreateConvolution3dDescriptor(m_flatBufferBuilder,
                                                              descriptor.m_PadLeft,
                                                              descriptor.m_PadRight,
                                                              descriptor.m_PadTop,
                                                              descriptor.m_PadBottom,
                                                              descriptor.m_PadFront,
                                                              descriptor.m_PadBack,
                                                              descriptor.m_StrideX,
                                                              descriptor.m_StrideY,
                                                              descriptor.m_StrideZ,
                                                              descriptor.m_DilationX,
                                                              descriptor.m_DilationY,
                                                              descriptor.m_DilationZ,
                                                              descriptor.m_BiasEnabled,
                                                              GetFlatBufferDataLayout(descriptor.m_DataLayout));

    // Create the FlatBuffer Convolution3dLayer
    auto flatBufferLayer = CreateConvolution3dLayer(m_flatBufferBuilder,
                                                    flatBufferBaseLayer,
                                                    flatBufferDescriptor);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_Convolution3dLayer);
}

void SerializerStrategy::SerializeDepthToSpaceLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::DepthToSpaceDescriptor& descriptor,
                                               const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_DepthToSpace);
    auto fbDescriptor = CreateDepthToSpaceDescriptor(m_flatBufferBuilder,
                                                     descriptor.m_BlockSize,
                                                     GetFlatBufferDataLayout(descriptor.m_DataLayout));

    auto fbLayer = serializer::CreateDepthToSpaceLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_DepthToSpaceLayer);
}

void SerializerStrategy::SerializeDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                                              const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                                              const char* name)
{
    IgnoreUnused(name);

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

    auto flatBufferLayer = CreateDepthwiseConvolution2dLayer(m_flatBufferBuilder,
                                                             fbBaseLayer,
                                                             fbDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_DepthwiseConvolution2dLayer);
}

void SerializerStrategy::SerializeDequantizeLayer(const armnn::IConnectableLayer* layer,
                                             const char* name)
{
    IgnoreUnused(name);

    auto fbDequantizeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Dequantize);
    auto fbDequantizeLayer     = serializer::CreateDequantizeLayer(m_flatBufferBuilder, fbDequantizeBaseLayer);

    CreateAnyLayer(fbDequantizeLayer.o, serializer::Layer::Layer_DequantizeLayer);
}

void SerializerStrategy::SerializeDetectionPostProcessLayer(const armnn::IConnectableLayer* layer,
                                                            const armnn::DetectionPostProcessDescriptor& descriptor,
                                                            const std::vector<armnn::ConstTensor>& constants,
                                                            const char* name)
{
    IgnoreUnused(name);

    const armnn::ConstTensor& anchors = constants[0];

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

void SerializerStrategy::SerializeDivisionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbDivisionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Division);
    auto fbDivisionLayer     = serializer::CreateDivisionLayer(m_flatBufferBuilder, fbDivisionBaseLayer);

    CreateAnyLayer(fbDivisionLayer.o, serializer::Layer::Layer_DivisionLayer);
}

void SerializerStrategy::SerializeElementwiseBinaryLayer(const armnn::IConnectableLayer* layer,
                                                         const armnn::ElementwiseBinaryDescriptor& descriptor,
                                                         const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_ElementwiseBinary);
    auto fbDescriptor = serializer::CreateElementwiseBinaryDescriptor(
            m_flatBufferBuilder,
            GetFlatBufferBinaryOperation(descriptor.m_Operation));

    auto fbLayer = serializer::CreateElementwiseBinaryLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);
    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_ElementwiseBinaryLayer);
}

void SerializerStrategy::SerializeElementwiseUnaryLayer(const armnn::IConnectableLayer* layer,
                                                   const armnn::ElementwiseUnaryDescriptor& descriptor,
                                                   const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_ElementwiseUnary);
    auto fbDescriptor = serializer::CreateElementwiseUnaryDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferUnaryOperation(descriptor.m_Operation));

    auto fbLayer = serializer::CreateElementwiseUnaryLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);
    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_ElementwiseUnaryLayer);
}

void SerializerStrategy::SerializeFillLayer(const armnn::IConnectableLayer* layer,
                                       const armnn::FillDescriptor& fillDescriptor,
                                       const char* name)
{
    IgnoreUnused(name);

    auto fbFillBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Fill);

    auto fbDescriptor = serializer::CreateFillDescriptor(m_flatBufferBuilder, fillDescriptor.m_Value);

    auto fbFillLayer = serializer::CreateFillLayer(m_flatBufferBuilder, fbFillBaseLayer, fbDescriptor);

    CreateAnyLayer(fbFillLayer.o, serializer::Layer::Layer_FillLayer);
}

void SerializerStrategy::SerializeFloorLayer(const armnn::IConnectableLayer *layer, const char *name)
{
    IgnoreUnused(name);

    auto flatBufferFloorBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Floor);
    auto flatBufferFloorLayer = serializer::CreateFloorLayer(m_flatBufferBuilder, flatBufferFloorBaseLayer);

    CreateAnyLayer(flatBufferFloorLayer.o, serializer::Layer::Layer_FloorLayer);
}

void SerializerStrategy::SerializeGatherLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::GatherDescriptor& gatherDescriptor,
                                         const char* name)
{
    IgnoreUnused(name);

    auto fbGatherDescriptor = CreateGatherDescriptor(m_flatBufferBuilder,
                                                     gatherDescriptor.m_Axis);
    auto fbGatherBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Gather);
    auto flatBufferLayer   = serializer::CreateGatherLayer(m_flatBufferBuilder, fbGatherBaseLayer, fbGatherDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_GatherLayer);
}

void SerializerStrategy::SerializeGatherNdLayer(const armnn::IConnectableLayer* layer,
                                                const char* name)
{
    IgnoreUnused(name);

    auto fbGatherNdBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_GatherNd);
    auto flatBufferLayer     = serializer::CreateGatherNdLayer(m_flatBufferBuilder, fbGatherNdBaseLayer);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_GatherNdLayer);
}

void SerializerStrategy::SerializeInstanceNormalizationLayer(
    const armnn::IConnectableLayer* layer,
    const armnn::InstanceNormalizationDescriptor& instanceNormalizationDescriptor,
    const char* name)
{
    IgnoreUnused(name);

    auto fbDescriptor = serializer::CreateInstanceNormalizationDescriptor(
            m_flatBufferBuilder,
            instanceNormalizationDescriptor.m_Gamma,
            instanceNormalizationDescriptor.m_Beta,
            instanceNormalizationDescriptor.m_Eps,
            GetFlatBufferDataLayout(instanceNormalizationDescriptor.m_DataLayout));

    auto fbBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_InstanceNormalization);
    auto fbLayer     = serializer::CreateInstanceNormalizationLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_InstanceNormalizationLayer);
}

void SerializerStrategy::SerializeL2NormalizationLayer(const armnn::IConnectableLayer* layer,
                                                  const armnn::L2NormalizationDescriptor& l2NormalizationDescriptor,
                                                  const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeLogicalBinaryLayer(const armnn::IConnectableLayer* layer,
                                                const armnn::LogicalBinaryDescriptor& descriptor,
                                                const char* name)
{
    IgnoreUnused(name);

    auto fbBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_LogicalBinary);
    auto fbDescriptor = serializer::CreateLogicalBinaryDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferLogicalBinaryOperation(descriptor.m_Operation));

    auto fbLayer = serializer::CreateLogicalBinaryLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);
    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_LogicalBinaryLayer);
}

void SerializerStrategy::SerializeLogSoftmaxLayer(const armnn::IConnectableLayer* layer,
                                             const armnn::LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                             const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferLogSoftmaxBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_LogSoftmax);

    // Create the FlatBuffer LogSoftmaxDescriptor
    auto flatBufferLogSoftmaxDesc =
        serializer::CreateLogSoftmaxDescriptor(m_flatBufferBuilder,
                                               logSoftmaxDescriptor.m_Beta,
                                               logSoftmaxDescriptor.m_Axis);

    // Create the FlatBuffer LogSoftmaxLayer
    auto flatBufferLogSoftmaxLayer =
        serializer::CreateLogSoftmaxLayer(m_flatBufferBuilder,
                                          flatBufferLogSoftmaxBaseLayer,
                                          flatBufferLogSoftmaxDesc);

    CreateAnyLayer(flatBufferLogSoftmaxLayer.o, serializer::Layer::Layer_LogSoftmaxLayer);
}

void SerializerStrategy::SerializeLstmLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::LstmDescriptor& descriptor,
                                            const std::vector<armnn::ConstTensor>& constants,
                                            const char* name)
{
    IgnoreUnused(name);

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

    // Index for constants vector
    std::size_t i = 0;

    // Get mandatory/basic input parameters
    auto inputToForgetWeights     = CreateConstTensorInfo(constants[i++]); //InputToForgetWeights
    auto inputToCellWeights       = CreateConstTensorInfo(constants[i++]); //InputToCellWeights
    auto inputToOutputWeights     = CreateConstTensorInfo(constants[i++]); //InputToOutputWeights
    auto recurrentToForgetWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToForgetWeights
    auto recurrentToCellWeights   = CreateConstTensorInfo(constants[i++]); //RecurrentToCellWeights
    auto recurrentToOutputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToOutputWeights
    auto forgetGateBias           = CreateConstTensorInfo(constants[i++]); //ForgetGateBias
    auto cellBias                 = CreateConstTensorInfo(constants[i++]); //CellBias
    auto outputGateBias           = CreateConstTensorInfo(constants[i++]); //OutputGateBias



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
        inputToInputWeights = CreateConstTensorInfo(constants[i++]); //InputToInputWeights
        recurrentToInputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToInputWeights
        inputGateBias = CreateConstTensorInfo(constants[i++]); //InputGateBias
    }

    if (descriptor.m_PeepholeEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            cellToInputWeights = CreateConstTensorInfo(constants[i++]); //CellToInputWeights
        }
        cellToForgetWeights = CreateConstTensorInfo(constants[i++]); //CellToForgetWeights
        cellToOutputWeights = CreateConstTensorInfo(constants[i++]); //CellToOutputWeights
    }

    if (descriptor.m_ProjectionEnabled)
    {
        projectionWeights = CreateConstTensorInfo(constants[i++]); //ProjectionWeights
        projectionBias = CreateConstTensorInfo(constants[i++]); //ProjectionBias
    }

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            inputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //InputLayerNormWeights
        }
        forgetLayerNormWeights = CreateConstTensorInfo(constants[i++]); //ForgetLayerNormWeights
        cellLayerNormWeights   = CreateConstTensorInfo(constants[i++]); //CellLayerNormWeights
        outputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //OutputLayerNormWeights
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

void SerializerStrategy::SerializeMaximumLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbMaximumBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Maximum);
    auto fbMaximumLayer     = serializer::CreateMaximumLayer(m_flatBufferBuilder, fbMaximumBaseLayer);

    CreateAnyLayer(fbMaximumLayer.o, serializer::Layer::Layer_MaximumLayer);
}

void SerializerStrategy::SerializeMeanLayer(const armnn::IConnectableLayer* layer,
                                       const armnn::MeanDescriptor& descriptor,
                                       const char* name)
{
    IgnoreUnused(name);

    auto fbMeanBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Mean);
    auto fbMeanDescriptor = serializer::CreateMeanDescriptor(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(descriptor.m_Axis),
                                                             descriptor.m_KeepDims);

    auto fbMeanLayer = serializer::CreateMeanLayer(m_flatBufferBuilder,
                                                   fbMeanBaseLayer,
                                                   fbMeanDescriptor);

    CreateAnyLayer(fbMeanLayer.o, serializer::Layer::Layer_MeanLayer);
}

void SerializerStrategy::SerializeMinimumLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbMinimumBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Minimum);
    auto fbMinimumLayer     = serializer::CreateMinimumLayer(m_flatBufferBuilder, fbMinimumBaseLayer);

    CreateAnyLayer(fbMinimumLayer.o, serializer::Layer::Layer_MinimumLayer);
}

void SerializerStrategy::SerializeMergeLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbMergeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Merge);
    auto fbMergeLayer     = serializer::CreateMergeLayer(m_flatBufferBuilder, fbMergeBaseLayer);

    CreateAnyLayer(fbMergeLayer.o, serializer::Layer::Layer_MergeLayer);
}

void SerializerStrategy::SerializeConcatLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::ConcatDescriptor& concatDescriptor,
                                         const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeMultiplicationLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbMultiplicationBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Multiplication);
    auto fbMultiplicationLayer     = serializer::CreateMultiplicationLayer(m_flatBufferBuilder,
                                                                           fbMultiplicationBaseLayer);

    CreateAnyLayer(fbMultiplicationLayer.o, serializer::Layer::Layer_MultiplicationLayer);
}

void SerializerStrategy::SerializePadLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::PadDescriptor& padDescriptor,
                                      const char* name)
{
    IgnoreUnused(name);

    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Pad);

    std::vector<unsigned int> padList;
    for (auto& p: padDescriptor.m_PadList)
    {
        padList.push_back(p.first);
        padList.push_back(p.second);
    }

    auto flatBufferPadDesc = serializer::CreatePadDescriptor(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(padList),
                                                             padDescriptor.m_PadValue,
                                                             GetFlatBufferPaddingMode(padDescriptor.m_PaddingMode));

    auto flatBufferPadLayer = serializer::CreatePadLayer(m_flatBufferBuilder,
                                                         flatBufferBaseLayer,
                                                         flatBufferPadDesc);

    CreateAnyLayer(flatBufferPadLayer.o, serializer::Layer::Layer_PadLayer);
}

void SerializerStrategy::SerializePermuteLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::PermuteDescriptor& permuteDescriptor,
                                          const char* name)
{
    IgnoreUnused(name);

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

// Build FlatBuffer for Rank Layer
void SerializerStrategy::SerializeRankLayer(const armnn::IConnectableLayer* layer,
                                       const char* name)
{
    IgnoreUnused(name);
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Rank);
    auto flatBufferRankLayer = serializer::CreateRankLayer(m_flatBufferBuilder, flatBufferBaseLayer);

    CreateAnyLayer(flatBufferRankLayer.o, serializer::Layer::Layer_RankLayer);
}

void SerializerStrategy::SerializeReduceLayer(const armnn::IConnectableLayer* layer,
                                             const armnn::ReduceDescriptor& reduceDescriptor,
                                             const char*)
{
    auto fbReduceBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Reduce);
    auto fbDescriptor = CreateReduceDescriptor(m_flatBufferBuilder,
                                               reduceDescriptor.m_KeepDims,
                                               m_flatBufferBuilder.CreateVector(reduceDescriptor.m_vAxis),
                                               GetFlatBufferReduceOperation(reduceDescriptor.m_ReduceOperation));
    auto fbReduceLayer = serializer::CreateReduceLayer(m_flatBufferBuilder,
                                                       fbReduceBaseLayer,
                                                       fbDescriptor);

    CreateAnyLayer(fbReduceLayer.o, serializer::Layer::Layer_ReduceLayer);
}

// Build FlatBuffer for Reshape Layer
void SerializerStrategy::SerializeReshapeLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::ReshapeDescriptor& reshapeDescriptor,
                                          const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeResizeLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::ResizeDescriptor& resizeDescriptor,
                                         const char* name)
{
    IgnoreUnused(name);

    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Resize);

    auto flatBufferDescriptor =
            CreateResizeDescriptor(m_flatBufferBuilder,
                                   resizeDescriptor.m_TargetHeight,
                                   resizeDescriptor.m_TargetWidth,
                                   GetFlatBufferResizeMethod(resizeDescriptor.m_Method),
                                   GetFlatBufferDataLayout(resizeDescriptor.m_DataLayout),
                                   resizeDescriptor.m_AlignCorners,
                                   resizeDescriptor.m_HalfPixelCenters);

    auto flatBufferLayer = serializer::CreateResizeLayer(m_flatBufferBuilder,
                                                         flatBufferBaseLayer,
                                                         flatBufferDescriptor);

    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_ResizeLayer);
}

void SerializerStrategy::SerializeSliceLayer(const armnn::IConnectableLayer* layer,
                                        const armnn::SliceDescriptor& sliceDescriptor,
                                        const char* name)
{
    IgnoreUnused(name);

    auto fbSliceBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Slice);
    auto fbSliceDescriptor = CreateSliceDescriptor(m_flatBufferBuilder,
                                                   m_flatBufferBuilder.CreateVector(sliceDescriptor.m_Begin),
                                                   m_flatBufferBuilder.CreateVector(sliceDescriptor.m_Size));

    auto fbSliceLayer = serializer::CreateSliceLayer(m_flatBufferBuilder, fbSliceBaseLayer, fbSliceDescriptor);

    CreateAnyLayer(fbSliceLayer.o, serializer::Layer::Layer_SliceLayer);
}

// Build FlatBuffer for Softmax Layer
void SerializerStrategy::SerializeSoftmaxLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::SoftmaxDescriptor& softmaxDescriptor,
                                          const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferSoftmaxBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Softmax);

    // Create the FlatBuffer SoftmaxDescriptor
    auto flatBufferSoftmaxDesc =
        serializer::CreateSoftmaxDescriptor(m_flatBufferBuilder,
                                            softmaxDescriptor.m_Beta,
                                            softmaxDescriptor.m_Axis);

    // Create the FlatBuffer SoftmaxLayer
    auto flatBufferSoftmaxLayer =
        serializer::CreateSoftmaxLayer(m_flatBufferBuilder,
                                       flatBufferSoftmaxBaseLayer,
                                       flatBufferSoftmaxDesc);

    CreateAnyLayer(flatBufferSoftmaxLayer.o, serializer::Layer::Layer_SoftmaxLayer);
}

void SerializerStrategy::SerializePooling2dLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::Pooling2dDescriptor& pooling2dDescriptor,
                                            const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializePooling3dLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::Pooling3dDescriptor& pooling3dDescriptor,
                                            const char* name)
{
    IgnoreUnused(name);

    auto fbPooling3dBaseLayer  = CreateLayerBase(layer, serializer::LayerType::LayerType_Pooling3d);
    auto fbPooling3dDescriptor = serializer::CreatePooling3dDescriptor(
        m_flatBufferBuilder,
        GetFlatBufferPoolingAlgorithm(pooling3dDescriptor.m_PoolType),
        pooling3dDescriptor.m_PadLeft,
        pooling3dDescriptor.m_PadRight,
        pooling3dDescriptor.m_PadTop,
        pooling3dDescriptor.m_PadBottom,
        pooling3dDescriptor.m_PadFront,
        pooling3dDescriptor.m_PadBack,
        pooling3dDescriptor.m_PoolWidth,
        pooling3dDescriptor.m_PoolHeight,
        pooling3dDescriptor.m_PoolDepth,
        pooling3dDescriptor.m_StrideX,
        pooling3dDescriptor.m_StrideY,
        pooling3dDescriptor.m_StrideZ,
        GetFlatBufferOutputShapeRounding(pooling3dDescriptor.m_OutputShapeRounding),
        GetFlatBufferPaddingMethod(pooling3dDescriptor.m_PaddingMethod),
        GetFlatBufferDataLayout(pooling3dDescriptor.m_DataLayout));

    auto fbPooling3dLayer = serializer::CreatePooling3dLayer(m_flatBufferBuilder,
                                                             fbPooling3dBaseLayer,
                                                             fbPooling3dDescriptor);

    CreateAnyLayer(fbPooling3dLayer.o, serializer::Layer::Layer_Pooling3dLayer);
}

void SerializerStrategy::SerializePreluLayer(const armnn::IConnectableLayer* layer,
                                        const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferPreluBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Prelu);

    // Create the FlatBuffer AdditionLayer
    auto flatBufferPreluLayer = serializer::CreatePreluLayer(m_flatBufferBuilder, flatBufferPreluBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferPreluLayer.o, serializer::Layer::Layer_PreluLayer);
}

void SerializerStrategy::SerializeQuantizeLayer(const armnn::IConnectableLayer *layer, const char *name)
{
    IgnoreUnused(name);

    auto fbQuantizeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Quantize);
    auto fbQuantizeLayer = serializer::CreateQuantizeLayer(m_flatBufferBuilder,
                                                           fbQuantizeBaseLayer);
    CreateAnyLayer(fbQuantizeLayer.o, serializer::Layer::Layer_QuantizeLayer);
}

// Build FlatBuffer for FullyConnected Layer
void SerializerStrategy::SerializeFullyConnectedLayer(const armnn::IConnectableLayer* layer,
                                                      const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                      const char*)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_FullyConnected);

    // Create FlatBuffer FullyConnectedDescriptor
    auto flatBufferDescriptor =
        serializer::CreateFullyConnectedDescriptor(m_flatBufferBuilder,
                                                   fullyConnectedDescriptor.m_BiasEnabled,
                                                   fullyConnectedDescriptor.m_TransposeWeightMatrix,
                                                   fullyConnectedDescriptor.m_ConstantWeights);

    // Create FlatBuffer FullyConnectedLayer
    auto flatBufferLayer = serializer::CreateFullyConnectedLayer(m_flatBufferBuilder,
                                                                 flatBufferBaseLayer,
                                                                 flatBufferDescriptor);

    // Add created FullyConnectedLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_FullyConnectedLayer);
}

// Build FlatBuffer for SpaceToBatchNd Layer
void SerializerStrategy::SerializeSpaceToBatchNdLayer(const armnn::IConnectableLayer* layer,
                                                 const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                 const char* name)
{
    IgnoreUnused(name);

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
void SerializerStrategy::SerializeSpaceToDepthLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                               const char* name)
{
    IgnoreUnused(name);

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
void SerializerStrategy::SerializeSplitterLayer(const armnn::IConnectableLayer* layer,
                                           const armnn::ViewsDescriptor& viewsDescriptor,
                                           const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeNormalizationLayer(const armnn::IConnectableLayer* layer,
                                                const armnn::NormalizationDescriptor& descriptor,
                                                const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeShapeLayer(const armnn::IConnectableLayer* layer,
                                             const char* name)
{
    IgnoreUnused(name);

    auto shapeBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Shape);
    auto shapeLayer = serializer::CreateShapeLayer(m_flatBufferBuilder, shapeBaseLayer);

    CreateAnyLayer(shapeLayer.o, serializer::Layer::Layer_ShapeLayer);
}

void SerializerStrategy::SerializeStackLayer(const armnn::IConnectableLayer* layer,
                                        const armnn::StackDescriptor& stackDescriptor,
                                        const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeStandInLayer(const armnn::IConnectableLayer *layer,
                                          const armnn::StandInDescriptor& standInDescriptor,
                                          const char *name)
{
    IgnoreUnused(name);

    auto fbDescriptor = serializer::CreateStandInDescriptor(m_flatBufferBuilder,
                                                            standInDescriptor.m_NumInputs,
                                                            standInDescriptor.m_NumOutputs);

    auto fbBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_StandIn);
    auto fbLayer     = serializer::CreateStandInLayer(m_flatBufferBuilder, fbBaseLayer, fbDescriptor);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_StandInLayer);
}

void SerializerStrategy::SerializeStridedSliceLayer(const armnn::IConnectableLayer* layer,
                                               const armnn::StridedSliceDescriptor& stridedSliceDescriptor,
                                               const char* name)
{
    IgnoreUnused(name);

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

void SerializerStrategy::SerializeSubtractionLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbSubtractionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Subtraction);
    auto fbSubtractionLayer = serializer::CreateSubtractionLayer(m_flatBufferBuilder, fbSubtractionBaseLayer);

    CreateAnyLayer(fbSubtractionLayer.o, serializer::Layer::Layer_SubtractionLayer);
}

void SerializerStrategy::SerializeSwitchLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    IgnoreUnused(name);

    auto fbSwitchBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Switch);
    auto fbSwitchLayer = serializer::CreateSwitchLayer(m_flatBufferBuilder, fbSwitchBaseLayer);

    CreateAnyLayer(fbSwitchLayer.o, serializer::Layer::Layer_SwitchLayer);
}

void SerializerStrategy::SerializeTransposeConvolution2dLayer(
    const armnn::IConnectableLayer* layer,
    const armnn::TransposeConvolution2dDescriptor& descriptor,
    const std::vector<armnn::ConstTensor>& constants,
    const char* name)
{
    IgnoreUnused(name);

    const armnn::ConstTensor& weights = constants.at(0);

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
    if (constants.size() > 1)
    {
        const armnn::ConstTensor& biases = constants.at(1);
        fbBiasesConstTensorInfo = CreateConstTensorInfo(biases);
    }

    auto fbLayer = CreateTransposeConvolution2dLayer(m_flatBufferBuilder,
                                                     fbBaseLayer,
                                                     fbDescriptor,
                                                     fbWeightsConstTensorInfo,
                                                     fbBiasesConstTensorInfo);

    CreateAnyLayer(fbLayer.o, serializer::Layer::Layer_TransposeConvolution2dLayer);
}

void SerializerStrategy::SerializeTransposeLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::TransposeDescriptor& descriptor,
                                            const char* name)
{
    IgnoreUnused(name);

    // Create FlatBuffer BaseLayer
    auto flatBufferBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Transpose);

    std::vector<unsigned int> dimMappings;
    for (unsigned int i=0; i<descriptor.m_DimMappings.GetSize(); ++i)
    {
        dimMappings.push_back(descriptor.m_DimMappings[i]);
    }

    auto flatBufferDesc = serializer::CreateTransposeDescriptor(m_flatBufferBuilder,
                                                                m_flatBufferBuilder.CreateVector(dimMappings));

    // Create the FlatBuffer TransposeLayer
    auto flatBufferLayer = serializer::CreateTransposeLayer(m_flatBufferBuilder,
                                                            flatBufferBaseLayer,
                                                            flatBufferDesc);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferLayer.o, serializer::Layer::Layer_TransposeLayer);
}

void SerializerStrategy::SerializeQLstmLayer(const armnn::IConnectableLayer* layer,
                                             const armnn::QLstmDescriptor& descriptor,
                                             const std::vector<armnn::ConstTensor>& constants,
                                             const char* name)
{
    IgnoreUnused(name);

    auto fbQLstmBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_QLstm);

    auto fbQLstmDescriptor = serializer::CreateQLstmDescriptor(
            m_flatBufferBuilder,
            descriptor.m_CifgEnabled,
            descriptor.m_PeepholeEnabled,
            descriptor.m_ProjectionEnabled,
            descriptor.m_LayerNormEnabled,
            descriptor.m_CellClip,
            descriptor.m_ProjectionClip,
            descriptor.m_InputIntermediateScale,
            descriptor.m_ForgetIntermediateScale,
            descriptor.m_CellIntermediateScale,
            descriptor.m_OutputIntermediateScale,
            descriptor.m_HiddenStateZeroPoint,
            descriptor.m_HiddenStateScale
            );

    // Index for constants vector
    std::size_t i = 0;

    // Mandatory params
    auto inputToForgetWeights     = CreateConstTensorInfo(constants[i++]); //InputToForgetWeights
    auto inputToCellWeights       = CreateConstTensorInfo(constants[i++]); //InputToCellWeights
    auto inputToOutputWeights     = CreateConstTensorInfo(constants[i++]); //InputToOutputWeights
    auto recurrentToForgetWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToForgetWeights
    auto recurrentToCellWeights   = CreateConstTensorInfo(constants[i++]); //RecurrentToCellWeights
    auto recurrentToOutputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToOutputWeights
    auto forgetGateBias           = CreateConstTensorInfo(constants[i++]); //ForgetGateBias
    auto cellBias                 = CreateConstTensorInfo(constants[i++]); //CellBias
    auto outputGateBias           = CreateConstTensorInfo(constants[i++]); //OutputGateBias

    // CIFG
    flatbuffers::Offset<serializer::ConstTensor> inputToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> recurrentToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> inputGateBias;

    if (!descriptor.m_CifgEnabled)
    {
        inputToInputWeights = CreateConstTensorInfo(constants[i++]); //InputToInputWeights
        recurrentToInputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToInputWeights
        inputGateBias = CreateConstTensorInfo(constants[i++]); //InputGateBias
    }

    // Peephole
    flatbuffers::Offset<serializer::ConstTensor> cellToInputWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellToForgetWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellToOutputWeights;

    if (descriptor.m_PeepholeEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            cellToInputWeights = CreateConstTensorInfo(constants[i++]); //CellToInputWeights
        }
        cellToForgetWeights = CreateConstTensorInfo(constants[i++]); //CellToForgetWeights
        cellToOutputWeights = CreateConstTensorInfo(constants[i++]); //CellToOutputWeights
    }

    // Projection
    flatbuffers::Offset<serializer::ConstTensor> projectionWeights;
    flatbuffers::Offset<serializer::ConstTensor> projectionBias;

    if (descriptor.m_ProjectionEnabled)
    {
        projectionWeights = CreateConstTensorInfo(constants[i++]); //ProjectionWeights
        projectionBias = CreateConstTensorInfo(constants[i++]); //ProjectionBias
    }

    // Layer norm
    flatbuffers::Offset<serializer::ConstTensor> inputLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> forgetLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> cellLayerNormWeights;
    flatbuffers::Offset<serializer::ConstTensor> outputLayerNormWeights;

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            inputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //InputLayerNormWeights
        }
        forgetLayerNormWeights = CreateConstTensorInfo(constants[i++]); //ForgetLayerNormWeights
        cellLayerNormWeights   = CreateConstTensorInfo(constants[i++]); //CellLayerNormWeights
        outputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //OutputLayerNormWeights
    }

    auto fbQLstmParams = serializer::CreateQLstmInputParams(
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
            inputGateBias,
            projectionWeights,
            projectionBias,
            cellToInputWeights,
            cellToForgetWeights,
            cellToOutputWeights,
            inputLayerNormWeights,
            forgetLayerNormWeights,
            cellLayerNormWeights,
            outputLayerNormWeights);

    auto fbQLstmLayer = serializer::CreateQLstmLayer(
            m_flatBufferBuilder,
            fbQLstmBaseLayer,
            fbQLstmDescriptor,
            fbQLstmParams);

    CreateAnyLayer(fbQLstmLayer.o, serializer::Layer::Layer_QLstmLayer);
}

void SerializerStrategy::SerializeQuantizedLstmLayer(const armnn::IConnectableLayer* layer,
                                                     const std::vector<armnn::ConstTensor>& constants,
                                                     const char* name)
{
    IgnoreUnused(name);

    auto fbQuantizedLstmBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_QuantizedLstm);

    // index for constants vector
    size_t i = 0;

    // Get input parameters
    auto inputToInputWeights  = CreateConstTensorInfo(constants[i++]);
    auto inputToForgetWeights = CreateConstTensorInfo(constants[i++]);
    auto inputToCellWeights   = CreateConstTensorInfo(constants[i++]);
    auto inputToOutputWeights = CreateConstTensorInfo(constants[i++]);

    auto recurrentToInputWeights  = CreateConstTensorInfo(constants[i++]);
    auto recurrentToForgetWeights = CreateConstTensorInfo(constants[i++]);
    auto recurrentToCellWeights   = CreateConstTensorInfo(constants[i++]);
    auto recurrentToOutputWeights = CreateConstTensorInfo(constants[i++]);

    auto inputGateBias  = CreateConstTensorInfo(constants[i++]);
    auto forgetGateBias = CreateConstTensorInfo(constants[i++]);
    auto cellBias       = CreateConstTensorInfo(constants[i++]);
    auto outputGateBias = CreateConstTensorInfo(constants[i++]);

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

void SerializerStrategy::SerializeUnidirectionalSequenceLstmLayer(
    const armnn::IConnectableLayer* layer,
    const armnn::UnidirectionalSequenceLstmDescriptor& descriptor,
    const std::vector<armnn::ConstTensor>& constants,
    const char* name)
{
    IgnoreUnused(name);

    auto fbUnidirectionalSequenceLstmBaseLayer =
        CreateLayerBase(layer, serializer::LayerType::LayerType_UnidirectionalSequenceLstm);

    auto fbUnidirectionalSequenceLstmDescriptor = serializer::CreateUnidirectionalSequenceLstmDescriptor(
        m_flatBufferBuilder,
        descriptor.m_ActivationFunc,
        descriptor.m_ClippingThresCell,
        descriptor.m_ClippingThresProj,
        descriptor.m_CifgEnabled,
        descriptor.m_PeepholeEnabled,
        descriptor.m_ProjectionEnabled,
        descriptor.m_LayerNormEnabled,
        descriptor.m_TimeMajor);

    // Index for constants vector
    std::size_t i = 0;

    // Get mandatory/basic input parameters
    auto inputToForgetWeights     = CreateConstTensorInfo(constants[i++]); //InputToForgetWeights
    auto inputToCellWeights       = CreateConstTensorInfo(constants[i++]); //InputToCellWeights
    auto inputToOutputWeights     = CreateConstTensorInfo(constants[i++]); //InputToOutputWeights
    auto recurrentToForgetWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToForgetWeights
    auto recurrentToCellWeights   = CreateConstTensorInfo(constants[i++]); //RecurrentToCellWeights
    auto recurrentToOutputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToOutputWeights
    auto forgetGateBias           = CreateConstTensorInfo(constants[i++]); //ForgetGateBias
    auto cellBias                 = CreateConstTensorInfo(constants[i++]); //CellBias
    auto outputGateBias           = CreateConstTensorInfo(constants[i++]); //OutputGateBias

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
        inputToInputWeights = CreateConstTensorInfo(constants[i++]); //InputToInputWeights
        recurrentToInputWeights = CreateConstTensorInfo(constants[i++]); //RecurrentToInputWeights
        inputGateBias = CreateConstTensorInfo(constants[i++]); //InputGateBias
    }

    if (descriptor.m_PeepholeEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            cellToInputWeights = CreateConstTensorInfo(constants[i++]); //CellToInputWeights
        }
        cellToForgetWeights = CreateConstTensorInfo(constants[i++]); //CellToForgetWeights
        cellToOutputWeights = CreateConstTensorInfo(constants[i++]); //CellToOutputWeights
    }

    if (descriptor.m_ProjectionEnabled)
    {
        projectionWeights = CreateConstTensorInfo(constants[i++]); //ProjectionWeights
        projectionBias = CreateConstTensorInfo(constants[i++]); //ProjectionBias
    }

    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            inputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //InputLayerNormWeights
        }
        forgetLayerNormWeights = CreateConstTensorInfo(constants[i++]); //ForgetLayerNormWeights
        cellLayerNormWeights   = CreateConstTensorInfo(constants[i++]); //CellLayerNormWeights
        outputLayerNormWeights = CreateConstTensorInfo(constants[i++]); //OutputLayerNormWeights
    }

    auto fbUnidirectionalSequenceLstmParams = serializer::CreateLstmInputParams(
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

    auto fbUnidirectionalSequenceLstmLayer = serializer::CreateUnidirectionalSequenceLstmLayer(
        m_flatBufferBuilder,
        fbUnidirectionalSequenceLstmBaseLayer,
        fbUnidirectionalSequenceLstmDescriptor,
        fbUnidirectionalSequenceLstmParams);

    CreateAnyLayer(fbUnidirectionalSequenceLstmLayer.o, serializer::Layer::Layer_UnidirectionalSequenceLstmLayer);
}

fb::Offset<serializer::LayerBase> SerializerStrategy::CreateLayerBase(const IConnectableLayer* layer,
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

void SerializerStrategy::CreateAnyLayer(const flatbuffers::Offset<void>& layer, const serializer::Layer serializerLayer)
{

    auto anyLayer = armnnSerializer::CreateAnyLayer(m_flatBufferBuilder, serializerLayer, layer);
    m_serializedLayers.push_back(anyLayer);
}

template <typename T>
flatbuffers::Offset<flatbuffers::Vector<T>> SerializerStrategy::CreateDataVector(const void* memory, unsigned int size)
{
    const T* buffer = reinterpret_cast<const T*>(memory);
    std::vector<T> vector(buffer, buffer + (size / sizeof(T)));
    auto fbVector = m_flatBufferBuilder.CreateVector(vector);
    return fbVector;
}

flatbuffers::Offset<TensorInfo>  SerializerStrategy::CreateTensorInfo(const armnn::TensorInfo& tensorInfo)
{
    // Get the dimensions
    std::vector<unsigned int> shape;
    std::vector<bool> specificity;
    // This assumes that the TensorShape constructors have ensured that the size of m_DimensionsSpecificity
    // matches the size of dimensions.
    for(unsigned int dim = 0; dim < tensorInfo.GetShape().GetNumDimensions(); ++dim)
    {
        specificity.push_back(tensorInfo.GetShape().GetDimensionSpecificity(dim));

        if (tensorInfo.GetShape().GetDimensionSpecificity(dim))
        {
            shape.push_back(tensorInfo.GetShape()[dim]);
        }
        else
        {
            shape.push_back(0);
        }
    }

    if (tensorInfo.HasPerAxisQuantization())
    {
        // Create FlatBuffer TensorInfo
        auto flatBufferTensorInfo =
            serializer::CreateTensorInfo(m_flatBufferBuilder,
                                         m_flatBufferBuilder.CreateVector(shape),
                                         GetFlatBufferDataType(tensorInfo.GetDataType()),
                                         tensorInfo.GetQuantizationScales()[0],
                                         tensorInfo.GetQuantizationOffset(),
                                         m_flatBufferBuilder.CreateVector(tensorInfo.GetQuantizationScales()),
                                         tensorInfo.GetQuantizationDim().value(),
                                         static_cast<unsigned int>
                                         (tensorInfo.GetShape().GetDimensionality()),
                                         m_flatBufferBuilder.CreateVector(specificity));
        return flatBufferTensorInfo;
    }

    // Create FlatBuffer TensorInfo
    auto flatBufferTensorInfo = serializer::CreateTensorInfo(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(shape),
                                                             GetFlatBufferDataType(tensorInfo.GetDataType()),
                                                             tensorInfo.GetQuantizationScale(),
                                                             tensorInfo.GetQuantizationOffset(),
                                                             0,
                                                             0,
                                                             static_cast<unsigned int>
                                                             (tensorInfo.GetShape().GetDimensionality()),
                                                             m_flatBufferBuilder.CreateVector(specificity));
    return flatBufferTensorInfo;
}

flatbuffers::Offset<serializer::ConstTensor>
    SerializerStrategy::CreateConstTensorInfo(const armnn::ConstTensor& constTensor)
{
    armnn::TensorInfo tensorInfo = constTensor.GetInfo();

    flatbuffers::Offset<void> fbPayload;

    switch (tensorInfo.GetDataType())
    {
        case armnn::DataType::Signed64:
        {
            auto fbVector = CreateDataVector<int64_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::LongData> flatBuffersData = serializer::CreateLongData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
            break;
        }
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
        case armnn::DataType::BFloat16:
        case armnn::DataType::QSymmS16:
        {
            auto fbVector = CreateDataVector<int16_t>(constTensor.GetMemoryArea(), constTensor.GetNumBytes());
            flatbuffers::Offset<serializer::ShortData> flatBuffersData = serializer::CreateShortData(
                    m_flatBufferBuilder,
                    fbVector);
            fbPayload = flatBuffersData.o;
            break;
        }
        case armnn::DataType::QSymmS8:
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
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
            CreateTensorInfo(tensorInfo),
            GetFlatBufferConstTensorData(tensorInfo.GetDataType()),
            fbPayload);
    return flatBufferConstTensor;
}

flatbuffers::Offset<armnnSerializer::FeatureCompatibilityVersions> SerializerStrategy::GetVersionTable()
{
    flatbuffers::Offset<armnnSerializer::FeatureCompatibilityVersions> versionsTable =
        serializer::CreateFeatureCompatibilityVersions(
                m_flatBufferBuilder,
                1, // Binding ids scheme version
                1, // Weights layout scheme version
                1  // Constant tensors as inputs version
            );
    return versionsTable;
}

std::vector<fb::Offset<serializer::InputSlot>>
    SerializerStrategy::CreateInputSlots(const armnn::IConnectableLayer* layer)
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
    SerializerStrategy::CreateOutputSlots(const armnn::IConnectableLayer* layer)
{
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots;

    // Get the OutputSlots
    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        const IOutputSlot& outputSlot = layer->GetOutputSlot(slotIndex);
        const armnn::TensorInfo& tensorInfo = outputSlot.GetTensorInfo();

        // Create FlatBuffer Outputslot
        outputSlots.push_back(serializer::CreateOutputSlot(m_flatBufferBuilder,
                                                           slotIndex,
                                                           CreateTensorInfo(tensorInfo)));
    }
    return outputSlots;
}

void SerializerStrategy::ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                         const BaseDescriptor& descriptor,
                                         const std::vector<armnn::ConstTensor>& constants,
                                         const char* name,
                                         const armnn::LayerBindingId id)
{
    IgnoreUnused(constants);

    switch (layer->GetType())
    {
        case armnn::LayerType::Activation :
        {
            const armnn::ActivationDescriptor& layerDescriptor =
                    static_cast<const armnn::ActivationDescriptor&>(descriptor);
            SerializeActivationLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Addition :
        {
            SerializeAdditionLayer(layer, name);
            break;
        }
        case armnn::LayerType::ArgMinMax :
        {
            const armnn::ArgMinMaxDescriptor& layerDescriptor =
                    static_cast<const armnn::ArgMinMaxDescriptor&>(descriptor);
            SerializeArgMinMaxLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::BatchMatMul:
        {
            const armnn::BatchMatMulDescriptor& layerDescriptor =
                    static_cast<const armnn::BatchMatMulDescriptor&>(descriptor);
            SerializeBatchMatMulLayer(layer,
                                      layerDescriptor,
                                      name);
            break;
        }
        case armnn::LayerType::BatchNormalization :
        {
            const armnn::BatchNormalizationDescriptor& layerDescriptor =
                    static_cast<const armnn::BatchNormalizationDescriptor&>(descriptor);
            SerializeBatchNormalizationLayer(layer,
                                             layerDescriptor,
                                             constants,
                                             name);
            break;
        }
        case armnn::LayerType::BatchToSpaceNd :
        {
            const armnn::BatchToSpaceNdDescriptor& layerDescriptor =
                    static_cast<const armnn::BatchToSpaceNdDescriptor&>(descriptor);
            SerializeBatchToSpaceNdLayer(layer,
                                         layerDescriptor,
                                         name);
            break;
        }
        case armnn::LayerType::Cast :
        {
            SerializeCastLayer(layer, name);
            break;
        }
        case armnn::LayerType::ChannelShuffle :
        {
            const armnn::ChannelShuffleDescriptor& layerDescriptor =
                                                     static_cast<const armnn::ChannelShuffleDescriptor&>(descriptor);
            SerializeChannelShuffleLayer(layer,
                                         layerDescriptor,
                                         name);
            break;
        }
        case armnn::LayerType::Comparison :
        {
            const armnn::ComparisonDescriptor& layerDescriptor =
                    static_cast<const armnn::ComparisonDescriptor&>(descriptor);
            SerializeComparisonLayer(layer,
                                     layerDescriptor,
                                     name);
            break;
        }
        case armnn::LayerType::Concat :
        {
            const armnn::ConcatDescriptor& layerDescriptor =
                    static_cast<const armnn::ConcatDescriptor&>(descriptor);
            SerializeConcatLayer(layer,
                                 layerDescriptor,
                                 name);
            break;
        }
        case armnn::LayerType::Constant :
        {
            SerializeConstantLayer(layer,
                                   constants,
                                   name);
            break;
        }
        case armnn::LayerType::Convolution2d :
        {
            const armnn::Convolution2dDescriptor& layerDescriptor =
                    static_cast<const armnn::Convolution2dDescriptor&>(descriptor);
            SerializeConvolution2dLayer(layer,
                                        layerDescriptor,
                                        name);
            break;
        }
        case armnn::LayerType::Convolution3d :
        {
            const armnn::Convolution3dDescriptor& layerDescriptor =
                    static_cast<const armnn::Convolution3dDescriptor&>(descriptor);
            SerializeConvolution3dLayer(layer,
                                        layerDescriptor,
                                        name);
            break;
        }
        case armnn::LayerType::DepthToSpace :
        {
            const armnn::DepthToSpaceDescriptor& layerDescriptor =
                    static_cast<const armnn::DepthToSpaceDescriptor&>(descriptor);
            SerializeDepthToSpaceLayer(layer,
                                       layerDescriptor,
                                       name);
            break;
        }
        case armnn::LayerType::DepthwiseConvolution2d :
        {
            const armnn::DepthwiseConvolution2dDescriptor& layerDescriptor =
                    static_cast<const armnn::DepthwiseConvolution2dDescriptor&>(descriptor);
            SerializeDepthwiseConvolution2dLayer(layer,
                                                 layerDescriptor,
                                                 name);
            break;
        }
        case armnn::LayerType::Dequantize :
        {
            SerializeDequantizeLayer(layer,
                                     name);
            break;
        }
        case armnn::LayerType::DetectionPostProcess :
        {
            const armnn::DetectionPostProcessDescriptor& layerDescriptor =
                    static_cast<const armnn::DetectionPostProcessDescriptor&>(descriptor);
            SerializeDetectionPostProcessLayer(layer, layerDescriptor, constants, name);
            break;
        }
        case armnn::LayerType::Division :
        {
            SerializeDivisionLayer(layer, name);
            break;
        }
        case armnn::LayerType::ElementwiseBinary :
        {
            const armnn::ElementwiseBinaryDescriptor& layerDescriptor =
                    static_cast<const armnn::ElementwiseBinaryDescriptor&>(descriptor);
            SerializeElementwiseBinaryLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::ElementwiseUnary :
        {
            const armnn::ElementwiseUnaryDescriptor& layerDescriptor =
                    static_cast<const armnn::ElementwiseUnaryDescriptor&>(descriptor);
            SerializeElementwiseUnaryLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Fill :
        {
            const armnn::FillDescriptor& layerDescriptor =
                    static_cast<const armnn::FillDescriptor&>(descriptor);
            SerializeFillLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Floor :
        {
            SerializeFloorLayer(layer, name);
            break;
        }
        case armnn::LayerType::FullyConnected :
        {
            const armnn::FullyConnectedDescriptor& layerDescriptor =
                    static_cast<const armnn::FullyConnectedDescriptor&>(descriptor);
            SerializeFullyConnectedLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Gather :
        {
            const armnn::GatherDescriptor& layerDescriptor =
                    static_cast<const armnn::GatherDescriptor&>(descriptor);
            SerializeGatherLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::GatherNd :
        {
            SerializeGatherNdLayer(layer, name);
            break;
        }
        case armnn::LayerType::Input:
        {
            SerializeInputLayer(layer, id, name);
            break;
        }
        case armnn::LayerType::InstanceNormalization :
        {
            const armnn::InstanceNormalizationDescriptor& layerDescriptor =
                    static_cast<const armnn::InstanceNormalizationDescriptor&>(descriptor);
            SerializeInstanceNormalizationLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::L2Normalization :
        {
            const armnn::L2NormalizationDescriptor& layerDescriptor =
                    static_cast<const armnn::L2NormalizationDescriptor&>(descriptor);
            SerializeL2NormalizationLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::LogicalBinary :
        {
            const armnn::LogicalBinaryDescriptor& layerDescriptor =
                    static_cast<const armnn::LogicalBinaryDescriptor&>(descriptor);
            SerializeLogicalBinaryLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::LogSoftmax :
        {
            const armnn::LogSoftmaxDescriptor& layerDescriptor =
                    static_cast<const armnn::LogSoftmaxDescriptor&>(descriptor);
            SerializeLogSoftmaxLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Lstm :
        {
            const armnn::LstmDescriptor& layerDescriptor =
                    static_cast<const armnn::LstmDescriptor&>(descriptor);
            SerializeLstmLayer(layer, layerDescriptor, constants, name);
            break;
        }
        case armnn::LayerType::QLstm :
        {
            const armnn::QLstmDescriptor& layerDescriptor =
                    static_cast<const armnn::QLstmDescriptor&>(descriptor);
            SerializeQLstmLayer(layer, layerDescriptor, constants, name);
            break;
        }
        case armnn::LayerType::Maximum :
        {
            SerializeMaximumLayer(layer, name);
            break;
        }
        case armnn::LayerType::Mean :
        {
            const armnn::MeanDescriptor& layerDescriptor =
                    static_cast<const armnn::MeanDescriptor&>(descriptor);
            SerializeMeanLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Merge :
        {
            SerializeMergeLayer(layer, name);
            break;
        }
        case armnn::LayerType::Minimum :
        {
            SerializeMinimumLayer(layer, name);
            break;
        }
        case armnn::LayerType::Multiplication :
        {
            SerializeMultiplicationLayer(layer, name);
            break;
        }
        case armnn::LayerType::Normalization :
        {
            const armnn::NormalizationDescriptor& layerDescriptor =
                    static_cast<const armnn::NormalizationDescriptor&>(descriptor);
            SerializeNormalizationLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Output:
        {
            SerializeOutputLayer(layer, id, name);
            break;
        }
        case armnn::LayerType::Pad :
        {
            const armnn::PadDescriptor& layerDescriptor =
                    static_cast<const armnn::PadDescriptor&>(descriptor);
            SerializePadLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Permute :
        {
            const armnn::PermuteDescriptor& layerDescriptor =
                    static_cast<const armnn::PermuteDescriptor&>(descriptor);
            SerializePermuteLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Pooling2d :
        {
            const armnn::Pooling2dDescriptor& layerDescriptor =
                    static_cast<const armnn::Pooling2dDescriptor&>(descriptor);
            SerializePooling2dLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Pooling3d :
        {
            const armnn::Pooling3dDescriptor& layerDescriptor =
                    static_cast<const armnn::Pooling3dDescriptor&>(descriptor);
            SerializePooling3dLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Prelu :
        {
            SerializePreluLayer(layer, name);
            break;
        }
        case armnn::LayerType::Quantize :
        {
            SerializeQuantizeLayer(layer, name);
            break;
        }
        case armnn::LayerType::QuantizedLstm:
            SerializeQuantizedLstmLayer(layer, constants, name);
            break;
        case armnn::LayerType::Reshape:
        {
            const armnn::ReshapeDescriptor &layerDescriptor =
                    static_cast<const armnn::ReshapeDescriptor &>(descriptor);
            SerializeReshapeLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Rank:
        {
            SerializeRankLayer(layer, name);
            break;
        }
        case armnn::LayerType::Reduce:
        {
            const armnn::ReduceDescriptor& layerDescriptor =
                    static_cast<const armnn::ReduceDescriptor&>(descriptor);
            SerializeReduceLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Resize:
        {
            const armnn::ResizeDescriptor& layerDescriptor =
                    static_cast<const armnn::ResizeDescriptor&>(descriptor);
            SerializeResizeLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Shape:
        {
            SerializeShapeLayer(layer, name);
            break;
        }
        case armnn::LayerType::Slice:
        {
            const armnn::SliceDescriptor& layerDescriptor =
                    static_cast<const armnn::SliceDescriptor&>(descriptor);
            SerializeSliceLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Softmax:
        {
            const armnn::SoftmaxDescriptor& layerDescriptor =
                    static_cast<const armnn::SoftmaxDescriptor&>(descriptor);
            SerializeSoftmaxLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::SpaceToBatchNd:
        {
            const armnn::SpaceToBatchNdDescriptor& layerDescriptor =
                    static_cast<const armnn::SpaceToBatchNdDescriptor&>(descriptor);
            SerializeSpaceToBatchNdLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::SpaceToDepth:
        {
            const armnn::SpaceToDepthDescriptor& layerDescriptor =
                    static_cast<const armnn::SpaceToDepthDescriptor&>(descriptor);
            SerializeSpaceToDepthLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Splitter:
        {
            const armnn::SplitterDescriptor& layerDescriptor =
                    static_cast<const armnn::SplitterDescriptor&>(descriptor);
            SerializeSplitterLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Stack:
        {
            const armnn::StackDescriptor& layerDescriptor =
                    static_cast<const armnn::StackDescriptor&>(descriptor);
            SerializeStackLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::StandIn:
        {
            const armnn::StandInDescriptor& layerDescriptor =
                    static_cast<const armnn::StandInDescriptor&>(descriptor);
            SerializeStandInLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::StridedSlice:
        {
            const armnn::StridedSliceDescriptor& layerDescriptor =
                    static_cast<const armnn::StridedSliceDescriptor&>(descriptor);
            SerializeStridedSliceLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::Subtraction:
        {
            SerializeSubtractionLayer(layer, name);
            break;
        }
        case armnn::LayerType::Switch:
        {
            SerializeSwitchLayer(layer, name);
            break;
        }
        case armnn::LayerType::Transpose:
        {
            const armnn::TransposeDescriptor& layerDescriptor =
                    static_cast<const armnn::TransposeDescriptor&>(descriptor);
            SerializeTransposeLayer(layer, layerDescriptor, name);
            break;
        }
        case armnn::LayerType::TransposeConvolution2d:
        {
            const armnn::TransposeConvolution2dDescriptor& layerDescriptor =
                    static_cast<const armnn::TransposeConvolution2dDescriptor&>(descriptor);
            SerializeTransposeConvolution2dLayer(layer, layerDescriptor, constants, name);
            break;
        }
        case armnn::LayerType::UnidirectionalSequenceLstm :
        {
            const armnn::UnidirectionalSequenceLstmDescriptor& layerDescriptor =
                    static_cast<const armnn::UnidirectionalSequenceLstmDescriptor&>(descriptor);
            SerializeUnidirectionalSequenceLstmLayer(layer, layerDescriptor, constants, name);
            break;
        }
        default:
        {
            throw InvalidArgumentException(
                    fmt::format("A layer of unknown type was given to the serializer. Layer name: {}; Layer Id: {}",
                                layer->GetName(),
                                id));
        }
    }
}

void ISerializer::SerializerImpl::Serialize(const INetwork& inNetwork)
{
    // Iterate through to network
    inNetwork.ExecuteStrategy(m_SerializerStrategy);
    flatbuffers::FlatBufferBuilder& fbBuilder = m_SerializerStrategy.GetFlatBufferBuilder();

    // Create FlatBuffer SerializedGraph
    auto serializedGraph = serializer::CreateSerializedGraph(
            fbBuilder,
            fbBuilder.CreateVector(m_SerializerStrategy.GetSerializedLayers()),
            fbBuilder.CreateVector(m_SerializerStrategy.GetInputIds()),
            fbBuilder.CreateVector(m_SerializerStrategy.GetOutputIds()),
            m_SerializerStrategy.GetVersionTable());

    // Serialize the graph
    fbBuilder.Finish(serializedGraph);
}


bool ISerializer::SerializerImpl::SaveSerializedToStream(std::ostream& stream)
{
    flatbuffers::FlatBufferBuilder& fbBuilder = m_SerializerStrategy.GetFlatBufferBuilder();

    auto bytesToWrite = armnn::numeric_cast<std::streamsize>(fbBuilder.GetSize());
    stream.write(reinterpret_cast<const char*>(fbBuilder.GetBufferPointer()), bytesToWrite);
    return !stream.bad();
}

} // namespace armnnSerializer
