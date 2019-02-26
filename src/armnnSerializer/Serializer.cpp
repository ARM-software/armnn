//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Serializer.hpp"

#include "SerializerUtils.hpp"

#include <armnn/ArmNN.hpp>

#include <iostream>

#include <ArmnnSchema_generated.h>

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

// Build FlatBuffer for Multiplication Layer
void SerializerVisitor::VisitMultiplicationLayer(const armnn::IConnectableLayer* layer, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferMultiplicationBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Multiplication);

    // Create the FlatBuffer MultiplicationLayer
    auto flatBufferMultiplicationLayer =
        serializer::CreateMultiplicationLayer(m_flatBufferBuilder, flatBufferMultiplicationBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferMultiplicationLayer.o, serializer::Layer::Layer_MultiplicationLayer);
}

void SerializerVisitor::VisitPermuteLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::PermuteDescriptor& permuteDescriptor,
                                          const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferPermuteBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Permute);

    std::vector<unsigned int> dimMappings;
    for (auto& v: permuteDescriptor.m_DimMappings)
    {
        dimMappings.push_back(v);
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

fb::Offset<serializer::LayerBase> SerializerVisitor::CreateLayerBase(const IConnectableLayer* layer,
                                                                     const serializer::LayerType layerType)
{
    std::vector<fb::Offset<serializer::InputSlot>> inputSlots = CreateInputSlots(layer);
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots = CreateOutputSlots(layer);

    return serializer::CreateLayerBase(m_flatBufferBuilder,
                                       GetSerializedId(layer->GetGuid()),
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
