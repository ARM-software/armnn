//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Serializer.hpp"
#include <armnn/ArmNN.hpp>
#include <iostream>
#include <Schema_generated.h>
#include <flatbuffers/util.h>

using namespace armnn;
namespace fb = flatbuffers;
namespace serializer = armnn::armnnSerializer;

namespace armnnSerializer
{

serializer::DataType GetFlatBufferDataType(DataType dataType)
{
    switch (dataType)
    {
        case DataType::Float32:
            return serializer::DataType::DataType_Float32;
        case DataType::Float16:
            return serializer::DataType::DataType_Float16;
        case DataType::Signed32:
            return serializer::DataType::DataType_Signed32;
        case DataType::QuantisedAsymm8:
            return serializer::DataType::DataType_QuantisedAsymm8;
        case DataType::Boolean:
            return serializer::DataType::DataType_Boolean;
        default:
            return serializer::DataType::DataType_Float16;
    }
}

// Build FlatBuffer for Input Layer
void Serializer::VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferInputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Input);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferInputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                flatBufferInputBaseLayer,
                                                                                id);

    // Push layer Guid to outputIds.
    m_inputIds.push_back(layer->GetGuid());

    // Create the FlatBuffer InputLayer
    auto flatBufferInputLayer = serializer::CreateInputLayer(m_flatBufferBuilder, flatBufferInputBindableBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferInputLayer.o, serializer::Layer::Layer_InputLayer);
}

// Build FlatBuffer for Output Layer
void Serializer::VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferOutputBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Output);

    // Create FlatBuffer BindableBaseLayer
    auto flatBufferOutputBindableBaseLayer = serializer::CreateBindableLayerBase(m_flatBufferBuilder,
                                                                                 flatBufferOutputBaseLayer,
                                                                                 id);
    // Push layer Guid to outputIds.
    m_outputIds.push_back(layer->GetGuid());

    // Create the FlatBuffer OutputLayer
    auto flatBufferOutputLayer = serializer::CreateOutputLayer(m_flatBufferBuilder, flatBufferOutputBindableBaseLayer);
    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferOutputLayer.o, serializer::Layer::Layer_OutputLayer);
}

// Build FlatBuffer for Addition Layer
void Serializer::VisitAdditionLayer(const IConnectableLayer* layer, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferAdditionBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Addition);

    // Create the FlatBuffer AdditionLayer
    auto flatBufferAdditionLayer = serializer::CreateAdditionLayer(m_flatBufferBuilder, flatBufferAdditionBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferAdditionLayer.o, serializer::Layer::Layer_AdditionLayer);
}

// Build FlatBuffer for Multiplication Layer
void Serializer::VisitMultiplicationLayer(const IConnectableLayer* layer, const char* name)
{
    // Create FlatBuffer BaseLayer
    auto flatBufferMultiplicationBaseLayer = CreateLayerBase(layer, serializer::LayerType::LayerType_Multiplication);

    // Create the FlatBuffer MultiplicationLayer
    auto flatBufferMultiplicationLayer =
        serializer::CreateMultiplicationLayer(m_flatBufferBuilder, flatBufferMultiplicationBaseLayer);

    // Add the AnyLayer to the FlatBufferLayers
    CreateAnyLayer(flatBufferMultiplicationLayer.o, serializer::Layer::Layer_MultiplicationLayer);
}

void Serializer::Serialize(const INetwork& inNetwork)
{
    // Iterate through to network
    inNetwork.Accept(*this);

    // Create FlatBuffer SerializedGraph
    auto serializedGraph = serializer::CreateSerializedGraph(m_flatBufferBuilder,
                                                             m_flatBufferBuilder.CreateVector(m_serializedLayers),
                                                             m_flatBufferBuilder.CreateVector(m_inputIds),
                                                             m_flatBufferBuilder.CreateVector(m_outputIds));

    // Serialize the graph
    m_flatBufferBuilder.Finish(serializedGraph);
}

bool Serializer::SaveSerializedToStream(std::ostream& stream)
{
    stream.write(reinterpret_cast<const char*>(m_flatBufferBuilder.GetBufferPointer()), m_flatBufferBuilder.GetSize());
    return !stream.bad();
}

fb::Offset<serializer::LayerBase> Serializer::CreateLayerBase(const IConnectableLayer* layer,
                                                              const serializer::LayerType layerType)
{
    std::vector<fb::Offset<serializer::InputSlot>> inputSlots = CreateInputSlots(layer);
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots = CreateOutputSlots(layer);

    return serializer::CreateLayerBase(m_flatBufferBuilder,
                                       layer->GetGuid(),
                                       m_flatBufferBuilder.CreateString(layer->GetName()),
                                       layerType,
                                       m_flatBufferBuilder.CreateVector(inputSlots),
                                       m_flatBufferBuilder.CreateVector(outputSlots));
}

void Serializer::CreateAnyLayer(const flatbuffers::Offset<void>& layer, const serializer::Layer serializerLayer)
{
    auto anyLayer = armnn::armnnSerializer::CreateAnyLayer(m_flatBufferBuilder,
                                                           serializerLayer,
                                                           layer);
    m_serializedLayers.push_back(anyLayer);
}

std::vector<fb::Offset<serializer::InputSlot>> Serializer::CreateInputSlots(const IConnectableLayer* layer)
{
    std::vector<fb::Offset <serializer::InputSlot>> inputSlots;

    // Get the InputSlots
    for (unsigned int slotIndex = 0; slotIndex<layer->GetNumInputSlots(); ++slotIndex)
    {
        const IInputSlot& inputSlot = layer->GetInputSlot(slotIndex);

        // Get the Connection for the InputSlot
        const IOutputSlot* connection = inputSlot.GetConnection();

        // Create FlatBuffer Connection
        serializer::Connection conn(connection->GetOwningLayerGuid(), connection->CalculateIndexOnOwner());
        // Create FlatBuffer InputSlot
        inputSlots.push_back(serializer::CreateInputSlot(m_flatBufferBuilder, slotIndex, &conn));
    }
    return inputSlots;
}

std::vector<fb::Offset<serializer::OutputSlot>> Serializer::CreateOutputSlots(const IConnectableLayer* layer)
{
    std::vector<fb::Offset<serializer::OutputSlot>> outputSlots;

    // Get the OutputSlots
    for (unsigned int slotIndex = 0; slotIndex < layer->GetNumOutputSlots(); ++slotIndex)
    {
        const IOutputSlot& outputSlot = layer->GetOutputSlot(slotIndex);
        const TensorInfo& tensorInfo = outputSlot.GetTensorInfo();

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

} //namespace armnnSerializer
