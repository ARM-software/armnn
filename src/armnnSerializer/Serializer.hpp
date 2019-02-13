//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ILayerVisitor.hpp>
#include <armnn/LayerVisitorBase.hpp>

#include <armnnSerializer/ISerializer.hpp>

#include <unordered_map>

#include <Schema_generated.h>

namespace armnnSerializer
{

class SerializerVisitor : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
{
public:
    SerializerVisitor() : m_layerId(0) {};
    ~SerializerVisitor() {}

    flatbuffers::FlatBufferBuilder& GetFlatBufferBuilder()
    {
        return m_flatBufferBuilder;
    }

    std::vector<unsigned int>& GetInputIds()
    {
        return m_inputIds;
    }

    std::vector<unsigned int>& GetOutputIds()
    {
        return m_outputIds;
    }

    std::vector<flatbuffers::Offset<armnn::armnnSerializer::AnyLayer>>& GetSerializedLayers()
    {
        return m_serializedLayers;
    }

    void VisitAdditionLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr) override;

    void VisitInputLayer(const armnn::IConnectableLayer* layer,
                         armnn::LayerBindingId id,
                         const char* name = nullptr) override;

    void VisitOutputLayer(const armnn::IConnectableLayer* layer,
                          armnn::LayerBindingId id,
                          const char* name = nullptr) override;

    void VisitMultiplicationLayer(const armnn::IConnectableLayer* layer,
                                  const char* name = nullptr) override;

    void VisitSoftmaxLayer(const armnn::IConnectableLayer* layer,
                           const armnn::SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override;

private:

    /// Creates the Input Slots and Output Slots and LayerBase for the layer.
    flatbuffers::Offset<armnn::armnnSerializer::LayerBase> CreateLayerBase(
            const armnn::IConnectableLayer* layer,
            const armnn::armnnSerializer::LayerType layerType);

    /// Creates the serializer AnyLayer for the layer and adds it to m_serializedLayers.
    void CreateAnyLayer(const flatbuffers::Offset<void>& layer, const armnn::armnnSerializer::Layer serializerLayer);

    ///Function which maps Guid to an index
    uint32_t GetSerializedId(unsigned int guid);

    /// Creates the serializer InputSlots for the layer.
    std::vector<flatbuffers::Offset<armnn::armnnSerializer::InputSlot>> CreateInputSlots(
            const armnn::IConnectableLayer* layer);

    /// Creates the serializer OutputSlots for the layer.
    std::vector<flatbuffers::Offset<armnn::armnnSerializer::OutputSlot>> CreateOutputSlots(
            const armnn::IConnectableLayer* layer);

    /// FlatBufferBuilder to create our layers' FlatBuffers.
    flatbuffers::FlatBufferBuilder m_flatBufferBuilder;

    /// AnyLayers required by the SerializedGraph.
    std::vector<flatbuffers::Offset<armnn::armnnSerializer::AnyLayer>> m_serializedLayers;

    /// Guids of all Input Layers required by the SerializedGraph.
    std::vector<unsigned int> m_inputIds;

    /// Guids of all Output Layers required by the SerializedGraph.
    std::vector<unsigned int> m_outputIds;

    /// Mapped Guids of all Layers to match our index.
    std::unordered_map<unsigned int, uint32_t > m_guidMap;

    /// layer within our FlatBuffer index.
    uint32_t m_layerId;
};

class Serializer : public ISerializer
{
public:
    Serializer() {}
    ~Serializer() {}

    /// Serializes the network to ArmNN SerializedGraph.
    /// @param [in] inNetwork The network to be serialized.
    void Serialize(const armnn::INetwork& inNetwork) override;

    /// Serializes the SerializedGraph to the stream.
    /// @param [stream] the stream to save to
    /// @return true if graph is Serialized to the Stream, false otherwise
    bool SaveSerializedToStream(std::ostream& stream) override;

private:

    /// Visitor to contruct serialized network
    SerializerVisitor m_SerializerVisitor;
};

} //namespace armnnSerializer
