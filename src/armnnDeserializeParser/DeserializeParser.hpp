//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/INetwork.hpp"
#include "armnnDeserializeParser/IDeserializeParser.hpp"
#include <Schema_generated.h>

namespace armnnDeserializeParser
{
class DeserializeParser : public IDeserializeParser
{
public:
    // Shorthands for deserializer types
    using GraphPtr = const armnn::armnnSerializer::SerializedGraph *;
    using TensorRawPtr = const armnn::armnnSerializer::TensorInfo *;
    using TensorRawPtrVector = std::vector<TensorRawPtr>;
    using LayerRawPtr = const armnn::armnnSerializer::LayerBase *;
    using LayerBaseRawPtr = const armnn::armnnSerializer::LayerBase *;
    using LayerBaseRawPtrVector = std::vector<LayerBaseRawPtr>;

public:

    /// Create the network from a flatbuffers binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile) override;

    virtual armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent) override;

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name
    virtual BindingPointInfo GetNetworkInputBindingInfo(unsigned int layerId,
                                                        const std::string& name) const override;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name
    virtual BindingPointInfo GetNetworkOutputBindingInfo(unsigned int layerId,
                                                         const std::string& name) const override;

    DeserializeParser();
    ~DeserializeParser() {}

public:
    // testable helpers
    static GraphPtr LoadGraphFromFile(const char* fileName, std::string& fileContent);
    static GraphPtr LoadGraphFromBinary(const uint8_t* binaryContent, size_t len);
    static TensorRawPtrVector GetInputs(const GraphPtr& graph, unsigned int layerIndex);
    static TensorRawPtrVector GetOutputs(const GraphPtr& graph, unsigned int layerIndex);
    static LayerBaseRawPtrVector GetGraphInputs(const GraphPtr& graphPtr);
    static LayerBaseRawPtrVector GetGraphOutputs(const GraphPtr& graphPtr);
    static LayerBaseRawPtr GetBaseLayer(const GraphPtr& graphPtr, unsigned int layerIndex);
    static int32_t GetBindingLayerInfo(const GraphPtr& graphPtr, unsigned int layerIndex);

private:
    // No copying allowed until it is wanted and properly implemented
    DeserializeParser(const DeserializeParser&) = delete;
    DeserializeParser& operator=(const DeserializeParser&) = delete;

    /// Create the network from an already loaded flatbuffers graph
    armnn::INetworkPtr CreateNetworkFromGraph();

    // signature for the parser functions
    using LayerParsingFunction = void(DeserializeParser::*)(unsigned int layerIndex);

    void ParseUnsupportedLayer(unsigned int serializeGraphIndex);
    void ParseAdd(unsigned int serializeGraphIndex);
    void ParseMultiplication(unsigned int serializeGraphIndex);

    void RegisterOutputSlotOfConnection(uint32_t connectionIndex, armnn::IOutputSlot* slot);
    void RegisterInputSlotOfConnection(uint32_t connectionIndex, armnn::IInputSlot* slot);
    void RegisterInputSlots(uint32_t layerIndex,
                            armnn::IConnectableLayer* layer);
    void RegisterOutputSlots(uint32_t layerIndex,
                             armnn::IConnectableLayer* layer);
    void ResetParser();

    void SetupInputLayers();
    void SetupOutputLayers();

    /// The network we're building. Gets cleared after it is passed to the user
    armnn::INetworkPtr                    m_Network;
    GraphPtr                              m_Graph;
    std::vector<LayerParsingFunction>     m_ParserFunctions;

    /// This holds the data of the file that was read in from CreateNetworkFromBinaryFile
    /// Needed for m_Graph to point to
    std::string                           m_FileContent;

    /// A mapping of an output slot to each of the input slots it should be connected to
    /// The outputSlot is from the layer that creates this tensor as one of its outputs
    /// The inputSlots are from the layers that use this tensor as one of their inputs
    struct Slots
    {
        armnn::IOutputSlot* outputSlot;
        std::vector<armnn::IInputSlot*> inputSlots;

        Slots() : outputSlot(nullptr) { }
    };
    typedef std::vector<Slots> Connection;
    std::vector<Connection>   m_GraphConnections;
};

}
