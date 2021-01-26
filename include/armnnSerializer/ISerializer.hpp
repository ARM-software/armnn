//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/INetwork.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Types.hpp"

namespace armnnSerializer
{

class ISerializer;
using ISerializerPtr = std::unique_ptr<ISerializer, void(*)(ISerializer* serializer)>;

class ISerializer
{
public:
    static ISerializer* CreateRaw();
    static ISerializerPtr Create();
    static void Destroy(ISerializer* serializer);

    /// Serializes the network to ArmNN SerializedGraph.
    /// @param [in] inNetwork The network to be serialized.
    void Serialize(const armnn::INetwork& inNetwork);

    /// Serializes the SerializedGraph to the stream.
    /// @param [stream] the stream to save to
    /// @return true if graph is Serialized to the Stream, false otherwise
    bool SaveSerializedToStream(std::ostream& stream);

private:
    ISerializer();
    ~ISerializer();

    class SerializerImpl;
    std::unique_ptr<SerializerImpl> pSerializerImpl;
};

} //namespace armnnSerializer
