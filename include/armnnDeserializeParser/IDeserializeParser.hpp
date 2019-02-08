//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"

#include <memory>
#include <map>
#include <vector>

namespace armnnDeserializeParser
{

using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

class IDeserializeParser;
using IDeserializeParserPtr = std::unique_ptr<IDeserializeParser, void(*)(IDeserializeParser* parser)>;

class IDeserializeParser
{
public:
    static IDeserializeParser* CreateRaw();
    static IDeserializeParserPtr Create();
    static void Destroy(IDeserializeParser* parser);

    /// Create the network from a flatbuffers binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile) = 0;

    /// Create the network from a flatbuffers binary
    virtual armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent) = 0;


    /// Retrieve binding info (layer id and tensor info) for the network input identified by
    /// the given layer name and layers id
    virtual BindingPointInfo GetNetworkInputBindingInfo(unsigned int layerId,
                                                        const std::string& name) const = 0;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by
    /// the given layer name and layers id
    virtual BindingPointInfo GetNetworkOutputBindingInfo(unsigned int layerId,
                                                         const std::string& name) const = 0;

protected:
    virtual ~IDeserializeParser() {};

};
}