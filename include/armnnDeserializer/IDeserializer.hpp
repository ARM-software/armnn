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

namespace armnnDeserializer
{
struct BindingPointInfo
{
    armnn::LayerBindingId   m_BindingId;
    armnn::TensorInfo       m_TensorInfo;
};

class IDeserializer;
using IDeserializerPtr = std::unique_ptr<IDeserializer, void(*)(IDeserializer* parser)>;

class IDeserializer
{
public:
    static IDeserializer* CreateRaw();
    static IDeserializerPtr Create();
    static void Destroy(IDeserializer* parser);

    /// Create an input network from binary file contents
    armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent);

    /// Create an input network from a binary input stream
    armnn::INetworkPtr CreateNetworkFromBinary(std::istream& binaryContent);

    /// Retrieve binding info (layer id and tensor info) for the network input identified by
    /// the given layer name and layers id
    BindingPointInfo GetNetworkInputBindingInfo(unsigned int layerId, const std::string& name) const;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by
    /// the given layer name and layers id
    BindingPointInfo GetNetworkOutputBindingInfo(unsigned int layerId, const std::string& name) const;

private:
    IDeserializer();
    ~IDeserializer();

    class DeserializerImpl;
    std::unique_ptr<DeserializerImpl> pDeserializerImpl;
};
} //namespace armnnDeserializer