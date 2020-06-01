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

namespace armnnCaffeParser
{

using BindingPointInfo = armnn::BindingPointInfo;

class ICaffeParser;
using ICaffeParserPtr = std::unique_ptr<ICaffeParser, void(*)(ICaffeParser* parser)>;

class ICaffeParser
{
public:
    static ICaffeParser* CreateRaw();
    static ICaffeParserPtr Create();
    static void Destroy(ICaffeParser* parser);

    /// Create the network from a protobuf text file on the disk.
    virtual armnn::INetworkPtr CreateNetworkFromTextFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) = 0;

    /// Create the network from a protobuf binary file on the disk.
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) = 0;

    /// Create the network directly from protobuf text in a string. Useful for debugging/testin.g
    virtual armnn::INetworkPtr CreateNetworkFromString(
        const char* protoText,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) = 0;

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const = 0;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const = 0;

protected:
    virtual ~ICaffeParser() {};
};

}