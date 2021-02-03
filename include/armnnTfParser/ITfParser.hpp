//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/Types.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace armnnTfParser
{

using BindingPointInfo = armnn::BindingPointInfo;

class ITfParser;
using ITfParserPtr = std::unique_ptr<ITfParser, void(*)(ITfParser* parser)>;

/// Parses a directed acyclic graph from a tensorflow protobuf file.
class ITfParser
{
public:
    static ITfParser* CreateRaw();
    static ITfParserPtr Create();
    static void Destroy(ITfParser* parser);

    /// Create the network from a protobuf text file on the disk.
    armnn::INetworkPtr CreateNetworkFromTextFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// Create the network from a protobuf binary file on the disk.
    armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// Create the network directly from protobuf text in a string. Useful for debugging/testing.
    armnn::INetworkPtr CreateNetworkFromString(
        const char* protoText,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name.
    BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name.
    BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const;

private:
    template <typename T>
    friend class ParsedConstTfOperation;
    friend class ParsedMatMulTfOperation;
    friend class ParsedMulTfOperation;
    friend class ParsedTfOperation;
    friend class SingleLayerParsedTfOperation;
    friend class DeferredSingleLayerParsedTfOperation;
    friend class ParsedIdentityTfOperation;

    template <template<typename> class OperatorType, typename T>
    friend struct MakeTfOperation;


    ITfParser();
    ~ITfParser();

    struct TfParserImpl;
    std::unique_ptr<TfParserImpl> pTfParserImpl;
};

}
