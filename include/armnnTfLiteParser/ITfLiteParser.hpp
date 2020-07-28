//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnn/Optional.hpp"

#include <memory>
#include <map>
#include <vector>

namespace armnnTfLiteParser
{

using BindingPointInfo = armnn::BindingPointInfo;

class ITfLiteParser;
using ITfLiteParserPtr = std::unique_ptr<ITfLiteParser, void(*)(ITfLiteParser* parser)>;

class ITfLiteParser
{
public:
    struct TfLiteParserOptions
    {
        TfLiteParserOptions()
            : m_StandInLayerForUnsupported(false),
              m_InferAndValidate(false) {}

        bool m_StandInLayerForUnsupported;
        bool m_InferAndValidate;
    };

    static ITfLiteParser* CreateRaw(const armnn::Optional<TfLiteParserOptions>& options = armnn::EmptyOptional());
    static ITfLiteParserPtr Create(const armnn::Optional<TfLiteParserOptions>& options = armnn::EmptyOptional());
    static void Destroy(ITfLiteParser* parser);

    /// Create the network from a flatbuffers binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile) = 0;

    /// Create the network from a flatbuffers binary
    virtual armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t> & binaryContent) = 0;

    /// Retrieve binding info (layer id and tensor info) for the network input identified by
    /// the given layer name and subgraph id
    virtual BindingPointInfo GetNetworkInputBindingInfo(size_t subgraphId,
                                                        const std::string& name) const = 0;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by
    /// the given layer name and subgraph id
    virtual BindingPointInfo GetNetworkOutputBindingInfo(size_t subgraphId,
                                                         const std::string& name) const = 0;

    /// Return the number of subgraphs in the parsed model
    virtual size_t GetSubgraphCount() const = 0;

    /// Return the input tensor names for a given subgraph
    virtual std::vector<std::string> GetSubgraphInputTensorNames(size_t subgraphId) const = 0;

    /// Return the output tensor names for a given subgraph
    virtual std::vector<std::string> GetSubgraphOutputTensorNames(size_t subgraphId) const = 0;

protected:
    virtual ~ITfLiteParser() {};
};

}
