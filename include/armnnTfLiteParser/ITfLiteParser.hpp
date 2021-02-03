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

class TfLiteParserImpl;
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
    armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile);

    /// Create the network from a flatbuffers binary
    armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t> & binaryContent);

    /// Retrieve binding info (layer id and tensor info) for the network input identified by
    /// the given layer name and subgraph id
    BindingPointInfo GetNetworkInputBindingInfo(size_t subgraphId,
                                                const std::string& name) const;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by
    /// the given layer name and subgraph id
    BindingPointInfo GetNetworkOutputBindingInfo(size_t subgraphId,
                                                         const std::string& name) const;

    /// Return the number of subgraphs in the parsed model
    size_t GetSubgraphCount() const;

    /// Return the input tensor names for a given subgraph
    std::vector<std::string> GetSubgraphInputTensorNames(size_t subgraphId) const;

    /// Return the output tensor names for a given subgraph
    std::vector<std::string> GetSubgraphOutputTensorNames(size_t subgraphId) const;

private:
    ITfLiteParser(const armnn::Optional<TfLiteParserOptions>& options = armnn::EmptyOptional());
    ~ITfLiteParser();

    std::unique_ptr<TfLiteParserImpl> pTfLiteParserImpl;
};

}
