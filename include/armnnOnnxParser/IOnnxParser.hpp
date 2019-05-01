//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Tensor.hpp>

#include <memory>
#include <vector>
#include <map>

namespace armnnOnnxParser
{

using BindingPointInfo = armnn::BindingPointInfo;

class IOnnxParser;
using IOnnxParserPtr = std::unique_ptr<IOnnxParser, void(*)(IOnnxParser* parser)>;

class IOnnxParser
{
public:
    static IOnnxParser* CreateRaw();
    static IOnnxParserPtr Create();
    static void Destroy(IOnnxParser* parser);

    /// Create the network from a protobuf binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile) = 0;

    /// Create the network from a protobuf text file on disk
    virtual armnn::INetworkPtr CreateNetworkFromTextFile(const char* graphFile) = 0;

    /// Create the network directly from protobuf text in a string. Useful for debugging/testing
    virtual armnn::INetworkPtr CreateNetworkFromString(const std::string& protoText) = 0;

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const = 0;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const = 0;

  protected:
      virtual ~IOnnxParser() {};
  };

  }
