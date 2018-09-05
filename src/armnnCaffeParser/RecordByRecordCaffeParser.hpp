//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "caffe/proto/caffe.pb.h"

#include "CaffeParser.hpp"



namespace armnnCaffeParser
{

class NetParameterInfo;
class LayerParameterInfo;


class RecordByRecordCaffeParser : public CaffeParserBase
{
public:

    /// Create the network from a protobuf binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    RecordByRecordCaffeParser();

private:
    void ProcessLayers(const NetParameterInfo& netParameterInfo,
                       std::vector<LayerParameterInfo>& layerInfo,
                       const std::vector<std::string>& m_RequestedOutputs,
                       std::vector<const LayerParameterInfo*>& sortedNodes);
    armnn::INetworkPtr LoadLayers(std::ifstream& ifs,
                                  std::vector<const LayerParameterInfo *>& sortedNodes,
                                  const NetParameterInfo& netParameterInfo);
    std::vector<const LayerParameterInfo*> GetInputs(
        const LayerParameterInfo& layerParam);

    std::map<std::string, const LayerParameterInfo*> m_CaffeLayersByTopName;
    std::vector<std::string> m_RequestedOutputs;
};

} // namespace armnnCaffeParser

