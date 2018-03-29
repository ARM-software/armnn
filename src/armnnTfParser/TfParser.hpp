//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "armnnTfParser/ITfParser.hpp"

#include "armnn/Types.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace armnn
{
class TensorInfo;
}

namespace tensorflow
{
class GraphDef;
class NodeDef;
}

namespace armnnTfParser
{

using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

class ParsedTfOperation;
using ParsedTfOperationPtr = std::unique_ptr<ParsedTfOperation>;

///
/// WithOutputTensorIndex wraps a value and an index. The purpose of
/// this template is to signify that in Tensorflow the input name of
/// a layer has the convention of 'inputTensorName:#index' where the
/// #index can be omitted and it implicitly means the 0. output of
/// the referenced layer. By supporting this notation we can handle
/// layers with multiple outputs, such as Split.
///
template <typename T>
struct WithOutputTensorIndex
{
    T                m_IndexedValue;
    unsigned int     m_Index;

    WithOutputTensorIndex(const T & value, unsigned int index)
    : m_IndexedValue{value}
    , m_Index{index} {}

    WithOutputTensorIndex(T && value, unsigned int index)
    : m_IndexedValue{value}
    , m_Index{index} {}
};

using OutputOfParsedTfOperation = WithOutputTensorIndex<ParsedTfOperation *>;
using OutputOfConstNodeDef = WithOutputTensorIndex<const tensorflow::NodeDef*>;
using OutputId = WithOutputTensorIndex<std::string>;

class TfParser : public ITfParser
{
public:
    /// Create the network from a protobuf text file on disk
    virtual armnn::INetworkPtr CreateNetworkFromTextFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Create the network from a protobuf binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Create the network directly from protobuf text in a string. Useful for debugging/testing
    virtual armnn::INetworkPtr CreateNetworkFromString(
        const char* protoText,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;

public:
    TfParser();

private:
    template <typename T>
    friend class ParsedConstTfOperation;
    friend class ParsedMatMulTfOperation;

    /// Parses a GraphDef loaded into memory from one of the other CreateNetwork*
    armnn::INetworkPtr CreateNetworkFromGraphDef(const tensorflow::GraphDef& graphDef,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// sets up variables and then performs BFS to parse all nodes
    void LoadGraphDef(const tensorflow::GraphDef& graphDef);

    /// parses a given node, assuming nodes before it in graph have been done
    void LoadNodeDef(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);

    /// Handling identity layers as the input for Conv2D layer
    const tensorflow::NodeDef* ResolveIdentityNode(const tensorflow::NodeDef* nodeDef);
    /// Finds the nodes connected as inputs of the given node in the graph.
    std::vector<OutputOfConstNodeDef> GetTfInputNodes(const tensorflow::NodeDef& nodeDef) const;
    /// Finds the IParsedTfOperations for the nodes connected as inputs of the given node in the graph,
    /// and throws an exception if the number of inputs does not match the expected one.
    /// This will automatically resolve any identity nodes. The result vector contains the parsed operation
    /// together with the output tensor index to make the connection unambiguous.
    std::vector<OutputOfParsedTfOperation> GetInputParsedTfOperationsChecked(const tensorflow::NodeDef& nodeDef,
                                                                             std::size_t expectedNumInputs);

    ParsedTfOperationPtr ParseConst(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);

    /// Checks if there is a pre-parsed const tensor is available with the given name and Type
    template<typename Type>
    bool HasParsedConstTensor(const std::string & nodeName) const;

    ParsedTfOperationPtr ParseAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseBiasAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseConv2D(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseDepthwiseConv2D(const tensorflow::NodeDef& nodeDef,const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseFusedBatchNorm(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseConcat(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseIdentity(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseLrn(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMatMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParsePlaceholder(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRelu(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRelu6(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseReshape(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseResizeBilinear(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseShape(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSqueeze(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSigmoid(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSoftmax(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSoftplus(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseTanh(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMaxPool(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseAvgPool(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParsePooling2d(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef,
        armnn::PoolingAlgorithm pooltype);
    ParsedTfOperationPtr AddActivationLayer(const tensorflow::NodeDef& nodeDef, armnn::ActivationDescriptor& desc);
    ParsedTfOperationPtr AddAdditionLayer(const tensorflow::NodeDef& nodeDef, bool isBiasAdd = false);
    armnn::IConnectableLayer* AddFullyConnectedLayer(const tensorflow::NodeDef& matMulNodeDef,
        const tensorflow::NodeDef* addNodeDef, const char* armnnLayerName);

    static std::pair<armnn::LayerBindingId, armnn::TensorInfo> GetBindingInfo(const std::string& layerName,
        const char* bindingPointDesc,
        const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

    void TrackInputBinding(armnn::IConnectableLayer* layer,
        armnn::LayerBindingId id,
        const armnn::TensorInfo& tensorInfo);

    void TrackOutputBinding(armnn::IConnectableLayer* layer,
        armnn::LayerBindingId id,
        const armnn::TensorInfo& tensorInfo);

    static void TrackBindingPoint(armnn::IConnectableLayer* layer, armnn::LayerBindingId id,
        const armnn::TensorInfo& tensorInfo,
        const char* bindingPointDesc,
        std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

    void Cleanup();

    /// The network we're building. Gets cleared after it is passed to the user
    armnn::INetworkPtr m_Network;

    using OperationParsingFunction = ParsedTfOperationPtr(TfParser::*)(const tensorflow::NodeDef& nodeDef,
                                                                 const tensorflow::GraphDef& graphDef);

    /// map of TensorFlow operation names to parsing member functions
    static const std::map<std::string, OperationParsingFunction> ms_OperationNameToParsingFunctions;

    std::map<std::string, armnn::TensorShape> m_InputShapes;
    std::vector<std::string> m_RequestedOutputs;

    /// map of nodes extracted from the GraphDef to speed up parsing
    std::unordered_map<std::string, const tensorflow::NodeDef*> m_NodesByName;

    std::unordered_map<std::string, ParsedTfOperationPtr> m_ParsedTfOperations;

    /// maps input layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

    /// maps output layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;
};
}
