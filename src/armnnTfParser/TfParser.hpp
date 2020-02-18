//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnnTfParser/ITfParser.hpp"

#include "armnn/Types.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"

#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <utility>
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

class ParsedTfOperation;
using ParsedTfOperationPtr = std::unique_ptr<ParsedTfOperation>;

///
/// WithOutputTensorIndex wraps a value and an index. The purpose of
/// this template is to signify that, in Tensorflow, the input name of
/// a layer has the convention of 'inputTensorName:#index', where the
/// #index can be omitted and it implicitly means the 0 output of
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
    /// Creates the network from a protobuf text file on the disk.
    virtual armnn::INetworkPtr CreateNetworkFromTextFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Creates the network from a protobuf binary file on the disk.
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Creates the network directly from protobuf text in a string. Useful for debugging/testing.
    virtual armnn::INetworkPtr CreateNetworkFromString(
        const char* protoText,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

    /// Retrieves binding info (layer id and tensor info) for the network input identified by the given layer name.
    virtual BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const override;

    /// Retrieves binding info (layer id and tensor info) for the network output identified by the given layer name.
    virtual BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const override;

public:
    TfParser();

private:
    template <typename T>
    friend class ParsedConstTfOperation;
    friend class ParsedMatMulTfOperation;
    friend class ParsedMulTfOperation;

    /// Parses a GraphDef loaded into memory from one of the other CreateNetwork*.
    armnn::INetworkPtr CreateNetworkFromGraphDef(const tensorflow::GraphDef& graphDef,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// Sets up variables and then performs BFS to parse all nodes.
    void LoadGraphDef(const tensorflow::GraphDef& graphDef);

    /// Parses a given node, assuming nodes before it in the graph have been done.
    void LoadNodeDef(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);

    /// Handling identity layers as the input for Conv2D layer.
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

    /// Checks if there is a pre-parsed const tensor available with the given name and Type.
    template<typename Type>
    bool HasParsedConstTensor(const std::string & nodeName) const;
    template<typename Type>
    bool HasParsedConstTensor(ParsedTfOperation* parsedTfOpPtr) const;

    unsigned int GetConstInputIndex(const std::vector<OutputOfParsedTfOperation>& inputs);

    ParsedTfOperationPtr ParseAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseAddN(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseBiasAdd(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseConv2D(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseDepthwiseConv2D(const tensorflow::NodeDef& nodeDef,
                                              const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseExpandDims(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseFusedBatchNorm(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseConcat(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseIdentity(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseLrn(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMatMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMean(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMul(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParsePlaceholder(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRealDiv(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRelu(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRelu6(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseReshape(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseResizeBilinear(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseRsqrt(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseShape(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSqueeze(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSigmoid(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSoftmax(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSoftplus(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSplit(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseStridedSlice(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseTanh(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMaxPool(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseAvgPool(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParsePooling2d(const tensorflow::NodeDef& nodeDef,
                                        const tensorflow::GraphDef& graphDef,
                                        armnn::PoolingAlgorithm pooltype);
    ParsedTfOperationPtr ParseEqual(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMaximum(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseMinimum(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseGather(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseGreater(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParsePad(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseSub(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseStack(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr ParseTranspose(const tensorflow::NodeDef& nodeDef, const tensorflow::GraphDef& graphDef);
    ParsedTfOperationPtr AddActivationLayer(const tensorflow::NodeDef& nodeDef, armnn::ActivationDescriptor& desc);
    ParsedTfOperationPtr AddAdditionLayer(const tensorflow::NodeDef& nodeDef, bool isBiasAdd = false);
    ParsedTfOperationPtr AddRealDivLayer(const tensorflow::NodeDef& nodeDef);
    ParsedTfOperationPtr AddMaximumLayer(const tensorflow::NodeDef& nodeDef);

private:
    armnn::IConnectableLayer* AddMultiplicationLayer(const tensorflow::NodeDef& nodeDef);

    armnn::IConnectableLayer* AddFullyConnectedLayer(const tensorflow::NodeDef& matMulNodeDef,
        const tensorflow::NodeDef* addNodeDef, const char* armnnLayerName);

    bool IsSupportedLeakyReluPattern(const tensorflow::NodeDef& mulNodeDef,
                                    size_t alphaLayerIndex,
                                    const OutputOfParsedTfOperation& otherOp,
                                    armnn::IOutputSlot** outputOfLeakyRelu,
                                    armnn::ActivationDescriptor & desc);

    std::pair<armnn::IOutputSlot*, armnn::IOutputSlot*> ProcessElementwiseInputSlots(
            const tensorflow::NodeDef& nodeDef, const std::string& layerName);

    ParsedTfOperationPtr ProcessComparisonLayer(
        armnn::IOutputSlot* input0Slot,
        armnn::IOutputSlot* input1Slot,
        armnn::IConnectableLayer* const layer,
        const tensorflow::NodeDef& nodeDef);

    ParsedTfOperationPtr ProcessElementwiseLayer(
            armnn::IOutputSlot* input0Slot,
            armnn::IOutputSlot* input1Slot,
            armnn::IConnectableLayer* const layer,
            const tensorflow::NodeDef& nodeDef);

    armnn::IConnectableLayer* CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            armnn::IOutputSlot* input0Slot,
            armnn::IOutputSlot* input1Slot,
            const std::string& layerName);

    armnn::IConnectableLayer* CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            const OutputOfParsedTfOperation& opOne,
            const OutputOfParsedTfOperation& opTwo,
            unsigned int numberOfAddition);

    armnn::IConnectableLayer* CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            armnn::IConnectableLayer* layerOne,
            armnn::IConnectableLayer* layerTwo,
            unsigned int numberOfAddition,
            unsigned long numberOfLayersToConnect,
            bool isOdd);

    armnn::IConnectableLayer* CreateAdditionLayer(
            const tensorflow::NodeDef& nodeDef,
            const OutputOfParsedTfOperation& op,
            armnn::IConnectableLayer* layer);

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

    /// The network we're building. Gets cleared after it is passed to the user.
    armnn::INetworkPtr m_Network;

    using OperationParsingFunction = ParsedTfOperationPtr(TfParser::*)(const tensorflow::NodeDef& nodeDef,
                                                                 const tensorflow::GraphDef& graphDef);

    /// Map of TensorFlow operation names to parsing member functions.
    static const std::map<std::string, OperationParsingFunction> ms_OperationNameToParsingFunctions;

    static const std::list<std::string> m_ControlInputs;

    std::map<std::string, armnn::TensorShape> m_InputShapes;
    std::vector<std::string> m_RequestedOutputs;

    /// Map of nodes extracted from the GraphDef to speed up parsing.
    std::unordered_map<std::string, const tensorflow::NodeDef*> m_NodesByName;

    std::unordered_map<std::string, ParsedTfOperationPtr> m_ParsedTfOperations;

    /// Maps input layer names to their corresponding ids and tensor info.
    std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

    /// Maps output layer names to their corresponding ids and tensor info.
    std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;
};

}
