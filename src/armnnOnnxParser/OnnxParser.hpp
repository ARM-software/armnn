//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnnOnnxParser/IOnnxParser.hpp"
#include "google/protobuf/repeated_field.h"
#include <unordered_map>

#include <onnx/onnx.pb.h>


namespace armnn
{
class TensorInfo;
enum class ActivationFunction;
}

namespace armnnOnnxParser
{

using ModelPtr = std::unique_ptr<onnx::ModelProto>;

class OnnxParserImpl
{

using OperationParsingFunction = void(OnnxParserImpl::*)(const onnx::NodeProto& NodeProto);

public:

    using GraphPtr = std::unique_ptr<onnx::GraphProto>;

    /// Create the network from a protobuf binary file on disk
    armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile);

    /// Create the network from a protobuf binary file on disk, with inputShapes specified
    armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile,
                                                   const std::map<std::string, armnn::TensorShape>& inputShapes);

    /// Create the network from a protobuf binary
    armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent);

    /// Create the network from a protobuf binary, with inputShapes specified
    armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t>& binaryContent,
                                               const std::map<std::string, armnn::TensorShape>& inputShapes);

    /// Create the network from a protobuf text file on disk
    armnn::INetworkPtr CreateNetworkFromTextFile(const char* graphFile);

    /// Create the network from a protobuf text file on disk, with inputShapes specified
    armnn::INetworkPtr CreateNetworkFromTextFile(const char* graphFile,
                                                 const std::map<std::string, armnn::TensorShape>& inputShapes);

    /// Create the network directly from protobuf text in a string. Useful for debugging/testing
    armnn::INetworkPtr CreateNetworkFromString(const std::string& protoText);

     /// Create the network directly from protobuf text in a string, with inputShapes specified.
     /// Useful for debugging/testing
    armnn::INetworkPtr CreateNetworkFromString(const std::string& protoText,
                                               const std::map<std::string, armnn::TensorShape>& inputShapes);

    /// Retrieve binding info (layer id and tensor info) for the network input identified by the given layer name
    BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) const;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by the given layer name
    BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) const;

public:

    OnnxParserImpl();
    ~OnnxParserImpl() = default;

    static ModelPtr LoadModelFromBinary(const std::vector<uint8_t>& binaryContent);
    static ModelPtr LoadModelFromBinaryFile(const char * fileName);
    static ModelPtr LoadModelFromTextFile(const char * fileName);
    static ModelPtr LoadModelFromString(const std::string& inputString);

    /// Retrieve inputs names
    static std::vector<std::string> GetInputs(ModelPtr& model);

    /// Retrieve outputs names
    static std::vector<std::string> GetOutputs(ModelPtr& model);

    /// Retrieve version in X.Y.Z form
    static const std::string GetVersion();

private:

    /// Parses a ModelProto loaded into memory from one of the other CreateNetwork*
    armnn::INetworkPtr CreateNetworkFromModel(onnx::ModelProto& model);

    /// Parse every node and make the connection between the resulting tensors
    void LoadGraph();

    void SetupInfo(const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto >* list);

    std::vector<armnn::TensorInfo> ComputeOutputInfo(
        std::vector<std::string> outNames,
        const armnn::IConnectableLayer* layer,
        std::vector<armnn::TensorShape> inputShapes,
        const onnx::TensorProto::DataType& type = onnx::TensorProto::FLOAT);

    void DetectFullyConnected();

    template <typename Location>
    void GetInputAndParam(const onnx::NodeProto& node,
                          std::string* inputName,
                          std::string* constName,
                          const Location& location);

    template <typename Location>
    void To1DTensor(const std::string &name, const Location& location);

    //Broadcast Preparation functions
    std::pair<std::string, std::string> AddPrepareBroadcast(const std::string& input0, const std::string& input1);
    void PrependForBroadcast(const std::string& outputName, const std::string& input0, const std::string& input1);

    void AddConvLayerWithDepthwiseConv(const onnx::NodeProto& node, const armnn::Convolution2dDescriptor& convDesc);
    void AddFullyConnected(const onnx::NodeProto& matmulNode, const onnx::NodeProto* addNode = nullptr);
    void AddPoolingLayer(const onnx::NodeProto& nodeProto, armnn::Pooling2dDescriptor& desc);

    void CreateConstantLayer(const std::string& tensorName, const std::string& layerName);
    void CreateInt64ConstantLayer(const std::string& tensorName, const std::string& layerName);
    void CreateReshapeLayer(const std::string& inputName,
                            const std::string& outputName,
                            const std::string& layerName);

    void ParseActivation(const onnx::NodeProto& nodeProto, const armnn::ActivationFunction func);
    void ParseClip(const onnx::NodeProto& nodeProto);
    void ParseSigmoid(const onnx::NodeProto& nodeProto);
    void ParseTanh(const onnx::NodeProto& nodeProto);
    void ParseRelu(const onnx::NodeProto& nodeProto);
    void ParseLeakyRelu(const onnx::NodeProto& nodeProto);

    void ParseAdd(const onnx::NodeProto& nodeProto);
    void ParseAveragePool(const onnx::NodeProto& nodeProto);
    void ParseBatchNormalization(const onnx::NodeProto& node);
    void ParseConcat(const onnx::NodeProto& nodeProto);
    void ParseConstant(const onnx::NodeProto& nodeProto);
    void ParseConv(const onnx::NodeProto& nodeProto);
    void ParseFlatten(const onnx::NodeProto& node);
    void ParseGather(const onnx::NodeProto& node);
    void ParseGemm(const onnx::NodeProto& node);
    void ParseGlobalAveragePool(const onnx::NodeProto& node);
    void ParseMaxPool(const onnx::NodeProto& nodeProto);
    void ParseShape(const onnx::NodeProto& node);
    void ParseReshape(const onnx::NodeProto& nodeProto);
    void ParseUnsqueeze(const onnx::NodeProto& nodeProto);

    void RegisterInputSlot(armnn::IConnectableLayer* layer,
                           const std::string& tensorId,
                           unsigned int slotIndex);
    void RegisterInputSlots(armnn::IConnectableLayer* layer, const std::vector<std::string>& tensorIndexes);
    void RegisterOutputSlots(armnn::IConnectableLayer* layer, const std::vector<std::string>& tensorIndexes);

    void SetupInputLayers();
    void SetupOutputLayers();

    void ResetParser();
    void Cleanup();

    std::pair<armnn::ConstTensor, std::unique_ptr<float[]>>
    CreateConstTensor(const std::string name,
                      armnn::Optional<armnn::PermutationVector&> permutationVector = armnn::EmptyOptional());

    std::pair<armnn::ConstTensor, std::unique_ptr<int32_t[]>>
    CreateInt64ConstTensor(const std::string name,
                           armnn::Optional<armnn::PermutationVector&> permutationVector = armnn::EmptyOptional());

    template <typename TypeList, typename Location>
    void ValidateInputs(const onnx::NodeProto& node,
                        TypeList validInputs,
                        const Location& location);

    /// The network we're building. Gets cleared after it is passed to the user
    armnn::INetworkPtr m_Network;

    /// Ptr to the graph we're building the network from
    GraphPtr m_Graph;

    /// Map of the information for every tensor
    struct OnnxTensor
    {
        std::unique_ptr<armnn::TensorInfo>          m_info;
        std::unique_ptr<const onnx::TensorProto>    m_tensor;
        onnx::TensorProto::DataType                 m_dtype;

        OnnxTensor() : m_info(nullptr), m_tensor(nullptr), m_dtype(onnx::TensorProto::FLOAT) { }
        bool isConstant() { return m_tensor != nullptr; }
    };

    std::unordered_map<std::string, OnnxTensor> m_TensorsInfo;

    /// map of onnx operation names to parsing member functions
    static const std::map<std::string, OperationParsingFunction> m_ParserFunctions;

    /// A mapping of an output slot to each of the input slots it should be connected to
    /// The outputSlot is from the layer that creates this tensor as one of its outputs
    /// The inputSlots are from the layers that use this tensor as one of their inputs
    struct TensorSlots
    {
        armnn::IOutputSlot* outputSlot;
        std::vector<armnn::IInputSlot*> inputSlots;

        TensorSlots() : outputSlot(nullptr) { }
    };
    /// Map of the tensor names to their connections for the connections of the layers of the graph
    std::unordered_map<std::string, TensorSlots> m_TensorConnections;

    /// Map of the tensor names to their node and index in graph.node()
    std::unordered_map<std::string, std::pair<const onnx::NodeProto*, int>> m_OutputsMap;

    /// Number of times a specific node (identified by its index number) was used as input
    /// and list of the nodes it was fused with
    struct UsageSummary
    {
        std::vector<size_t> fusedWithNodes;
        size_t inputForNodes;

        UsageSummary() : fusedWithNodes({}), inputForNodes(0) { }

    };

    std::vector<UsageSummary> m_OutputsFusedAndUsed;

    std::map<std::string, armnn::TensorShape> m_InputShapes;

    std::unordered_map<std::string, armnn::TensorInfo> m_InputInfos;

    std::unordered_map<std::string, armnn::TensorInfo> m_OutputInfos;

};
}
