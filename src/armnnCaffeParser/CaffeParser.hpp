//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once
#include "armnnCaffeParser/ICaffeParser.hpp"

#include "armnn/Types.hpp"
#include "armnn/NetworkFwd.hpp"
#include "armnn/Tensor.hpp"

#include <memory>
#include <vector>
#include <unordered_map>

namespace caffe
{
class BlobShape;
class LayerParameter;
class NetParameter;
}

namespace armnnCaffeParser
{

using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

class CaffeParser : public ICaffeParser
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
    CaffeParser();

private:
    static std::pair<armnn::LayerBindingId, armnn::TensorInfo> GetBindingInfo(const std::string& layerName,
        const char* bindingPointDesc,
        const std::unordered_map<std::string, BindingPointInfo>& bindingInfos);

    /// Parses a NetParameter loaded into memory from one of the other CreateNetwork*
    armnn::INetworkPtr CreateNetworkFromNetParameter(
        caffe::NetParameter&     netParam,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs);

    /// does the actual conversion from caffe::NetParameter to armnn::INetwork
    void LoadNetParam(caffe::NetParameter& netParameter);

    /// Find the Caffe layers listed as inputs (bottoms) for a given layer.
    std::vector<const caffe::LayerParameter*> GetInputs(const caffe::LayerParameter& layerParam);

    /// Modifies the Caffe network to replace "in-place" layers (whose top() and bottom() are both the same)
    /// with regular layers. This simplifies further parsing.
    void ResolveInPlaceLayers(caffe::NetParameter& netParameter);

    /// Converts Caffe's protobuf tensor shape format to ArmNN's
    armnn::TensorInfo BlobShapeToTensorInfo(const caffe::BlobShape& blobShape) const;

    /// Adds an armnn layer to m_Network given a Caffe LayerParameter of the correct type
    /// and is responsible for recording any newly created IOutputSlots using SetArmnnOutputSlotForCaffeTop().
    /// @{
    void ParseInputLayer(const caffe::LayerParameter& layerParam);
    void ParseConvLayer(const caffe::LayerParameter& layerParam);
    void ParsePoolingLayer(const caffe::LayerParameter& layerParam);
    void ParseReluLayer(const caffe::LayerParameter& layerParam);
    void ParseLRNLayer(const caffe::LayerParameter& layerParam);
    void ParseInnerProductLayer(const caffe::LayerParameter& layerParam);
    void ParseSoftmaxLayer(const caffe::LayerParameter& layerParam);
    void ParseEltwiseLayer(const caffe::LayerParameter& layerParam);
    void ParseConcatLayer(const caffe::LayerParameter& layerParam);
    void ParseBatchNormLayer(const caffe::LayerParameter& layerParam);
    void ParseScaleLayer(const caffe::LayerParameter& layerParam);
    void ParseSplitLayer(const caffe::LayerParameter& layerParam);
    void ParseDropoutLayer(const caffe::LayerParameter& layerParam);
    // add layers that yolov2 need.
    void ParseReorgLayer(const caffe::LayerParameter& layerParam);
    void ParseDetectionOutputLayer(const caffe::LayerParameter& layerParam);
    /// @}

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

    /// Retrieves the Armnn IOutputSlot representing the given Caffe top.
    /// Throws if it cannot be found (e.g. not parsed yet).
    armnn::IOutputSlot& GetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName) const;

    void SetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName, armnn::IOutputSlot& armnnOutputSlot);

    void Cleanup();

    armnn::INetworkPtr m_Network;

    std::map<std::string, const caffe::LayerParameter*> m_CaffeLayersByTopName;

    using OperationParsingFunction = void(CaffeParser::*)(const caffe::LayerParameter& layerParam);

    /// map of Caffe layer names to parsing member functions
    static const std::map<std::string, OperationParsingFunction> ms_CaffeLayerNameToParsingFunctions;

    std::map<std::string, armnn::TensorShape> m_InputShapes;
    std::vector<std::string> m_RequestedOutputs;

    /// As we add armnn layers we store the armnn IOutputSlot which corresponds to the Caffe tops.
    std::unordered_map<std::string, armnn::IOutputSlot*> m_ArmnnOutputSlotForCaffeTop;

    /// maps input layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

    /// maps output layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;
};
}