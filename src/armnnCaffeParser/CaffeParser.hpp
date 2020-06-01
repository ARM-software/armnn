//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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

class CaffeParserBase:  public ICaffeParser
{
public:

    // Because we haven't looked at reducing the memory usage when loading from Text/String
    // have to retain these functions here for the moment.
    /// Create the network from a protobuf text file on disk
    virtual armnn::INetworkPtr CreateNetworkFromTextFile(
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

    CaffeParserBase();

protected:
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
    /// @}

    /// ParseConv may use these helpers depending on the group parameter
    /// @{
    void AddConvLayerWithSplits(const caffe::LayerParameter& layerParam,
                                const armnn::Convolution2dDescriptor & desc,
                                unsigned int kernelW,
                                unsigned int kernelH);
    void AddConvLayerWithDepthwiseConv(const caffe::LayerParameter& layerParam,
                                       const armnn::Convolution2dDescriptor & desc,
                                       unsigned int kernelW,
                                       unsigned int kernelH);
    /// @}

    /// Converts Caffe's protobuf tensor shape format to ArmNN's
    armnn::TensorInfo BlobShapeToTensorInfo(const caffe::BlobShape& blobShape) const;

    void TrackInputBinding(armnn::IConnectableLayer* layer,
                           armnn::LayerBindingId id,
                           const armnn::TensorInfo& tensorInfo);

    static void TrackBindingPoint(armnn::IConnectableLayer* layer, armnn::LayerBindingId id,
                                  const armnn::TensorInfo& tensorInfo,
                                  const char* bindingPointDesc,
                                  std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo);

    void TrackOutputBinding(armnn::IConnectableLayer* layer,
                            armnn::LayerBindingId id,
                            const armnn::TensorInfo& tensorInfo);


    void SetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName, armnn::IOutputSlot& armnnOutputSlot);

    /// Retrieves the Armnn IOutputSlot representing the given Caffe top.
    /// Throws if it cannot be found (e.g. not parsed yet).
    armnn::IOutputSlot& GetArmnnOutputSlotForCaffeTop(const std::string& caffeTopName) const;

    static std::pair<armnn::LayerBindingId, armnn::TensorInfo> GetBindingInfo(
        const std::string& layerName,
        const char* bindingPointDesc,
        const std::unordered_map<std::string, BindingPointInfo>& bindingInfos);


    void Cleanup();

    using OperationParsingFunction = void(CaffeParserBase::*)(const caffe::LayerParameter& layerParam);

    /// Maps Caffe layer names to parsing member functions.
    static const std::map<std::string, OperationParsingFunction> ms_CaffeLayerNameToParsingFunctions;

    /// maps input layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkInputsBindingInfo;

    /// maps output layer names to their corresponding ids and tensor infos
    std::unordered_map<std::string, BindingPointInfo> m_NetworkOutputsBindingInfo;

    armnn::INetworkPtr m_Network;

    std::map<std::string, armnn::TensorShape> m_InputShapes;

    /// As we add armnn layers we store the armnn IOutputSlot which corresponds to the Caffe tops.
    std::unordered_map<std::string, armnn::IOutputSlot*> m_ArmnnOutputSlotForCaffeTop;

    std::vector<std::string> m_RequestedOutputs;


    // Stuff which has gone to base class simply because we haven't done any
    // memory optimisation on the text/string format. If we move this to a layer
    // by layer parse as well these can move to the CaffeParser class.
    std::map<std::string, const caffe::LayerParameter*> m_CaffeLayersByTopName;

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

};

class CaffeParser : public CaffeParserBase
{
public:

    /// Create the network from a protobuf binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(
        const char* graphFile,
        const std::map<std::string, armnn::TensorShape>& inputShapes,
        const std::vector<std::string>& requestedOutputs) override;

public:
    CaffeParser();

};
}