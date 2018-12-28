//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/INetwork.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "armnn/Types.hpp"

#include <schema_generated.h>
#include <functional>
#include <vector>

namespace armnnTfLiteParser
{

class TfLiteParser : public ITfLiteParser
{
public:
    // Shorthands for TfLite types
    using ModelPtr = std::unique_ptr<tflite::ModelT>;
    using SubGraphPtr = std::unique_ptr<tflite::SubGraphT>;
    using OperatorPtr = std::unique_ptr<tflite::OperatorT>;
    using OperatorCodePtr = std::unique_ptr<tflite::OperatorCodeT>;
    using TensorPtr = std::unique_ptr<tflite::TensorT>;
    using TensorRawPtr = const tflite::TensorT *;
    using TensorRawPtrVector = std::vector<TensorRawPtr>;
    using TensorIdRawPtr = std::pair<size_t, TensorRawPtr>;
    using TensorIdRawPtrVector = std::vector<TensorIdRawPtr>;
    using BufferPtr = std::unique_ptr<tflite::BufferT>;
    using BufferRawPtr = const tflite::BufferT *;

public:
    /// Create the network from a flatbuffers binary file on disk
    virtual armnn::INetworkPtr CreateNetworkFromBinaryFile(const char* graphFile) override;

    /// Create the network from a flatbuffers binary
    virtual armnn::INetworkPtr CreateNetworkFromBinary(const std::vector<uint8_t> & binaryContent) override;


    /// Retrieve binding info (layer id and tensor info) for the network input identified by
    /// the given layer name and subgraph id
    virtual BindingPointInfo GetNetworkInputBindingInfo(size_t subgraphId,
                                                        const std::string& name) const override;

    /// Retrieve binding info (layer id and tensor info) for the network output identified by
    /// the given layer name and subgraph id
    virtual BindingPointInfo GetNetworkOutputBindingInfo(size_t subgraphId,
                                                         const std::string& name) const override;

    /// Return the number of subgraphs in the parsed model
    virtual size_t GetSubgraphCount() const override;

    /// Return the input tensor names for a given subgraph
    virtual std::vector<std::string> GetSubgraphInputTensorNames(size_t subgraphId) const override;

    /// Return the output tensor names for a given subgraph
    virtual std::vector<std::string> GetSubgraphOutputTensorNames(size_t subgraphId) const override;

    TfLiteParser();
    virtual ~TfLiteParser() {}

public:
    // testable helpers
    static ModelPtr LoadModelFromFile(const char * fileName);
    static ModelPtr LoadModelFromBinary(const uint8_t * binaryContent, size_t len);
    static TensorRawPtrVector GetInputs(const ModelPtr & model, size_t subgraphIndex, size_t operatorIndex);
    static TensorRawPtrVector GetOutputs(const ModelPtr & model, size_t subgraphIndex, size_t operatorIndex);
    static TensorIdRawPtrVector GetSubgraphInputs(const ModelPtr & model, size_t subgraphIndex);
    static TensorIdRawPtrVector GetSubgraphOutputs(const ModelPtr & model, size_t subgraphIndex);
    static std::vector<int32_t>& GetInputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
    static std::vector<int32_t>& GetOutputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);

    static BufferRawPtr GetBuffer(const ModelPtr& model, size_t bufferIndex);
    static armnn::TensorInfo OutputShapeOfSqueeze(const std::vector<uint32_t> & squeezeDims,
                                                  const armnn::TensorInfo & inputTensorInfo);
    static armnn::TensorInfo OutputShapeOfReshape(const armnn::TensorInfo & inputTensorInfo,
                                                  const std::vector<int32_t> & targetDimsIn);

private:
    // No copying allowed until it is wanted and properly implemented
    TfLiteParser(const TfLiteParser &) = delete;
    TfLiteParser & operator=(const TfLiteParser &) = delete;

    /// Create the network from an already loaded flatbuffers model
    armnn::INetworkPtr CreateNetworkFromModel();

    // signature for the parser functions
    using OperatorParsingFunction = void(TfLiteParser::*)(size_t subgraphIndex, size_t operatorIndex);

    void ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex);
    void ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseConcatenation(size_t subgraphIndex, size_t operatorIndex);
    void ParseConv2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex);
    void ParseMaxPool2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseRelu(size_t subgraphIndex, size_t operatorIndex);
    void ParseRelu6(size_t subgraphIndex, size_t operatorIndex);
    void ParseReshape(size_t subgraphIndex, size_t operatorIndex);
    void ParseSoftmax(size_t subgraphIndex, size_t operatorIndex);
    void ParseSqueeze(size_t subgraphIndex, size_t operatorIndex);
    void ParseAdd(size_t subgraphIndex, size_t operatorIndex);
    void ParseMul(size_t subgraphIndex, size_t operatorIndex);
    void ParseMean(size_t subgraphIndex, size_t operatorIndex);
    void ParsePad(size_t subgraphIndex, size_t operatorIndex);

    void ParsePool(size_t subgraphIndex, size_t operatorIndex, armnn::PoolingAlgorithm algorithm);

    void RegisterProducerOfTensor(size_t subgraphIndex, size_t tensorIndex, armnn::IOutputSlot* slot);
    void RegisterConsumerOfTensor(size_t subgraphIndex, size_t tensorIndex, armnn::IInputSlot* slot);
    void RegisterInputSlots(size_t subgraphIndex,
                            size_t operatorIndex,
                            armnn::IConnectableLayer* layer,
                            const std::vector<unsigned int>& tensorIndexes);
    void RegisterOutputSlots(size_t subgraphIndex,
                             size_t operatorIndex,
                             armnn::IConnectableLayer* layer,
                             const std::vector<unsigned int>& tensorIndexes);

    void SetupInputLayers(size_t subgraphIndex);
    void SetupOutputLayers(size_t subgraphIndex);

    void ResetParser();

    void AddBroadcastReshapeLayer(size_t subgraphIndex,
                                  size_t operatorIndex,
                                  armnn::IConnectableLayer* layer);

    /// Attach an activation layer to the one passed as a parameter
    armnn::IConnectableLayer* AddFusedActivationLayer(armnn::IConnectableLayer* layer,
                                                      unsigned int outputSlot,
                                                      tflite::ActivationFunctionType activationType);

    // SupportedDataStorage's purpose is to hold data till we pass over to the network.
    // We don't care about the content, and we want a single datatype to simplify the code.
    struct SupportedDataStorage
    {
        std::unique_ptr<float[]>    m_FloatData;
        std::unique_ptr<uint8_t[]>  m_Uint8Data;
        std::unique_ptr<int32_t[]>  m_Int32Data;

        SupportedDataStorage(std::unique_ptr<float[]> && data);
        SupportedDataStorage(std::unique_ptr<uint8_t[]> && data);
        SupportedDataStorage(std::unique_ptr<int32_t[]> && data);
    };

    std::pair<armnn::ConstTensor, SupportedDataStorage> CreateConstTensor(TensorRawPtr tensorPtr,
                                                                          armnn::TensorInfo & tensorInfo);

    /// The network we're building. Gets cleared after it is passed to the user
    armnn::INetworkPtr                    m_Network;
    std::vector<OperatorParsingFunction>  m_ParserFunctions;
    ModelPtr                              m_Model;

    /// A mapping of an output slot to each of the input slots it should be connected to
    /// The outputSlot is from the layer that creates this tensor as one of its ouputs
    /// The inputSlots are from the layers that use this tensor as one of their inputs
    struct TensorSlots
    {
        armnn::IOutputSlot* outputSlot;
        std::vector<armnn::IInputSlot*> inputSlots;

        TensorSlots() : outputSlot(nullptr) { }
    };
    typedef std::vector<TensorSlots> TensorConnections;
    /// Connections for tensors in each subgraph
    /// The first index is the subgraph ID, the second index is the tensor ID
    std::vector<TensorConnections> m_SubgraphConnections;
};

}
