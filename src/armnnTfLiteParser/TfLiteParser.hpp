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
#include <unordered_map>
#include <vector>

#include <tensorflow/lite/version.h>

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION > 3)
#define ARMNN_POST_TFLITE_2_3
#endif

namespace armnnTfLiteParser
{

class TfLiteParserImpl
{
public:
    // Shorthands for TfLite types
    using ModelPtr = std::unique_ptr<tflite::ModelT>;
    using SubgraphPtr = std::unique_ptr<tflite::SubGraphT>;
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

    TfLiteParserImpl(const armnn::Optional<ITfLiteParser::TfLiteParserOptions>& options = armnn::EmptyOptional());
    ~TfLiteParserImpl() = default;

public:
    // testable helpers
    armnn::INetworkPtr CreateNetworkFromBinaryAsDynamic(const std::vector<uint8_t>& binaryContent);

    armnn::INetworkPtr LoadModel(std::unique_ptr<tflite::ModelT> model);

    static ModelPtr LoadModelFromFile(const char* fileName);
    static ModelPtr LoadModelFromBinary(const uint8_t* binaryContent, size_t len);
    static TensorRawPtrVector GetInputs(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
    static TensorRawPtrVector GetOutputs(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
    static TensorIdRawPtrVector GetSubgraphInputs(const ModelPtr& model, size_t subgraphIndex);
    static TensorIdRawPtrVector GetSubgraphOutputs(const ModelPtr& model, size_t subgraphIndex);
    static std::vector<int32_t>& GetInputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);
    static std::vector<int32_t>& GetOutputTensorIds(const ModelPtr& model, size_t subgraphIndex, size_t operatorIndex);

    static BufferRawPtr GetBuffer(const ModelPtr& model, size_t bufferIndex);
    static armnn::TensorInfo OutputShapeOfSqueeze(std::vector<uint32_t> squeezeDims,
                                                  const armnn::TensorInfo& inputTensorInfo);
    static armnn::TensorInfo OutputShapeOfReshape(const armnn::TensorInfo& inputTensorInfo,
                                                  const std::vector<int32_t>& targetDimsIn);

    /// Retrieve version in X.Y.Z form
    static const std::string GetVersion();

private:

    // No copying allowed until it is wanted and properly implemented
    TfLiteParserImpl(const TfLiteParserImpl &) = delete;
    TfLiteParserImpl & operator=(const TfLiteParserImpl &) = delete;

    /// Create the network from an already loaded flatbuffers model
    armnn::INetworkPtr CreateNetworkFromModel();

    // signature for the parser functions
    using OperatorParsingFunction = void(TfLiteParserImpl::*)(size_t subgraphIndex, size_t operatorIndex);

    void ParseCustomOperator(size_t subgraphIndex, size_t operatorIndex);
    void ParseUnsupportedOperator(size_t subgraphIndex, size_t operatorIndex);

    void ParseAbs(size_t subgraphIndex, size_t operatorIndex);
    void ParseActivation(size_t subgraphIndex, size_t operatorIndex, armnn::ActivationFunction activationType);
    void ParseAdd(size_t subgraphIndex, size_t operatorIndex);
    void ParseArgMinMax(size_t subgraphIndex, size_t operatorIndex, armnn::ArgMinMaxFunction argMinMaxFunction);
    void ParseArgMin(size_t subgraphIndex, size_t operatorIndex);
    void ParseArgMax(size_t subgraphIndex, size_t operatorIndex);
    void ParseAveragePool2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseBatchToSpaceND(size_t subgraphIndex, size_t operatorIndex);
    void ParseCast(size_t subgraphIndex, size_t operatorIndex);
    void ParseComparison(size_t subgraphIndex, size_t operatorIndex, armnn::ComparisonOperation comparisonOperation);
    void ParseConcatenation(size_t subgraphIndex, size_t operatorIndex);
    void ParseConv2D(size_t subgraphIndex, size_t operatorIndex);
    // Conv3D support was added in TF 2.5, so for backwards compatibility a hash define is needed.
    #if defined(ARMNN_POST_TFLITE_2_3)
    void ParseConv3D(size_t subgraphIndex, size_t operatorIndex);
    #endif
    void ParseDepthToSpace(size_t subgraphIndex, size_t operatorIndex);
    void ParseDepthwiseConv2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseDequantize(size_t subgraphIndex, size_t operatorIndex);
    void ParseDetectionPostProcess(size_t subgraphIndex, size_t operatorIndex);
    void ParseDiv(size_t subgraphIndex, size_t operatorIndex);
    void ParseElementwiseUnary(size_t subgraphIndex, size_t operatorIndex, armnn::UnaryOperation unaryOperation);
    void ParseElu(size_t subgraphIndex, size_t operatorIndex);
    void ParseEqual(size_t subgraphIndex, size_t operatorIndex);
    void ParseExp(size_t subgraphIndex, size_t operatorIndex);
    void ParseExpandDims(size_t subgraphIndex, size_t operatorIndex);
    void ParseFullyConnected(size_t subgraphIndex, size_t operatorIndex);
    void ParseGather(size_t subgraphIndex, size_t operatorIndex);
    void ParseGreater(size_t subgraphIndex, size_t operatorIndex);
    void ParseGreaterOrEqual(size_t subgraphIndex, size_t operatorIndex);
    void ParseHardSwish(size_t subgraphIndex, size_t operatorIndex);
    void ParseLeakyRelu(size_t subgraphIndex, size_t operatorIndex);
    void ParseLess(size_t subgraphIndex, size_t operatorIndex);
    void ParseLessOrEqual(size_t subgraphIndex, size_t operatorIndex);
    void ParseLocalResponseNormalization(size_t subgraphIndex, size_t operatorIndex);
    void ParseLogicalNot(size_t subgraphIndex, size_t operatorIndex);
    void ParseLogistic(size_t subgraphIndex, size_t operatorIndex);
    void ParseL2Normalization(size_t subgraphIndex, size_t operatorIndex);
    void ParseMaxPool2D(size_t subgraphIndex, size_t operatorIndex);
    void ParseMaximum(size_t subgraphIndex, size_t operatorIndex);
    void ParseMean(size_t subgraphIndex, size_t operatorIndex);
    void ParseMinimum(size_t subgraphIndex, size_t operatorIndex);
    void ParseMirrorPad(size_t subgraphIndex, size_t operatorIndex);
    void ParseMul(size_t subgraphIndex, size_t operatorIndex);
    void ParseNeg(size_t subgraphIndex, size_t operatorIndex);
    void ParseNotEqual(size_t subgraphIndex, size_t operatorIndex);
    void ParsePack(size_t subgraphIndex, size_t operatorIndex);
    void ParsePad(size_t subgraphIndex, size_t operatorIndex);
    void ParsePool(size_t subgraphIndex, size_t operatorIndex, armnn::PoolingAlgorithm algorithm);
    void ParsePrelu(size_t subgraphIndex, size_t operatorIndex);
    void ParseQuantize(size_t subgraphIndex, size_t operatorIndex);
    void ParseReduce(size_t subgraphIndex, size_t operatorIndex, armnn::ReduceOperation reduceOperation);
    void ParseReduceMax(size_t subgraphIndex, size_t operatorIndex);
    void ParseReduceMin(size_t subgraphIndex, size_t operatorIndex);
    void ParseReduceProd(size_t subgraphIndex, size_t operatorIndex);
    void ParseRelu(size_t subgraphIndex, size_t operatorIndex);
    void ParseRelu6(size_t subgraphIndex, size_t operatorIndex);
    void ParseReshape(size_t subgraphIndex, size_t operatorIndex);
    void ParseResize(size_t subgraphIndex, size_t operatorIndex, armnn::ResizeMethod resizeMethod);
    void ParseResizeBilinear(size_t subgraphIndex, size_t operatorIndex);
    void ParseResizeNearestNeighbor(size_t subgraphIndex, size_t operatorIndex);
    void ParseRsqrt(size_t subgraphIndex, size_t operatorIndex);
    void ParseShape(size_t subgraphIndex, size_t operatorIndex);
    void ParseSlice(size_t subgraphIndex, size_t operatorIndex);
    void ParseSoftmax(size_t subgraphIndex, size_t operatorIndex);
    void ParseSpaceToBatchND(size_t subgraphIndex, size_t operatorIndex);
    void ParseSplit(size_t subgraphIndex, size_t operatorIndex);
    void ParseSplitV(size_t subgraphIndex, size_t operatorIndex);
    void ParseSqueeze(size_t subgraphIndex, size_t operatorIndex);
    void ParseStridedSlice(size_t subgraphIndex, size_t operatorIndex);
    void ParseSub(size_t subgraphIndex, size_t operatorIndex);
    void ParseSum(size_t subgraphIndex, size_t operatorIndex);
    void ParseTanH(size_t subgraphIndex, size_t operatorIndex);
    void ParseTranspose(size_t subgraphIndex, size_t operatorIndex);
    void ParseTransposeConv(size_t subgraphIndex, size_t operatorIndex);
    void ParseUnpack(size_t subgraphIndex, size_t operatorIndex);

    void RegisterProducerOfTensor(size_t subgraphIndex, size_t tensorIndex, armnn::IOutputSlot* slot);
    void RegisterConsumerOfTensor(size_t subgraphIndex, size_t tensorIndex, armnn::IInputSlot* slot);
    void RegisterInputSlots(size_t subgraphIndex,
                            size_t operatorIndex,
                            armnn::IConnectableLayer* layer,
                            const std::vector<unsigned int>& tensorIndexes,
                            unsigned int startingSlotIndex = 0);
    void RegisterOutputSlots(size_t subgraphIndex,
                             size_t operatorIndex,
                             armnn::IConnectableLayer* layer,
                             const std::vector<unsigned int>& tensorIndexes);

    void SetupInputLayers(size_t subgraphIndex);
    void SetupOutputLayers(size_t subgraphIndex);
    void SetupConstantLayers(size_t subgraphIndex);

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
    public:
        // Convenience constructors
        SupportedDataStorage(std::unique_ptr<float[]>&&   data);
        SupportedDataStorage(std::unique_ptr<uint8_t[]>&& data);
        SupportedDataStorage(std::unique_ptr<int8_t[]>&&  data);
        SupportedDataStorage(std::unique_ptr<int32_t[]>&& data);

    private:
        // Pointers to the data buffers
        std::unique_ptr<float[]>   m_FloatData;
        std::unique_ptr<uint8_t[]> m_Uint8Data;
        std::unique_ptr<int8_t[]>  m_Int8Data;
        std::unique_ptr<int32_t[]> m_Int32Data;
    };

    bool IsConstTensor(TensorRawPtr tensorPtr);
    armnn::ConstTensor CreateConstTensorNonPermuted(TensorRawPtr tensorPtr,
                                                    armnn::TensorInfo& tensorInfo);
    std::pair<armnn::ConstTensor, SupportedDataStorage>
    CreateConstTensorPermuted(TensorRawPtr tensorPtr,
                              armnn::TensorInfo& tensorInfo,
                              armnn::Optional<armnn::PermutationVector&> permutationVector);

    template<typename T>
    std::pair<armnn::ConstTensor, TfLiteParserImpl::SupportedDataStorage>
    CreateConstTensorAndStoreData(TfLiteParserImpl::BufferRawPtr bufferPtr,
                                  TfLiteParserImpl::TensorRawPtr tensorPtr,
                                  armnn::TensorInfo& tensorInfo,
                                  armnn::Optional<armnn::PermutationVector&> permutationVector);

    // Settings for configuring the TfLiteParser
    armnn::Optional<ITfLiteParser::TfLiteParserOptions> m_Options;

    /// The network we're building. Gets cleared after it is passed to the user
    armnn::INetworkPtr                    m_Network;
    ModelPtr                              m_Model;

    std::vector<OperatorParsingFunction>                     m_ParserFunctions;
    std::unordered_map<std::string, OperatorParsingFunction> m_CustomParserFunctions;

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

    /// This is used in case that the model does not speciry the output.
    /// The shape can be calculated from the options.
    std::vector<std::vector<unsigned int>> m_OverridenOutputShapes;
};

}
