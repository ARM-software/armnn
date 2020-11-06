//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ILayerVisitor.hpp>
#include <armnn/LayerVisitorBase.hpp>

#include <armnnSerializer/ISerializer.hpp>

#include <unordered_map>

#include <ArmnnSchema_generated.h>

#include <armnn/Types.hpp>

namespace armnnSerializer
{

class SerializerVisitor : public armnn::ILayerVisitor
{
public:
    SerializerVisitor() : m_layerId(0) {}
    ~SerializerVisitor() {}

    flatbuffers::FlatBufferBuilder& GetFlatBufferBuilder()
    {
        return m_flatBufferBuilder;
    }

    std::vector<int>& GetInputIds()
    {
        return m_inputIds;
    }

    std::vector<int>& GetOutputIds()
    {
        return m_outputIds;
    }

    std::vector<flatbuffers::Offset<armnnSerializer::AnyLayer>>& GetSerializedLayers()
    {
        return m_serializedLayers;
    }

    flatbuffers::Offset<armnnSerializer::FeatureCompatibilityVersions> GetVersionTable();


    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    void VisitAbsLayer(const armnn::IConnectableLayer* layer,
                       const char* name = nullptr) override;

    void VisitActivationLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ActivationDescriptor& descriptor,
                              const char* name = nullptr) override;

    void VisitAdditionLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr) override;

    void VisitArgMinMaxLayer(const armnn::IConnectableLayer* layer,
                             const armnn::ArgMinMaxDescriptor& argMinMaxDescriptor,
                             const char* name = nullptr) override;

    void VisitBatchToSpaceNdLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::BatchToSpaceNdDescriptor& descriptor,
                                  const char* name = nullptr) override;

    void VisitBatchNormalizationLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::BatchNormalizationDescriptor& BatchNormalizationDescriptor,
                                      const armnn::ConstTensor& mean,
                                      const armnn::ConstTensor& variance,
                                      const armnn::ConstTensor& beta,
                                      const armnn::ConstTensor& gamma,
                                      const char* name = nullptr) override;

    void VisitComparisonLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ComparisonDescriptor& descriptor,
                              const char* name = nullptr) override;

    void VisitConcatLayer(const armnn::IConnectableLayer* layer,
                          const armnn::ConcatDescriptor& concatDescriptor,
                          const char* name = nullptr) override;

    void VisitConstantLayer(const armnn::IConnectableLayer* layer,
                            const armnn::ConstTensor& input,
                            const char* = nullptr) override;

    void VisitConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::Convolution2dDescriptor& descriptor,
                                 const armnn::ConstTensor& weights,
                                 const armnn::Optional<armnn::ConstTensor>& biases,
                                 const char* = nullptr) override;

    void VisitDepthToSpaceLayer(const armnn::IConnectableLayer* layer,
                                const armnn::DepthToSpaceDescriptor& descriptor,
                                const char* name = nullptr) override;

    void VisitDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                          const armnn::ConstTensor& weights,
                                          const armnn::Optional<armnn::ConstTensor>& biases,
                                          const char* name = nullptr) override;

    void VisitDequantizeLayer(const armnn::IConnectableLayer* layer,
                              const char* name = nullptr) override;

    void VisitDetectionPostProcessLayer(const armnn::IConnectableLayer* layer,
                                        const armnn::DetectionPostProcessDescriptor& descriptor,
                                        const armnn::ConstTensor& anchors,
                                        const char* name = nullptr) override;

    void VisitDivisionLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr) override;

    void VisitElementwiseUnaryLayer(const armnn::IConnectableLayer* layer,
                                    const armnn::ElementwiseUnaryDescriptor& descriptor,
                                    const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitComparisonLayer instead")
    void VisitEqualLayer(const armnn::IConnectableLayer* layer,
                         const char* name = nullptr) override;

    void VisitFillLayer(const armnn::IConnectableLayer* layer,
                        const armnn::FillDescriptor& fillDescriptor,
                        const char* name = nullptr) override;

    void VisitFloorLayer(const armnn::IConnectableLayer *layer,
                         const char *name = nullptr) override;

    void VisitFullyConnectedLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                  const armnn::ConstTensor& weights,
                                  const armnn::Optional<armnn::ConstTensor>& biases,
                                  const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitGatherLayer with descriptor instead")
    void VisitGatherLayer(const armnn::IConnectableLayer* layer,
                          const char* name = nullptr) override;

    void VisitGatherLayer(const armnn::IConnectableLayer* layer,
                          const armnn::GatherDescriptor& gatherDescriptor,
                          const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitComparisonLayer instead")
    void VisitGreaterLayer(const armnn::IConnectableLayer* layer,
                           const char* name = nullptr) override;

    void VisitInputLayer(const armnn::IConnectableLayer* layer,
                         armnn::LayerBindingId id,
                         const char* name = nullptr) override;

    void VisitInstanceNormalizationLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::InstanceNormalizationDescriptor& instanceNormalizationDescriptor,
                                         const char* name = nullptr) override;

    void VisitL2NormalizationLayer(const armnn::IConnectableLayer* layer,
                                   const armnn::L2NormalizationDescriptor& l2NormalizationDescriptor,
                                   const char* name = nullptr) override;

    void VisitLogicalBinaryLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::LogicalBinaryDescriptor& descriptor,
                                 const char* name = nullptr) override;

    void VisitLogSoftmaxLayer(const armnn::IConnectableLayer* layer,
                              const armnn::LogSoftmaxDescriptor& logSoftmaxDescriptor,
                              const char* name = nullptr) override;

    void VisitLstmLayer(const armnn::IConnectableLayer* layer,
                        const armnn::LstmDescriptor& descriptor,
                        const armnn::LstmInputParams& params,
                        const char* name = nullptr) override;

    void VisitMeanLayer(const armnn::IConnectableLayer* layer,
                        const armnn::MeanDescriptor& descriptor,
                        const char* name) override;

    void VisitMinimumLayer(const armnn::IConnectableLayer* layer,
                           const char* name = nullptr) override;

    void VisitMaximumLayer(const armnn::IConnectableLayer* layer,
                           const char* name = nullptr) override;

    void VisitMergeLayer(const armnn::IConnectableLayer* layer,
                         const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitConcatLayer instead")
    void VisitMergerLayer(const armnn::IConnectableLayer* layer,
                          const armnn::MergerDescriptor& mergerDescriptor,
                          const char* name = nullptr) override;

    void VisitMultiplicationLayer(const armnn::IConnectableLayer* layer,
                                  const char* name = nullptr) override;

    void VisitOutputLayer(const armnn::IConnectableLayer* layer,
                          armnn::LayerBindingId id,
                          const char* name = nullptr) override;

    void VisitPadLayer(const armnn::IConnectableLayer* layer,
                       const armnn::PadDescriptor& PadDescriptor,
                       const char* name = nullptr) override;

    void VisitPermuteLayer(const armnn::IConnectableLayer* layer,
                           const armnn::PermuteDescriptor& PermuteDescriptor,
                           const char* name = nullptr) override;

    void VisitPooling2dLayer(const armnn::IConnectableLayer* layer,
                             const armnn::Pooling2dDescriptor& pooling2dDescriptor,
                             const char* name = nullptr) override;

    void VisitPreluLayer(const armnn::IConnectableLayer* layer,
                         const char* name = nullptr) override;

    void VisitQuantizeLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr) override;

    void VisitQLstmLayer(const armnn::IConnectableLayer* layer,
                         const armnn::QLstmDescriptor& descriptor,
                         const armnn::LstmInputParams& params,
                         const char* name = nullptr) override;

    void VisitQuantizedLstmLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::QuantizedLstmInputParams& params,
                                 const char* name = nullptr) override;

    void VisitRankLayer(const armnn::IConnectableLayer* layer,
                        const char* name = nullptr) override;

    void VisitReshapeLayer(const armnn::IConnectableLayer* layer,
                           const armnn::ReshapeDescriptor& reshapeDescriptor,
                           const char* name = nullptr) override;

    void VisitResizeLayer(const armnn::IConnectableLayer* layer,
                          const armnn::ResizeDescriptor& resizeDescriptor,
                          const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitResizeLayer instead")
    void VisitResizeBilinearLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::ResizeBilinearDescriptor& resizeDescriptor,
                                  const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    void VisitRsqrtLayer(const armnn::IConnectableLayer* layer,
                         const char* name = nullptr) override;

    void VisitSliceLayer(const armnn::IConnectableLayer* layer,
                         const armnn::SliceDescriptor& sliceDescriptor,
                         const char* name = nullptr) override;

    void VisitSoftmaxLayer(const armnn::IConnectableLayer* layer,
                           const armnn::SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override;

    void VisitSpaceToBatchNdLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                  const char* name = nullptr) override;

    void VisitSpaceToDepthLayer(const armnn::IConnectableLayer* layer,
                                const armnn::SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                const char* name = nullptr) override;

    void VisitNormalizationLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::NormalizationDescriptor& normalizationDescriptor,
                                 const char* name = nullptr) override;

    void VisitSplitterLayer(const armnn::IConnectableLayer* layer,
                            const armnn::ViewsDescriptor& viewsDescriptor,
                            const char* name = nullptr) override;

    void VisitStandInLayer(const armnn::IConnectableLayer* layer,
                           const armnn::StandInDescriptor& standInDescriptor,
                           const char* name = nullptr) override;

    void VisitStackLayer(const armnn::IConnectableLayer* layer,
                         const armnn::StackDescriptor& stackDescriptor,
                         const char* name = nullptr) override;

    void VisitStridedSliceLayer(const armnn::IConnectableLayer* layer,
                                const armnn::StridedSliceDescriptor& stridedSliceDescriptor,
                                const char* name = nullptr) override;

    void VisitSubtractionLayer(const armnn::IConnectableLayer* layer,
                               const char* name = nullptr) override;

    void VisitSwitchLayer(const armnn::IConnectableLayer* layer,
                          const char* name = nullptr) override;

    void VisitTransposeConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::TransposeConvolution2dDescriptor& descriptor,
                                          const armnn::ConstTensor& weights,
                                          const armnn::Optional<armnn::ConstTensor>& biases,
                                          const char* = nullptr) override;

    void VisitTransposeLayer(const armnn::IConnectableLayer* layer,
                             const armnn::TransposeDescriptor& descriptor,
                             const char* name = nullptr) override;

private:

    /// Creates the Input Slots and Output Slots and LayerBase for the layer.
    flatbuffers::Offset<armnnSerializer::LayerBase> CreateLayerBase(
            const armnn::IConnectableLayer* layer,
            const armnnSerializer::LayerType layerType);

    /// Creates the serializer AnyLayer for the layer and adds it to m_serializedLayers.
    void CreateAnyLayer(const flatbuffers::Offset<void>& layer, const armnnSerializer::Layer serializerLayer);

    /// Creates the serializer ConstTensor for the armnn ConstTensor.
    flatbuffers::Offset<armnnSerializer::ConstTensor> CreateConstTensorInfo(
            const armnn::ConstTensor& constTensor);

    /// Creates the serializer TensorInfo for the armnn TensorInfo.
    flatbuffers::Offset<TensorInfo>  CreateTensorInfo(const armnn::TensorInfo& tensorInfo);

    template <typename T>
    flatbuffers::Offset<flatbuffers::Vector<T>> CreateDataVector(const void* memory, unsigned int size);

    ///Function which maps Guid to an index
    uint32_t GetSerializedId(armnn::LayerGuid guid);

    /// Creates the serializer InputSlots for the layer.
    std::vector<flatbuffers::Offset<armnnSerializer::InputSlot>> CreateInputSlots(
            const armnn::IConnectableLayer* layer);

    /// Creates the serializer OutputSlots for the layer.
    std::vector<flatbuffers::Offset<armnnSerializer::OutputSlot>> CreateOutputSlots(
            const armnn::IConnectableLayer* layer);

    /// FlatBufferBuilder to create our layers' FlatBuffers.
    flatbuffers::FlatBufferBuilder m_flatBufferBuilder;

    /// AnyLayers required by the SerializedGraph.
    std::vector<flatbuffers::Offset<armnnSerializer::AnyLayer>> m_serializedLayers;

    /// Vector of the binding ids of all Input Layers required by the SerializedGraph.
    std::vector<int> m_inputIds;

    /// Vector of the binding ids of all Output Layers required by the SerializedGraph.
    std::vector<int> m_outputIds;

    /// Mapped Guids of all Layers to match our index.
    std::unordered_map<armnn::LayerGuid, uint32_t > m_guidMap;

    /// layer within our FlatBuffer index.
    uint32_t m_layerId;
};

class Serializer : public ISerializer
{
public:
    Serializer() {}
    ~Serializer() {}

    /// Serializes the network to ArmNN SerializedGraph.
    /// @param [in] inNetwork The network to be serialized.
    void Serialize(const armnn::INetwork& inNetwork) override;

    /// Serializes the SerializedGraph to the stream.
    /// @param [stream] the stream to save to
    /// @return true if graph is Serialized to the Stream, false otherwise
    bool SaveSerializedToStream(std::ostream& stream) override;

private:

    /// Visitor to contruct serialized network
    SerializerVisitor m_SerializerVisitor;
};

} //namespace armnnSerializer
