//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ILayerVisitor.hpp>
#include <armnn/IStrategy.hpp>
#include <armnn/LayerVisitorBase.hpp>

#include <armnnSerializer/ISerializer.hpp>

#include <common/include/ProfilingGuid.hpp>

#include <unordered_map>

#include "ArmnnSchema_generated.h"

#include <armnn/Types.hpp>

namespace armnnSerializer
{

class SerializerStrategy : public armnn::IStrategy
{
public:
    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const armnn::BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id) override;

    SerializerStrategy() : m_layerId(0) {}
    ~SerializerStrategy() {}

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

private:
    void SerializeActivationLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::ActivationDescriptor& descriptor,
                                  const char* name = nullptr);

    void SerializeAdditionLayer(const armnn::IConnectableLayer* layer,
                                const char* name = nullptr);

    void SerializeArgMinMaxLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::ArgMinMaxDescriptor& argMinMaxDescriptor,
                                 const char* name = nullptr);

    void SerializeBatchToSpaceNdLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::BatchToSpaceNdDescriptor& descriptor,
                                      const char* name = nullptr);

    void SerializeBatchNormalizationLayer(const armnn::IConnectableLayer* layer,
                                          const armnn::BatchNormalizationDescriptor& BatchNormalizationDescriptor,
                                          const std::vector<armnn::ConstTensor>& constants,
                                          const char* name = nullptr);

    void SerializeCastLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr);

    void SerializeChannelShuffleLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::ChannelShuffleDescriptor& descriptor,
                                      const char* name = nullptr);

    void SerializeComparisonLayer(const armnn::IConnectableLayer* layer,
                                  const armnn::ComparisonDescriptor& descriptor,
                                  const char* name = nullptr);

    void SerializeConcatLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ConcatDescriptor& concatDescriptor,
                              const char* name = nullptr);

    void SerializeConstantLayer(const armnn::IConnectableLayer* layer,
                                const std::vector<armnn::ConstTensor>& constants,
                                const char* name = nullptr);

    void SerializeConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                     const armnn::Convolution2dDescriptor& descriptor,
                                     const std::vector<armnn::ConstTensor>& constants,
                                     const char* name = nullptr);

    void SerializeConvolution3dLayer(const armnn::IConnectableLayer* layer,
                                     const armnn::Convolution3dDescriptor& descriptor,
                                     const char* name = nullptr);

    void SerializeDepthToSpaceLayer(const armnn::IConnectableLayer* layer,
                                    const armnn::DepthToSpaceDescriptor& descriptor,
                                    const char* name = nullptr);

    void SerializeDepthwiseConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                              const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                              const std::vector<armnn::ConstTensor>& constants,
                                              const char* name = nullptr);

    void SerializeDequantizeLayer(const armnn::IConnectableLayer* layer,
                                  const char* name = nullptr);

    void SerializeDetectionPostProcessLayer(const armnn::IConnectableLayer* layer,
                                            const armnn::DetectionPostProcessDescriptor& descriptor,
                                            const std::vector<armnn::ConstTensor>& constants,
                                            const char* name = nullptr);

    void SerializeDivisionLayer(const armnn::IConnectableLayer* layer,
                                const char* name = nullptr);

    void SerializeElementwiseUnaryLayer(const armnn::IConnectableLayer* layer,
                                        const armnn::ElementwiseUnaryDescriptor& descriptor,
                                        const char* name = nullptr);

    void SerializeFillLayer(const armnn::IConnectableLayer* layer,
                            const armnn::FillDescriptor& fillDescriptor,
                            const char* name = nullptr);

    void SerializeFloorLayer(const armnn::IConnectableLayer *layer,
                             const char *name = nullptr);

    void SerializeFullyConnectedLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                      const char* name = nullptr);

    void SerializeGatherLayer(const armnn::IConnectableLayer* layer,
                              const armnn::GatherDescriptor& gatherDescriptor,
                              const char* name = nullptr);

    void SerializeInputLayer(const armnn::IConnectableLayer* layer,
                         armnn::LayerBindingId id,
                         const char* name = nullptr);

    void SerializeInstanceNormalizationLayer(const armnn::IConnectableLayer* layer,
                                         const armnn::InstanceNormalizationDescriptor& instanceNormalizationDescriptor,
                                         const char* name = nullptr);

    void SerializeL2NormalizationLayer(const armnn::IConnectableLayer* layer,
                                   const armnn::L2NormalizationDescriptor& l2NormalizationDescriptor,
                                   const char* name = nullptr);

    void SerializeLogicalBinaryLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::LogicalBinaryDescriptor& descriptor,
                                 const char* name = nullptr);

    void SerializeLogSoftmaxLayer(const armnn::IConnectableLayer* layer,
                              const armnn::LogSoftmaxDescriptor& logSoftmaxDescriptor,
                              const char* name = nullptr);

    void SerializeLstmLayer(const armnn::IConnectableLayer* layer,
                            const armnn::LstmDescriptor& descriptor,
                            const std::vector<armnn::ConstTensor>& constants,
                            const char* name = nullptr);

    void SerializeMeanLayer(const armnn::IConnectableLayer* layer,
                            const armnn::MeanDescriptor& descriptor,
                            const char* name);

    void SerializeMinimumLayer(const armnn::IConnectableLayer* layer,
                               const char* name = nullptr);

    void SerializeMaximumLayer(const armnn::IConnectableLayer* layer,
                               const char* name = nullptr);

    void SerializeMergeLayer(const armnn::IConnectableLayer* layer,
                             const char* name = nullptr);

    void SerializeMultiplicationLayer(const armnn::IConnectableLayer* layer,
                                      const char* name = nullptr);

    void SerializeOutputLayer(const armnn::IConnectableLayer* layer,
                              armnn::LayerBindingId id,
                              const char* name = nullptr);

    void SerializePadLayer(const armnn::IConnectableLayer* layer,
                           const armnn::PadDescriptor& PadDescriptor,
                           const char* name = nullptr);

    void SerializePermuteLayer(const armnn::IConnectableLayer* layer,
                               const armnn::PermuteDescriptor& PermuteDescriptor,
                               const char* name = nullptr);

    void SerializePooling2dLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::Pooling2dDescriptor& pooling2dDescriptor,
                                 const char* name = nullptr);

    void SerializePooling3dLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::Pooling3dDescriptor& pooling3dDescriptor,
                                 const char* name = nullptr);

    void SerializePreluLayer(const armnn::IConnectableLayer* layer,
                             const char* name = nullptr);

    void SerializeQuantizeLayer(const armnn::IConnectableLayer* layer,
                                const char* name = nullptr);

    void SerializeQLstmLayer(const armnn::IConnectableLayer* layer,
                             const armnn::QLstmDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name = nullptr);

    void SerializeQuantizedLstmLayer(const armnn::IConnectableLayer* layer,
                                     const std::vector<armnn::ConstTensor>& constants,
                                     const char* name = nullptr);

    void SerializeRankLayer(const armnn::IConnectableLayer* layer,
                            const char* name = nullptr);

    void SerializeReduceLayer(const armnn::IConnectableLayer* layer,
                          const armnn::ReduceDescriptor& reduceDescriptor,
                          const char* name = nullptr);

    void SerializeReshapeLayer(const armnn::IConnectableLayer* layer,
                               const armnn::ReshapeDescriptor& reshapeDescriptor,
                               const char* name = nullptr);

    void SerializeResizeLayer(const armnn::IConnectableLayer* layer,
                              const armnn::ResizeDescriptor& resizeDescriptor,
                              const char* name = nullptr);

    void SerializeSliceLayer(const armnn::IConnectableLayer* layer,
                             const armnn::SliceDescriptor& sliceDescriptor,
                             const char* name = nullptr);

    void SerializeSoftmaxLayer(const armnn::IConnectableLayer* layer,
                               const armnn::SoftmaxDescriptor& softmaxDescriptor,
                               const char* name = nullptr);

    void SerializeSpaceToBatchNdLayer(const armnn::IConnectableLayer* layer,
                                      const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                      const char* name = nullptr);

    void SerializeSpaceToDepthLayer(const armnn::IConnectableLayer* layer,
                                    const armnn::SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                    const char* name = nullptr);

    void SerializeNormalizationLayer(const armnn::IConnectableLayer* layer,
                                     const armnn::NormalizationDescriptor& normalizationDescriptor,
                                     const char* name = nullptr);

    void SerializeShapeLayer(const armnn::IConnectableLayer* layer,
                             const char* name = nullptr);

    void SerializeSplitterLayer(const armnn::IConnectableLayer* layer,
                                const armnn::ViewsDescriptor& viewsDescriptor,
                                const char* name = nullptr);

    void SerializeStandInLayer(const armnn::IConnectableLayer* layer,
                               const armnn::StandInDescriptor& standInDescriptor,
                               const char* name = nullptr);

    void SerializeStackLayer(const armnn::IConnectableLayer* layer,
                             const armnn::StackDescriptor& stackDescriptor,
                             const char* name = nullptr);

    void SerializeStridedSliceLayer(const armnn::IConnectableLayer* layer,
                                    const armnn::StridedSliceDescriptor& stridedSliceDescriptor,
                                    const char* name = nullptr);

    void SerializeSubtractionLayer(const armnn::IConnectableLayer* layer,
                                   const char* name = nullptr);

    void SerializeSwitchLayer(const armnn::IConnectableLayer* layer,
                              const char* name = nullptr);

    void SerializeTransposeConvolution2dLayer(const armnn::IConnectableLayer* layer,
                                              const armnn::TransposeConvolution2dDescriptor& descriptor,
                                              const std::vector<armnn::ConstTensor>& constants,
                                              const char* = nullptr);

    void SerializeTransposeLayer(const armnn::IConnectableLayer* layer,
                                 const armnn::TransposeDescriptor& descriptor,
                                 const char* name = nullptr);

    void SerializeUnidirectionalSequenceLstmLayer(const armnn::IConnectableLayer* layer,
                                                  const armnn::UnidirectionalSequenceLstmDescriptor& descriptor,
                                                  const std::vector<armnn::ConstTensor>& constants,
                                                  const char* name = nullptr);
};



class ISerializer::SerializerImpl
{
public:
    SerializerImpl() = default;
    ~SerializerImpl() = default;

    /// Serializes the network to ArmNN SerializedGraph.
    /// @param [in] inNetwork The network to be serialized.
    void Serialize(const armnn::INetwork& inNetwork);

    /// Serializes the SerializedGraph to the stream.
    /// @param [stream] the stream to save to
    /// @return true if graph is Serialized to the Stream, false otherwise
    bool SaveSerializedToStream(std::ostream& stream);

private:

    /// Visitor to contruct serialized network
    SerializerStrategy m_SerializerStrategy;
};

} //namespace armnnSerializer
