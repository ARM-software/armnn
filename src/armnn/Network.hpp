//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/INetwork.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <Graph.hpp>
#include <Layer.hpp>
#include <OptimizedNetworkImpl.hpp>
#include <armnn/backends/SubgraphView.hpp>

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace armnn
{
class Graph;

using NetworkImplPtr = std::unique_ptr<NetworkImpl, void (*)(NetworkImpl* network)>;

/// Private implementation of INetwork.
class NetworkImpl
{
public:
    NetworkImpl(NetworkOptions networkOptions = {});
    ~NetworkImpl();

    const Graph& GetGraph() const
    { return *m_Graph; }

    Status PrintGraph();

    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name = nullptr);

    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
                                          const char* name = nullptr);

    IConnectableLayer* AddAdditionLayer(const char* name = nullptr);

    IConnectableLayer* AddArgMinMaxLayer(const ArgMinMaxDescriptor& desc,
                                         const char* name = nullptr);

    IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                  const ConstTensor& mean,
                                                  const ConstTensor& variance,
                                                  const ConstTensor& beta,
                                                  const ConstTensor& gamma,
                                                  const char* name = nullptr);

    IConnectableLayer* AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddCastLayer(const char* name = nullptr);

    IConnectableLayer* AddChannelShuffleLayer(const ChannelShuffleDescriptor& channelShuffleDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                          const char* name = nullptr);

    IConnectableLayer* AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const Optional<ConstTensor>& biases,
                                             const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This AddConvolution2dLayer overload is deprecated", "22.11")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const char* name = nullptr);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This AddConvolution2dLayer overload is deprecated", "22.11")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const ConstTensor& biases,
                                             const char* name = nullptr);

    IConnectableLayer* AddConvolution3dLayer(const Convolution3dDescriptor& convolution3dDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddConstantLayer(const ConstTensor& input, const char* name = nullptr);

    IConnectableLayer* AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                            const char* name = nullptr);

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const Optional<ConstTensor>& biases,
        const char* name = nullptr);

    IConnectableLayer* AddDequantizeLayer(const char* name = nullptr);

    IConnectableLayer* AddDetectionPostProcessLayer(
        const DetectionPostProcessDescriptor& descriptor,
        const ConstTensor& anchors,
        const char* name = nullptr);

    IConnectableLayer* AddDivisionLayer(const char* name = nullptr);

    IConnectableLayer* AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                const char* name = nullptr);

    IConnectableLayer* AddMergeLayer(const char* name = nullptr);

    IConnectableLayer* AddFillLayer(const FillDescriptor& fillDescriptor,
                                    const char* name = nullptr);

    IConnectableLayer* AddFloorLayer(const char* name = nullptr);

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const Optional<ConstTensor>& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char* name = nullptr);

    IConnectableLayer* AddGatherLayer(const GatherDescriptor& gatherDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                     const char* name = nullptr);

    IConnectableLayer* AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                               const char* name = nullptr);

    IConnectableLayer* AddLogSoftmaxLayer(const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                          const char* name = nullptr);

    IConnectableLayer* AddLogicalBinaryLayer(const LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                    const LstmInputParams& params,
                                    const char* name = nullptr);

    IConnectableLayer* AddMaximumLayer(const char* name = nullptr);

    IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr);

    IConnectableLayer* AddMinimumLayer(const char* name = nullptr);

    IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr);

    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
                                             const char* name = nullptr);

    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr);

    IConnectableLayer* AddPadLayer(const PadDescriptor& padDescriptor, const char* name = nullptr);

    IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddPooling3dLayer(const Pooling3dDescriptor& pooling3dDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddPrecompiledLayer(const PreCompiledDescriptor& preCompiledDescriptor,
                                           CompiledBlobPtr compiledBlobPtr,
                                           const Optional<BackendId>& backend,
                                           const char* name = nullptr);

    IConnectableLayer* AddPreluLayer(const char* name = nullptr);

    IConnectableLayer* AddQuantizeLayer(const char* name = nullptr);

    IConnectableLayer* AddQLstmLayer(const QLstmDescriptor& descriptor,
                                     const LstmInputParams& params,
                                     const char* name = nullptr);

    IConnectableLayer* AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                             const char* name = nullptr);

    IConnectableLayer* AddRankLayer(const char* name = nullptr);

    IConnectableLayer* AddReduceLayer(const ReduceDescriptor& reduceDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddResizeLayer(const ResizeDescriptor& resizeDescriptor,
                                      const char* name = nullptr);

    IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddShapeLayer(const char* name = nullptr);

    IConnectableLayer* AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name = nullptr);

    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
                                        const char* name = nullptr);

    IConnectableLayer* AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                              const char* name = nullptr);

    IConnectableLayer* AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                            const char* name = nullptr);

    IConnectableLayer* AddStackLayer(const StackDescriptor& stackDescriptor,
                                     const char* name = nullptr);

    IConnectableLayer* AddStandInLayer(const StandInDescriptor& descriptor,
                                       const char* name = nullptr);

    IConnectableLayer* AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                            const char* name = nullptr);

    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr);

    IConnectableLayer* AddSwitchLayer(const char* name = nullptr);

    IConnectableLayer* AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                      const ConstTensor& weights,
                                                      const Optional<ConstTensor>& biases,
                                                      const char* name = nullptr);

    IConnectableLayer* AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                         const char* name = nullptr);

    IConnectableLayer* AddUnidirectionalSequenceLstmLayer(const UnidirectionalSequenceLstmDescriptor& descriptor,
                                                          const LstmInputParams& params,
                                                          const char* name = nullptr);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    void Accept(ILayerVisitor& visitor) const;
    ARMNN_NO_DEPRECATE_WARN_END

    void ExecuteStrategy(IStrategy& strategy) const;

private:
    IConnectableLayer* AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
                                                 const ConstTensor& weights,
                                                 const Optional<ConstTensor>& biases,
                                                 const char* name);

    IConnectableLayer* AddDepthwiseConvolution2dLayerImpl(const DepthwiseConvolution2dDescriptor& conv2dDescriptor,
                                                          const ConstTensor& weights,
                                                          const Optional<ConstTensor>& biases,
                                                          const char* name);

    bool GetShapeInferenceMethod();
    NetworkOptions m_NetworkOptions;

    std::unique_ptr<Graph> m_Graph;
    ModelOptions           m_ModelOptions;
};

struct OptimizationResult
{
    bool m_Warning;
    bool m_Error;

    OptimizationResult(bool warning, bool error)
        : m_Warning(warning), m_Error(error)
    {}

    OptimizationResult()
        : OptimizationResult(false, false)
    {}

    bool IsOk() const
    { return !m_Warning && !m_Error; }
    bool IsWarningOnly() const
    { return m_Warning && !m_Error; }
    bool IsError() const
    { return m_Error; }

};

using BackendsMap = std::map<BackendId, std::unique_ptr<class IBackendInternal>>;

BackendsMap CreateSupportedBackends(TensorHandleFactoryRegistry& handleFactoryRegistry,
                                    struct BackendSettings& backendSettings);

OptimizationResult SelectTensorHandleStrategy(Graph& optGraph,
                                              BackendsMap& backends,
                                              TensorHandleFactoryRegistry& registry,
                                              bool importEnabled,
                                              Optional<std::vector<std::string>&> errMessages);

OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  Graph::Iterator& firstLayer,
                                  Graph::Iterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages);


OptimizationResult AssignBackends(OptimizedNetworkImpl* optNetObjPtr,
                                  BackendSettings& backendSettings,
                                  SubgraphView::IConnectableLayerIterator& firstLayer,
                                  SubgraphView::IConnectableLayerIterator& lastLayer,
                                  Optional<std::vector<std::string>&> errMessages);

} // namespace armnn
