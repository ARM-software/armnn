//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <armnn/INetwork.hpp>

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Layer.hpp"

namespace armnn
{
class Graph;

/// Private implementation of INetwork.
class Network final : public INetwork
{
public:
    Network();
    ~Network();

    const Graph& GetGraph() const { return *m_Graph; }

    Status PrintGraph() override;

    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name=nullptr) override;

    IConnectableLayer* AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                              const char* name = nullptr) override;

    IConnectableLayer* AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                      const char* name = nullptr) override;

    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const Optional<ConstTensor>& biases,
                                             const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const ConstTensor& biases,
                                             const char* name = nullptr) override;

    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const Optional<ConstTensor>& biases,
        const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddDepthwiseConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddDepthwiseConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) override;

    IConnectableLayer* AddDequantizeLayer(const char* name = nullptr) override;

    IConnectableLayer* AddDetectionPostProcessLayer(
        const DetectionPostProcessDescriptor& descriptor,
        const ConstTensor& anchors,
        const char* name = nullptr) override;

    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddFullyConnectedLayer overload is deprecated")
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("This AddFullyConnectedLayer overload is deprecated")
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const ConstTensor& biases,
                                              const char* name = nullptr) override;

    IConnectableLayer* AddGatherLayer(const char* name = nullptr) override;

    IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                       const char* name = nullptr) override;

    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr) override;

    IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
        const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use AddConcatLayer instead")
    IConnectableLayer* AddMergerLayer(const MergerDescriptor& mergerDescriptor,
                                      const char* name = nullptr) override;

    IConnectableLayer* AddAdditionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr) override;

    IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
                                                  const ConstTensor&                  mean,
                                                  const ConstTensor&                  variance,
                                                  const ConstTensor&                  beta,
                                                  const ConstTensor&                  gamma,
                                                  const char*                         name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use AddResizeLayer instead")
    IConnectableLayer* AddResizeBilinearLayer(const ResizeBilinearDescriptor& resizeDesc,
                                              const char* name = nullptr) override;

    IConnectableLayer* AddResizeLayer(const ResizeDescriptor& resizeDescriptor,
                                      const char* name = nullptr) override;

    IConnectableLayer* AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                               const char* name = nullptr) override;

    IConnectableLayer* AddConstantLayer(const ConstTensor& input, const char* name = nullptr) override;

    IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr) override;

    IConnectableLayer* AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                              const char* name = nullptr) override;

    IConnectableLayer* AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                            const char* name = nullptr) override;

    IConnectableLayer* AddFloorLayer(const char* name = nullptr) override;

    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr) override;

    IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                    const LstmInputParams& params,
                                    const char* name = nullptr) override;

    IConnectableLayer* AddDivisionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMaximumLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr) override;

    IConnectableLayer* AddPadLayer(const PadDescriptor& padDescriptor, const char* name = nullptr) override;

    IConnectableLayer* AddQuantizeLayer(const char* name = nullptr) override;

    IConnectableLayer* AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                            const char* name = nullptr) override;

    IConnectableLayer* AddMinimumLayer(const char* name = nullptr) override;

    IConnectableLayer* AddGreaterLayer(const char* name = nullptr) override;

    IConnectableLayer* AddEqualLayer(const char* name = nullptr) override;

    IConnectableLayer* AddRsqrtLayer(const char* name = nullptr) override;

    IConnectableLayer* AddMergeLayer(const char* name = nullptr) override;

    IConnectableLayer* AddSwitchLayer(const char* name = nullptr) override;

    IConnectableLayer* AddPreluLayer(const char* name = nullptr) override;

    IConnectableLayer* AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                      const ConstTensor& weights,
                                                      const Optional<ConstTensor>& biases,
                                                      const char* name = nullptr) override;

    IConnectableLayer* AddStackLayer(const StackDescriptor& stackDescriptor,
                                     const char* name = nullptr) override;

    IConnectableLayer* AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                             const char* name = nullptr) override;

    void Accept(ILayerVisitor& visitor) const override;

private:
    IConnectableLayer* AddFullyConnectedLayerImpl(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  const char* name);

    IConnectableLayer* AddConvolution2dLayerImpl(const Convolution2dDescriptor& convolution2dDescriptor,
                                                 const ConstTensor& weights,
                                                 const Optional<ConstTensor>& biases,
                                                 const char* name);

    IConnectableLayer* AddDepthwiseConvolution2dLayerImpl(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const Optional<ConstTensor>& biases,
        const char* name);

    std::unique_ptr<Graph> m_Graph;
};

class OptimizedNetwork final : public IOptimizedNetwork
{
public:
    OptimizedNetwork(std::unique_ptr<Graph> graph);
    ~OptimizedNetwork();

    Status PrintGraph() override;
    Status SerializeToDot(std::ostream& stream) const override;

    Graph& GetGraph() { return *m_Graph; }

private:
    std::unique_ptr<Graph> m_Graph;
};



struct OptimizationResult
{
    bool m_Warning;
    bool m_Error;

    OptimizationResult()
        : m_Warning(false)
        , m_Error(false)
    {}
};

using BackendsMap = std::map<BackendId, std::unique_ptr<class IBackendInternal>>;

BackendsMap CreateSupportedBackends(TensorHandleFactoryRegistry& handleFactoryRegistry,
                                    struct BackendSettings& backendSettings);

OptimizationResult SelectTensorHandleStrategy(Graph& optGraph,
                                              BackendsMap& backends,
                                              TensorHandleFactoryRegistry& registry,
                                              Optional<std::vector<std::string>&> errMessages);

} // namespace armnn
