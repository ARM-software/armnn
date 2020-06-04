//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/LayerVisitorBase.hpp"
#include "StaticRangeVisitor.hpp"
#include "NetworkQuantizationScheme.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>

#include <unordered_map>

namespace armnn
{

// Forward declaration
class StaticRangeVisitor;

/// Visitor object for quantizing layers in a network
class QuantizerVisitor : public LayerVisitorBase<VisitorThrowingPolicy>
{
public:
    QuantizerVisitor(const RangeTracker& rangeTracker,
                     const IQuantizationScheme* quantizationScheme,
                     bool preserveType = false);

    ~QuantizerVisitor() = default;

    /// Functions to quantize the individual layers, overridden from ILayerVisitor
    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    void VisitAbsLayer(const IConnectableLayer* layer, const char* name = nullptr) override;

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char* name = nullptr) override;

    void VisitAdditionLayer(const IConnectableLayer* layer, const char* name = nullptr) override;

    void VisitArgMinMaxLayer(const IConnectableLayer* layer,
                             const ArgMinMaxDescriptor& argMinMaxDescriptor,
                             const char* name = nullptr) override;

    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override;

    void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                  const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                  const char* name = nullptr) override;

    void VisitComparisonLayer(const IConnectableLayer* layer,
                              const ComparisonDescriptor& comparisonDescriptor,
                              const char* name = nullptr) override;

    void VisitConcatLayer(const IConnectableLayer* layer,
                          const OriginsDescriptor& originsDescriptor,
                          const char* name = nullptr) override;

    void VisitConstantLayer(const IConnectableLayer* layer,
                            const ConstTensor& input,
                            const char* name = nullptr) override;

    void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                 const ConstTensor& weights,
                                 const Optional<ConstTensor>& biases,
                                 const char* name = nullptr) override;

    void VisitDepthToSpaceLayer(const IConnectableLayer* layer,
                                const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                const char* name = nullptr) override;

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                          const DepthwiseConvolution2dDescriptor& desc,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char* name = nullptr) override;

    void VisitElementwiseUnaryLayer(const IConnectableLayer* layer,
                                    const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                    const char* name = nullptr) override;

    void VisitFillLayer(const IConnectableLayer* layer,
                        const FillDescriptor& desc,
                        const char* name) override;

    void VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                  const FullyConnectedDescriptor& desc,
                                  const ConstTensor& weights,
                                  const Optional<ConstTensor>& biases,
                                  const char *name = nullptr)  override;

    void VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr) override;

    void VisitInstanceNormalizationLayer(const IConnectableLayer* layer,
                                         const InstanceNormalizationDescriptor& instanceNormalizationDescriptor,
                                         const char* name = nullptr) override;

    void VisitLogSoftmaxLayer(const IConnectableLayer* layer,
                              const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                              const char* name = nullptr) override;

    void VisitMeanLayer(const IConnectableLayer* layer,
                        const MeanDescriptor& meanDescriptor,
                        const char* name = nullptr) override;

    void VisitMultiplicationLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) override;

    void VisitNormalizationLayer(const IConnectableLayer* layer,
                                 const NormalizationDescriptor& normalizationDescriptor,
                                 const char* name = nullptr) override;

    void VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr)  override;

    void VisitPadLayer(const IConnectableLayer*,
                       const PadDescriptor&,
                       const char* name = nullptr) override;

    void VisitPermuteLayer(const IConnectableLayer* layer,
                           const PermuteDescriptor& permuteDescriptor,
                           const char* name = nullptr) override;

    void VisitPooling2dLayer(const IConnectableLayer* layer,
                             const Pooling2dDescriptor& pooling2dDescriptor,
                             const char* name = nullptr) override;

    void VisitPreluLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override;

    void VisitReshapeLayer(const IConnectableLayer* layer,
                           const ReshapeDescriptor& reshapeDescriptor,
                           const char* name = nullptr) override;

    void VisitResizeLayer(const IConnectableLayer* layer,
                          const ResizeDescriptor& resizeDescriptor,
                          const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitResizeLayer instead")
    void VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                  const ResizeBilinearDescriptor& resizeDesc,
                                  const char* name = nullptr) override;

    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    void VisitRsqrtLayer(const IConnectableLayer*,
                         const char* name = nullptr) override;

    void VisitSliceLayer(const IConnectableLayer* layer,
                         const SliceDescriptor& sliceDescriptor,
                         const char* name = nullptr) override;

    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override;

    void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                  const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                  const char* name = nullptr) override;

    void VisitSpaceToDepthLayer(const IConnectableLayer* layer,
                                const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                const char* name = nullptr) override;

    void VisitSplitterLayer(const IConnectableLayer* layer,
                            const SplitterDescriptor& splitterDescriptor,
                            const char* name = nullptr) override;

    void VisitStackLayer(const IConnectableLayer* layer,
                         const StackDescriptor& stackDescriptor,
                         const char* name = nullptr) override;

    void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                const StridedSliceDescriptor& stridedSliceDescriptor,
                                const char* name = nullptr) override;

    void VisitSubtractionLayer(const IConnectableLayer* layer,
                               const char* name = nullptr) override;

    void VisitTransposeConvolution2dLayer(const IConnectableLayer* layer,
                                          const TransposeConvolution2dDescriptor& descriptor,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char* name = nullptr) override;

    void VisitTransposeLayer(const IConnectableLayer* layer,
                             const TransposeDescriptor& descriptor,
                             const char* name = nullptr) override;

    /// Extract the quantized network
    INetworkPtr RetrieveFinalNetwork() { return std::move(m_QuantizedNetwork); }

private:
    /// Connects the layer to preceeding layers and sets the quantization parameters based on recorded ranges
    void SetQuantizedInputConnections(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer);

    /// Record the guids so we can easily find the layers later
    void RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* qLayer);

    /// Sets the bias quantization scale based on input and weight scales
    ConstTensor CreateQuantizedBias(const IConnectableLayer* srcLayer,
                                    const ConstTensor& weights,
                                    const Optional<ConstTensor>& biases,
                                    std::vector<int32_t>& weightsBacking);

    /// Reference to the static range visitor used to retrieve the quantization ranges
    const RangeTracker& m_Ranges;

    /// Quantized version of the model we are building up
    INetworkPtr m_QuantizedNetwork;

    /// Mapping from input network guids to quantized network guids
    std::unordered_map<LayerGuid, LayerGuid> m_OriginalToQuantizedGuidMap;

    /// Mapping from guid to layer in quantized network
    std::unordered_map<LayerGuid, IConnectableLayer*> m_QuantizedGuidToLayerMap;

    const IQuantizationScheme* m_QuantizationScheme;

    const bool m_PreserveType;
};

} //namespace armnn
