//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{
// Abstract base class with do nothing implementations for all layer visit methods
class TestLayerVisitor : public ILayerVisitor
{
protected:
    virtual ~TestLayerVisitor() {};

    void CheckLayerName(const char* name);

private:
    const char* m_LayerName;

public:
    explicit TestLayerVisitor(const char* name) : m_LayerName(name) {};

    virtual void VisitInputLayer(const IConnectableLayer* layer,
                                 LayerBindingId id,
                                 const char* name = nullptr) {};

    virtual void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                         const Convolution2dDescriptor& convolution2dDescriptor,
                                         const ConstTensor& weights,
                                         const char* name = nullptr) {};

    virtual void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                         const Convolution2dDescriptor& convolution2dDescriptor,
                                         const ConstTensor& weights,
                                         const ConstTensor& biases,
                                         const char* name = nullptr) {};

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                  const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const char* name = nullptr) {};

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                  const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const ConstTensor& biases,
                                                  const char* name = nullptr) {};

    virtual void VisitDetectionPostProcessLayer(const IConnectableLayer* layer,
                                                const DetectionPostProcessDescriptor& descriptor,
                                                const char* name = nullptr) {};

    virtual void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                          const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                          const ConstTensor& weights,
                                          const char* name = nullptr) {};

    virtual void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                          const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                          const ConstTensor& weights,
                                          const ConstTensor& biases,
                                          const char* name = nullptr) {};

    virtual void VisitPermuteLayer(const IConnectableLayer* layer,
                                   const PermuteDescriptor& permuteDescriptor,
                                   const char* name = nullptr) {};

    virtual void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                          const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                          const char* name = nullptr) {};

    virtual void VisitPooling2dLayer(const IConnectableLayer* layer,
                                     const Pooling2dDescriptor& pooling2dDescriptor,
                                     const char* name = nullptr) {};

    virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                      const ActivationDescriptor& activationDescriptor,
                                      const char* name = nullptr) {};

    virtual void VisitNormalizationLayer(const IConnectableLayer* layer,
                                         const NormalizationDescriptor& normalizationDescriptor,
                                         const char* name = nullptr) {};

    virtual void VisitSoftmaxLayer(const IConnectableLayer* layer,
                                   const SoftmaxDescriptor& softmaxDescriptor,
                                   const char* name = nullptr) {};

    virtual void VisitSplitterLayer(const IConnectableLayer* layer,
                                    const ViewsDescriptor& splitterDescriptor,
                                    const char* name = nullptr) {};

    virtual void VisitMergerLayer(const IConnectableLayer* layer,
                                  const OriginsDescriptor& mergerDescriptor,
                                  const char* name = nullptr) {};

    virtual void VisitAdditionLayer(const IConnectableLayer* layer,
                                    const char* name = nullptr) {};

    virtual void VisitMultiplicationLayer(const IConnectableLayer* layer,
                                          const char* name = nullptr) {};

    virtual void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                              const BatchNormalizationDescriptor& desc,
                                              const ConstTensor& mean,
                                              const ConstTensor& variance,
                                              const ConstTensor& beta,
                                              const ConstTensor& gamma,
                                              const char* name = nullptr) {};

    virtual void VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                          const ResizeBilinearDescriptor& resizeDesc,
                                          const char* name = nullptr) {};

    virtual void VisitL2NormalizationLayer(const IConnectableLayer* layer,
                                           const L2NormalizationDescriptor& desc,
                                           const char* name = nullptr) {};

    virtual void VisitConstantLayer(const IConnectableLayer* layer,
                                    const ConstTensor& input,
                                    const char* name = nullptr) {};

    virtual void VisitReshapeLayer(const IConnectableLayer* layer,
                                   const ReshapeDescriptor& reshapeDescriptor,
                                   const char* name = nullptr) {};

    virtual void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                          const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                          const char* name = nullptr) {};

    virtual void VisitFloorLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) {};

    virtual void VisitOutputLayer(const IConnectableLayer* layer,
                                  LayerBindingId id,
                                  const char* name = nullptr) {};

    virtual void VisitLstmLayer(const IConnectableLayer* layer,
                                const LstmDescriptor& descriptor,
                                const LstmInputParams& params,
                                const char* name = nullptr) {};

    virtual void VisitDivisionLayer(const IConnectableLayer* layer,
                                    const char* name = nullptr) {};

    virtual void VisitSubtractionLayer(const IConnectableLayer* layer,
                                       const char* name = nullptr) {};

    virtual void VisitMaximumLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) {};

    virtual void VisitMeanLayer(const IConnectableLayer* layer,
                                const MeanDescriptor& meanDescriptor,
                                const char* name = nullptr) {};

    virtual void VisitPadLayer(const IConnectableLayer* layer,
                               const PadDescriptor& padDescriptor,
                               const char* name = nullptr) {};

    virtual void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                        const StridedSliceDescriptor& stridedSliceDescriptor,
                                        const char* name = nullptr) {};

    virtual void VisitMinimumLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) {};

    virtual void VisitGreaterLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) {};

    virtual void VisitEqualLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) {};

    virtual void VisitRsqrtLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) {};

    virtual void VisitGatherLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) {};
};

} //namespace armnn
