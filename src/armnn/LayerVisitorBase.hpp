//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{

struct VisitorThrowingPolicy
{
    static void Apply() { throw UnimplementedException(); }
};

struct VisitorNoThrowPolicy
{
    static void Apply() {}
};

// Visitor base class with empty implementations.
template<typename DefaultPolicy>
class LayerVisitorBase : public ILayerVisitor
{
protected:
    LayerVisitorBase() {}
    virtual ~LayerVisitorBase() {}

public:
    void VisitInputLayer(const IConnectableLayer*,
                         LayerBindingId,
                         const char*) override { DefaultPolicy::Apply(); }

    void VisitConvolution2dLayer(const IConnectableLayer*,
                                 const Convolution2dDescriptor&,
                                 const ConstTensor&,
                                 const Optional<ConstTensor>&,
                                 const char*) override { DefaultPolicy::Apply(); }

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                          const DepthwiseConvolution2dDescriptor&,
                                          const ConstTensor&,
                                          const Optional<ConstTensor>&,
                                          const char*) override { DefaultPolicy::Apply(); }

    void VisitDetectionPostProcessLayer(const IConnectableLayer*,
                                        const DetectionPostProcessDescriptor&,
                                        const ConstTensor&,
                                        const char*) override { DefaultPolicy::Apply(); }

    void VisitFullyConnectedLayer(const IConnectableLayer*,
                                  const FullyConnectedDescriptor&,
                                  const ConstTensor&,
                                  const Optional<ConstTensor>&,
                                  const char*) override { DefaultPolicy::Apply(); }

    void VisitPermuteLayer(const IConnectableLayer*,
                           const PermuteDescriptor&,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitBatchToSpaceNdLayer(const IConnectableLayer*,
                                  const BatchToSpaceNdDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(); }

    void VisitPooling2dLayer(const IConnectableLayer*,
                             const Pooling2dDescriptor&,
                             const char*) override { DefaultPolicy::Apply(); }

    void VisitActivationLayer(const IConnectableLayer*,
                              const ActivationDescriptor&,
                              const char*) override { DefaultPolicy::Apply(); }

    void VisitNormalizationLayer(const IConnectableLayer*,
                                 const NormalizationDescriptor&,
                                 const char*) override { DefaultPolicy::Apply(); }

    void VisitSoftmaxLayer(const IConnectableLayer*,
                           const SoftmaxDescriptor&,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitSplitterLayer(const IConnectableLayer*,
                            const ViewsDescriptor&,
                            const char*) override { DefaultPolicy::Apply(); }

    void VisitMergerLayer(const IConnectableLayer*,
                          const OriginsDescriptor&,
                          const char*) override { DefaultPolicy::Apply(); }

    void VisitAdditionLayer(const IConnectableLayer*,
                            const char*) override { DefaultPolicy::Apply(); }

    void VisitMultiplicationLayer(const IConnectableLayer*,
                                  const char*) override { DefaultPolicy::Apply(); }

    void VisitBatchNormalizationLayer(const IConnectableLayer*,
                                      const BatchNormalizationDescriptor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const char*) override { DefaultPolicy::Apply(); }

    void VisitResizeBilinearLayer(const IConnectableLayer*,
                                  const ResizeBilinearDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(); }

    void VisitL2NormalizationLayer(const IConnectableLayer*,
                                   const L2NormalizationDescriptor&,
                                   const char*) override { DefaultPolicy::Apply(); }

    void VisitConstantLayer(const IConnectableLayer*,
                            const ConstTensor&,
                            const char*) override { DefaultPolicy::Apply(); }

    void VisitReshapeLayer(const IConnectableLayer*,
                           const ReshapeDescriptor&,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitSpaceToBatchNdLayer(const IConnectableLayer*,
                                  const SpaceToBatchNdDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(); }

    void VisitFloorLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(); }

    void VisitOutputLayer(const IConnectableLayer*,
                          LayerBindingId id,
                          const char*) override { DefaultPolicy::Apply(); }

    void VisitLstmLayer(const IConnectableLayer*,
                        const LstmDescriptor&,
                        const LstmInputParams&,
                        const char*) override { DefaultPolicy::Apply(); }

    void VisitDivisionLayer(const IConnectableLayer*,
                            const char*) override { DefaultPolicy::Apply(); }

    void VisitSubtractionLayer(const IConnectableLayer*,
                               const char*) override { DefaultPolicy::Apply(); }

    void VisitMaximumLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitMeanLayer(const IConnectableLayer*,
                        const MeanDescriptor&,
                        const char*) override { DefaultPolicy::Apply(); }

    void VisitPadLayer(const IConnectableLayer*,
                       const PadDescriptor&,
                       const char*) override { DefaultPolicy::Apply(); }

    void VisitStridedSliceLayer(const IConnectableLayer*,
                                const StridedSliceDescriptor&,
                                const char*) override { DefaultPolicy::Apply(); }

    void VisitMinimumLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitGreaterLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(); }

    void VisitEqualLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(); }

    void VisitRsqrtLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(); }

    void VisitGatherLayer(const IConnectableLayer*,
                          const char*) override { DefaultPolicy::Apply(); }
};

} //namespace armnn

