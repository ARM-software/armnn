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
    virtual void VisitInputLayer(const IConnectableLayer*,
                                 LayerBindingId,
                                 const char*) { DefaultPolicy::Apply(); }

    virtual void VisitConvolution2dLayer(const IConnectableLayer*,
                                         const Convolution2dDescriptor&,
                                         const ConstTensor&,
                                         const char*) { DefaultPolicy::Apply(); }

    virtual void VisitConvolution2dLayer(const IConnectableLayer*,
                                         const Convolution2dDescriptor&,
                                         const ConstTensor&,
                                         const ConstTensor&,
                                         const char*) { DefaultPolicy::Apply(); }

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                                  const DepthwiseConvolution2dDescriptor&,
                                                  const ConstTensor& ,
                                                  const char*) { DefaultPolicy::Apply(); }

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                                  const DepthwiseConvolution2dDescriptor&,
                                                  const ConstTensor&,
                                                  const ConstTensor&,
                                                  const char*) { DefaultPolicy::Apply(); }

    virtual void VisitDetectionPostProcessLayer(const IConnectableLayer*,
                                                const DetectionPostProcessDescriptor&,
                                                const ConstTensor&,
                                                const char*) { DefaultPolicy::Apply(); }

    virtual void VisitFullyConnectedLayer(const IConnectableLayer*,
                                          const FullyConnectedDescriptor&,
                                          const ConstTensor&,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitFullyConnectedLayer(const IConnectableLayer*,
                                          const FullyConnectedDescriptor&,
                                          const ConstTensor&,
                                          const ConstTensor&,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitPermuteLayer(const IConnectableLayer*,
                                   const PermuteDescriptor&,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitBatchToSpaceNdLayer(const IConnectableLayer*,
                                          const BatchToSpaceNdDescriptor&,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitPooling2dLayer(const IConnectableLayer*,
                                     const Pooling2dDescriptor&,
                                     const char*) { DefaultPolicy::Apply(); }

    virtual void VisitActivationLayer(const IConnectableLayer*,
                                      const ActivationDescriptor&,
                                      const char*) { DefaultPolicy::Apply(); }

    virtual void VisitNormalizationLayer(const IConnectableLayer*,
                                         const NormalizationDescriptor&,
                                         const char*) { DefaultPolicy::Apply(); }

    virtual void VisitSoftmaxLayer(const IConnectableLayer*,
                                   const SoftmaxDescriptor&,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitSplitterLayer(const IConnectableLayer*,
                                    const ViewsDescriptor&,
                                    const char*) { DefaultPolicy::Apply(); }

    virtual void VisitMergerLayer(const IConnectableLayer*,
                                  const OriginsDescriptor&,
                                  const char*) { DefaultPolicy::Apply(); }

    virtual void VisitAdditionLayer(const IConnectableLayer*,
                                    const char*) { DefaultPolicy::Apply(); }

    virtual void VisitMultiplicationLayer(const IConnectableLayer*,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitBatchNormalizationLayer(const IConnectableLayer*,
                                              const BatchNormalizationDescriptor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const char*) { DefaultPolicy::Apply(); }

    virtual void VisitResizeBilinearLayer(const IConnectableLayer*,
                                          const ResizeBilinearDescriptor&,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitL2NormalizationLayer(const IConnectableLayer*,
                                           const L2NormalizationDescriptor&,
                                           const char*) { DefaultPolicy::Apply(); }

    virtual void VisitConstantLayer(const IConnectableLayer*,
                                    const ConstTensor&,
                                    const char*) { DefaultPolicy::Apply(); }

    virtual void VisitReshapeLayer(const IConnectableLayer*,
                                   const ReshapeDescriptor&,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitSpaceToBatchNdLayer(const IConnectableLayer*,
                                          const SpaceToBatchNdDescriptor&,
                                          const char*) { DefaultPolicy::Apply(); }

    virtual void VisitFloorLayer(const IConnectableLayer*,
                                 const char*) { DefaultPolicy::Apply(); }

    virtual void VisitOutputLayer(const IConnectableLayer*,
                                  LayerBindingId id,
                                  const char*) { DefaultPolicy::Apply(); }

    virtual void VisitLstmLayer(const IConnectableLayer*,
                                const LstmDescriptor&,
                                const LstmInputParams&,
                                const char*) { DefaultPolicy::Apply(); }

    virtual void VisitDivisionLayer(const IConnectableLayer*,
                                    const char*) { DefaultPolicy::Apply(); }

    virtual void VisitSubtractionLayer(const IConnectableLayer*,
                                       const char*) { DefaultPolicy::Apply(); }

    virtual void VisitMaximumLayer(const IConnectableLayer*,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitMeanLayer(const IConnectableLayer*,
                                const MeanDescriptor&,
                                const char*) { DefaultPolicy::Apply(); }

    virtual void VisitPadLayer(const IConnectableLayer*,
                               const PadDescriptor&,
                               const char*) { DefaultPolicy::Apply(); }

    virtual void VisitStridedSliceLayer(const IConnectableLayer*,
                                        const StridedSliceDescriptor&,
                                        const char*) { DefaultPolicy::Apply(); }

    virtual void VisitMinimumLayer(const IConnectableLayer*,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitGreaterLayer(const IConnectableLayer*,
                                   const char*) { DefaultPolicy::Apply(); }

    virtual void VisitEqualLayer(const IConnectableLayer*,
                                 const char*) { DefaultPolicy::Apply(); }

    virtual void VisitRsqrtLayer(const IConnectableLayer*,
                                 const char*) { DefaultPolicy::Apply(); }

    virtual void VisitGatherLayer(const IConnectableLayer*,
                                  const char*) { DefaultPolicy::Apply(); }
};

} //namespace armnn

