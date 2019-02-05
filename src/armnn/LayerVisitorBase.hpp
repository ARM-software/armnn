//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{

// Visitor base class with empty implementations.
class LayerVisitorBase : public ILayerVisitor
{
protected:
    LayerVisitorBase() {}
    virtual ~LayerVisitorBase() {}

public:
    virtual void VisitInputLayer(const IConnectableLayer*,
                                 LayerBindingId,
                                 const char*) {}

    virtual void VisitConvolution2dLayer(const IConnectableLayer*,
                                         const Convolution2dDescriptor&,
                                         const ConstTensor&,
                                         const char*) {}

    virtual void VisitConvolution2dLayer(const IConnectableLayer*,
                                         const Convolution2dDescriptor&,
                                         const ConstTensor&,
                                         const ConstTensor&,
                                         const char*) {}

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                                  const DepthwiseConvolution2dDescriptor&,
                                                  const ConstTensor& ,
                                                  const char*) {}

    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                                  const DepthwiseConvolution2dDescriptor&,
                                                  const ConstTensor&,
                                                  const ConstTensor&,
                                                  const char*) {}

    virtual void VisitDetectionPostProcessLayer(const IConnectableLayer*,
                                                const DetectionPostProcessDescriptor&,
                                                const char*) {}

    virtual void VisitFullyConnectedLayer(const IConnectableLayer*,
                                          const FullyConnectedDescriptor&,
                                          const ConstTensor&,
                                          const char*) {}

    virtual void VisitFullyConnectedLayer(const IConnectableLayer*,
                                          const FullyConnectedDescriptor&,
                                          const ConstTensor&,
                                          const ConstTensor&,
                                          const char*) {}

    virtual void VisitPermuteLayer(const IConnectableLayer*,
                                   const PermuteDescriptor&,
                                   const char*) {}

    virtual void VisitBatchToSpaceNdLayer(const IConnectableLayer*,
                                          const BatchToSpaceNdDescriptor&,
                                          const char*) {}

    virtual void VisitPooling2dLayer(const IConnectableLayer*,
                                     const Pooling2dDescriptor&,
                                     const char*) {}

    virtual void VisitActivationLayer(const IConnectableLayer*,
                                      const ActivationDescriptor&,
                                      const char*) {}

    virtual void VisitNormalizationLayer(const IConnectableLayer*,
                                         const NormalizationDescriptor&,
                                         const char*) {}

    virtual void VisitSoftmaxLayer(const IConnectableLayer*,
                                   const SoftmaxDescriptor&,
                                   const char*) {}

    virtual void VisitSplitterLayer(const IConnectableLayer*,
                                    const ViewsDescriptor&,
                                    const char*) {}

    virtual void VisitMergerLayer(const IConnectableLayer*,
                                  const OriginsDescriptor&,
                                  const char*) {}

    virtual void VisitAdditionLayer(const IConnectableLayer*,
                                    const char*) {}

    virtual void VisitMultiplicationLayer(const IConnectableLayer*,
                                          const char*) {}

    virtual void VisitBatchNormalizationLayer(const IConnectableLayer*,
                                              const BatchNormalizationDescriptor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const ConstTensor&,
                                              const char*) {}

    virtual void VisitResizeBilinearLayer(const IConnectableLayer*,
                                          const ResizeBilinearDescriptor&,
                                          const char*) {}

    virtual void VisitL2NormalizationLayer(const IConnectableLayer*,
                                           const L2NormalizationDescriptor&,
                                           const char*) {}

    virtual void VisitConstantLayer(const IConnectableLayer*,
                                    const ConstTensor&,
                                    const char*) {}

    virtual void VisitReshapeLayer(const IConnectableLayer*,
                                   const ReshapeDescriptor&,
                                   const char*) {}

    virtual void VisitSpaceToBatchNdLayer(const IConnectableLayer*,
                                          const SpaceToBatchNdDescriptor&,
                                          const char*) {}

    virtual void VisitFloorLayer(const IConnectableLayer*,
                                 const char*) {}

    virtual void VisitOutputLayer(const IConnectableLayer*,
                                  LayerBindingId id,
                                  const char*) {}
    
    virtual void VisitLstmLayer(const IConnectableLayer*,
                                const LstmDescriptor&,
                                const LstmInputParams&,
                                const char*) {}
    
    virtual void VisitDivisionLayer(const IConnectableLayer*,
                                    const char*) {}
    
    virtual void VisitSubtractionLayer(const IConnectableLayer*,
                                       const char*) {}
    
    virtual void VisitMaximumLayer(const IConnectableLayer*,
                                   const char*) {}
    
    virtual void VisitMeanLayer(const IConnectableLayer*,
                                const MeanDescriptor&,
                                const char*) {}
    
    virtual void VisitPadLayer(const IConnectableLayer*,
                               const PadDescriptor&,
                               const char*) {}
    
    virtual void VisitStridedSliceLayer(const IConnectableLayer*,
                                        const StridedSliceDescriptor&,
                                        const char*) {}
    
    virtual void VisitMinimumLayer(const IConnectableLayer*,
                                   const char*) {}
    
    virtual void VisitGreaterLayer(const IConnectableLayer*,
                                   const char*) {}
    
    virtual void VisitEqualLayer(const IConnectableLayer*,
                                 const char*) {}
    
    virtual void VisitRsqrtLayer(const IConnectableLayer*,
                                 const char*) {}
    
    virtual void VisitGatherLayer(const IConnectableLayer*,
                                  const char*) {}
    
};

} //namespace armnn

