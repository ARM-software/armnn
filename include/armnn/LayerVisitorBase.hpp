//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{

struct VisitorThrowingPolicy
{
    static void Apply(const std::string& errorMessage = "") { throw UnimplementedException(errorMessage); }
};

struct VisitorNoThrowPolicy
{
    static void Apply(const std::string&) {}
};

/// Visitor base class with empty implementations.
template<typename DefaultPolicy>
class LayerVisitorBase : public ILayerVisitor
{
protected:
    LayerVisitorBase() {}
    virtual ~LayerVisitorBase() {}

public:
    void VisitAbsLayer(const IConnectableLayer*,
                       const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitActivationLayer(const IConnectableLayer*,
                              const ActivationDescriptor&,
                              const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitAdditionLayer(const IConnectableLayer*,
                            const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitArgMinMaxLayer(const IConnectableLayer*,
                             const ArgMinMaxDescriptor&,
                             const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitBatchNormalizationLayer(const IConnectableLayer*,
                                      const BatchNormalizationDescriptor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const ConstTensor&,
                                      const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitBatchToSpaceNdLayer(const IConnectableLayer*,
                                  const BatchToSpaceNdDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitComparisonLayer(const IConnectableLayer*,
                              const ComparisonDescriptor&,
                              const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitConcatLayer(const IConnectableLayer*,
                          const ConcatDescriptor&,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitConstantLayer(const IConnectableLayer*,
                            const ConstTensor&,
                            const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitConvolution2dLayer(const IConnectableLayer*,
                                 const Convolution2dDescriptor&,
                                 const ConstTensor&,
                                 const Optional<ConstTensor>&,
                                 const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitDepthToSpaceLayer(const IConnectableLayer*,
                                const DepthToSpaceDescriptor&,
                                const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer*,
                                          const DepthwiseConvolution2dDescriptor&,
                                          const ConstTensor&,
                                          const Optional<ConstTensor>&,
                                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitDequantizeLayer(const IConnectableLayer*,
                              const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitDetectionPostProcessLayer(const IConnectableLayer*,
                                        const DetectionPostProcessDescriptor&,
                                        const ConstTensor&,
                                        const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitDivisionLayer(const IConnectableLayer*,
                            const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitElementwiseUnaryLayer(const IConnectableLayer*,
                                    const ElementwiseUnaryDescriptor&,
                                    const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitEqualLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitFillLayer(const IConnectableLayer*,
                        const FillDescriptor&,
                        const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitFloorLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitFullyConnectedLayer(const IConnectableLayer*,
                                  const FullyConnectedDescriptor&,
                                  const ConstTensor&,
                                  const Optional<ConstTensor>&,
                                  const char*) override { DefaultPolicy::Apply(__func__); }

    ARMNN_DEPRECATED_MSG("Use VisitGatherLayer with descriptor instead")
    void VisitGatherLayer(const IConnectableLayer*,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitGatherLayer(const IConnectableLayer*,
                          const GatherDescriptor&,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitGreaterLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitInputLayer(const IConnectableLayer*,
                         LayerBindingId,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitInstanceNormalizationLayer(const IConnectableLayer*,
                                         const InstanceNormalizationDescriptor&,
                                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitL2NormalizationLayer(const IConnectableLayer*,
                                   const L2NormalizationDescriptor&,
                                   const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitLogSoftmaxLayer(const IConnectableLayer*,
                              const LogSoftmaxDescriptor&,
                              const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitLogicalBinaryLayer(const IConnectableLayer*,
                                 const LogicalBinaryDescriptor&,
                                 const char*) override {DefaultPolicy::Apply(__func__); }

    void VisitLstmLayer(const IConnectableLayer*,
                        const LstmDescriptor&,
                        const LstmInputParams&,
                        const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMaximumLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMeanLayer(const IConnectableLayer*,
                        const MeanDescriptor&,
                        const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMergeLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMergerLayer(const IConnectableLayer*,
                          const MergerDescriptor&,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMinimumLayer(const IConnectableLayer*,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitMultiplicationLayer(const IConnectableLayer*,
                                  const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitNormalizationLayer(const IConnectableLayer*,
                                 const NormalizationDescriptor&,
                                 const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitOutputLayer(const IConnectableLayer*,
                          LayerBindingId,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitPadLayer(const IConnectableLayer*,
                       const PadDescriptor&,
                       const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitPermuteLayer(const IConnectableLayer*,
                           const PermuteDescriptor&,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitPooling2dLayer(const IConnectableLayer*,
                             const Pooling2dDescriptor&,
                             const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitPreluLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitQuantizeLayer(const IConnectableLayer*,
                            const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitQLstmLayer(const IConnectableLayer*,
                         const QLstmDescriptor&,
                         const LstmInputParams&,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitQuantizedLstmLayer(const IConnectableLayer*,
                                 const QuantizedLstmInputParams&,
                                 const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitRankLayer(const IConnectableLayer*,
                        const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitReduceLayer(const IConnectableLayer*,
                          const ReduceDescriptor&,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitReshapeLayer(const IConnectableLayer*,
                           const ReshapeDescriptor&,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitResizeBilinearLayer(const IConnectableLayer*,
                                  const ResizeBilinearDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitResizeLayer(const IConnectableLayer*,
                          const ResizeDescriptor&,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitRsqrtLayer(const IConnectableLayer*,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSliceLayer(const IConnectableLayer*,
                         const SliceDescriptor&,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSoftmaxLayer(const IConnectableLayer*,
                           const SoftmaxDescriptor&,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSpaceToBatchNdLayer(const IConnectableLayer*,
                                  const SpaceToBatchNdDescriptor&,
                                  const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSpaceToDepthLayer(const IConnectableLayer*,
                                const SpaceToDepthDescriptor&,
                                const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSplitterLayer(const IConnectableLayer*,
                            const ViewsDescriptor&,
                            const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitStackLayer(const IConnectableLayer*,
                         const StackDescriptor&,
                         const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitStandInLayer(const IConnectableLayer*,
                           const StandInDescriptor&,
                           const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitStridedSliceLayer(const IConnectableLayer*,
                                const StridedSliceDescriptor&,
                                const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSubtractionLayer(const IConnectableLayer*,
                               const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitSwitchLayer(const IConnectableLayer*,
                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitTransposeConvolution2dLayer(const IConnectableLayer*,
                                          const TransposeConvolution2dDescriptor&,
                                          const ConstTensor&,
                                          const Optional<ConstTensor>&,
                                          const char*) override { DefaultPolicy::Apply(__func__); }

    void VisitTransposeLayer(const IConnectableLayer*,
                             const TransposeDescriptor&,
                             const char*) override { DefaultPolicy::Apply(__func__); }

};

} // namespace armnn
