//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ILayerVisitor.hpp>
#include <armnn/Descriptors.hpp>

namespace armnn
{
// Abstract base class with do nothing implementations for all layer visit methods
class TestLayerVisitor : public ILayerVisitor
{
protected:
    virtual ~TestLayerVisitor() {}

    void CheckLayerName(const char* name);

    void CheckLayerPointer(const IConnectableLayer* layer);

    void CheckConstTensors(const ConstTensor& expected, const ConstTensor& actual);

    void CheckOptionalConstTensors(const Optional<ConstTensor>& expected, const Optional<ConstTensor>& actual);

private:
    const char* m_LayerName;

public:
    explicit TestLayerVisitor(const char* name) : m_LayerName(name)
    {
        if (name == nullptr)
        {
            m_LayerName = "";
        }
    }

    void VisitInputLayer(const IConnectableLayer* layer,
                         LayerBindingId id,
                         const char* name = nullptr) override {}

    void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                 const ConstTensor& weights,
                                 const Optional<ConstTensor>& biases,
                                 const char* name = nullptr) override {}

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                          const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char* name = nullptr) override {}

    void VisitDetectionPostProcessLayer(const IConnectableLayer* layer,
                                        const DetectionPostProcessDescriptor& descriptor,
                                        const ConstTensor& anchors,
                                        const char* name = nullptr) override {}

    void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                  const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                  const ConstTensor& weights,
                                  const Optional<ConstTensor>& biases,
                                  const char* name = nullptr) override {}

    void VisitPermuteLayer(const IConnectableLayer* layer,
                           const PermuteDescriptor& permuteDescriptor,
                           const char* name = nullptr) override {}

    void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                  const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                  const char* name = nullptr) override {}

    void VisitPooling2dLayer(const IConnectableLayer* layer,
                             const Pooling2dDescriptor& pooling2dDescriptor,
                             const char* name = nullptr) override {}

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char* name = nullptr) override {}

    void VisitNormalizationLayer(const IConnectableLayer* layer,
                                 const NormalizationDescriptor& normalizationDescriptor,
                                 const char* name = nullptr) override {}

    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override {}

    void VisitSplitterLayer(const IConnectableLayer* layer,
                            const ViewsDescriptor& splitterDescriptor,
                            const char* name = nullptr) override {}

    void VisitMergerLayer(const IConnectableLayer* layer,
                          const OriginsDescriptor& mergerDescriptor,
                          const char* name = nullptr) override {}

    void VisitAdditionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override {}

    void VisitMultiplicationLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) override {}

    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override {}

    void VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                  const ResizeBilinearDescriptor& resizeDesc,
                                  const char* name = nullptr) override {}

    void VisitL2NormalizationLayer(const IConnectableLayer* layer,
                                   const L2NormalizationDescriptor& desc,
                                   const char* name = nullptr) override {}

    void VisitConstantLayer(const IConnectableLayer* layer,
                            const ConstTensor& input,
                            const char* name = nullptr) override {}

    void VisitReshapeLayer(const IConnectableLayer* layer,
                           const ReshapeDescriptor& reshapeDescriptor,
                           const char* name = nullptr) override {}

    void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                  const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                  const char* name = nullptr) override {}

    void VisitFloorLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {}

    void VisitOutputLayer(const IConnectableLayer* layer,
                          LayerBindingId id,
                          const char* name = nullptr) override {}

    void VisitLstmLayer(const IConnectableLayer* layer,
                        const LstmDescriptor& descriptor,
                        const LstmInputParams& params,
                        const char* name = nullptr) override {}

    void VisitDivisionLayer(const IConnectableLayer* layer,
                            const char* name = nullptr) override {}

    void VisitSubtractionLayer(const IConnectableLayer* layer,
                               const char* name = nullptr) override {}

    void VisitMaximumLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {}

    void VisitMeanLayer(const IConnectableLayer* layer,
                        const MeanDescriptor& meanDescriptor,
                        const char* name = nullptr) override {}

    void VisitPadLayer(const IConnectableLayer* layer,
                       const PadDescriptor& padDescriptor,
                       const char* name = nullptr) override {}

    void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                const StridedSliceDescriptor& stridedSliceDescriptor,
                                const char* name = nullptr) override {}

    void VisitMinimumLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {}

    void VisitGreaterLayer(const IConnectableLayer* layer,
                           const char* name = nullptr) override {}

    void VisitEqualLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {}

    void VisitRsqrtLayer(const IConnectableLayer* layer,
                         const char* name = nullptr) override {}

    void VisitGatherLayer(const IConnectableLayer* layer,
                          const char* name = nullptr) override {}
};

} //namespace armnn
