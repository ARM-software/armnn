//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendId.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/Types.hpp>

namespace armnn
{

// This handle calls its own IsXXXLayerSupported() functions which then call the polymorphic
// ILayerSupport::IsXXXLayerSupported() at the framework level so there is no risk of VTable misalignment.
// This is to make ILayerSupport in its abstract form a solely Backend interface alongside a
// separate ABI stable frontend class free of virtual functions via an added layer of indirection.
class LayerSupportHandle
{
public:
    explicit LayerSupportHandle(std::shared_ptr<ILayerSupport> layerSupport)
        : m_LayerSupport(std::move(layerSupport)), m_BackendId(Compute::Undefined) {};

    explicit LayerSupportHandle(std::shared_ptr<ILayerSupport> layerSupport, const BackendId& backendId)
        : m_LayerSupport(std::move(layerSupport)), m_BackendId(backendId) {};

    bool IsBackendRegistered() const;

    ARMNN_DEPRECATED_MSG("Use IsElementwiseUnarySupported instead")
    bool IsAbsSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsActivationSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsArgMinMaxSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const ArgMinMaxDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsBatchNormalizationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsBatchToSpaceNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const BatchToSpaceNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsCastSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsComparisonSupported(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               const ComparisonDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                           const TensorInfo& output,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConstantSupported(const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConvertBf16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConvertFp32ToBf16Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDebugSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDepthToSpaceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const DepthToSpaceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDepthwiseConvolutionSupported(
            const TensorInfo& input,
            const TensorInfo& output,
            const DepthwiseConvolution2dDescriptor& descriptor,
            const TensorInfo& weights,
            const Optional<TensorInfo>& biases,
            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDequantizeSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                         const TensorInfo& scores,
                                         const TensorInfo& anchors,
                                         const TensorInfo& detectionBoxes,
                                         const TensorInfo& detectionClasses,
                                         const TensorInfo& detectionScores,
                                         const TensorInfo& numDetections,
                                         const DetectionPostProcessDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDilatedDepthwiseConvolutionSupported(
            const TensorInfo& input,
            const TensorInfo& output,
            const DepthwiseConvolution2dDescriptor& descriptor,
            const TensorInfo& weights,
            const Optional<TensorInfo>& biases,
            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsDivisionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsElementwiseUnarySupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const ElementwiseUnaryDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsComparisonSupported instead")
    bool IsEqualSupported(const TensorInfo& input0,
                          const TensorInfo& input1,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsFakeQuantizationSupported(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsFillSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const FillDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsFloorSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsGatherSupported with descriptor instead")
    bool IsGatherSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsGatherSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           const GatherDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsComparisonSupported instead")
    bool IsGreaterSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& ouput,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsInstanceNormalizationSupported(
            const TensorInfo& input,
            const TensorInfo& output,
            const InstanceNormalizationDescriptor& descriptor,
            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsL2NormalizationSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const L2NormalizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsLogicalBinarySupported(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  const LogicalBinaryDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsLogicalUnarySupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const ElementwiseUnaryDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsLogSoftmaxSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const LogSoftmaxDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsLstmSupported(const TensorInfo& input,
                         const TensorInfo& outputStateIn,
                         const TensorInfo& cellStateIn,
                         const TensorInfo& scratchBuffer,
                         const TensorInfo& outputStateOut,
                         const TensorInfo& cellStateOut,
                         const TensorInfo& output,
                         const LstmDescriptor& descriptor,
                         const LstmInputParamsInfo& paramsInfo,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMaximumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMeanSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const MeanDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMemCopySupported(const TensorInfo& input,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMemImportSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMergeSupported(const TensorInfo& input0,
                          const TensorInfo& input1,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsConcatSupported instead")
    bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                           const TensorInfo& output,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMinimumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsMultiplicationSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsNormalizationSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsPadSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const PadDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsPermuteSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsPreCompiledSupported(const TensorInfo& input,
                                const PreCompiledDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsPreluSupported(const TensorInfo& input,
                          const TensorInfo& alpha,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsQuantizeSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsQLstmSupported(const TensorInfo& input,
                          const TensorInfo& previousOutputIn,
                          const TensorInfo& previousCellStateIn,
                          const TensorInfo& outputStateOut,
                          const TensorInfo& cellStateOut,
                          const TensorInfo& output,
                          const QLstmDescriptor& descriptor,
                          const LstmInputParamsInfo& paramsInfo,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsQuantizedLstmSupported(const TensorInfo& input,
                                  const TensorInfo& previousCellStateIn,
                                  const TensorInfo& previousOutputIn,
                                  const TensorInfo& cellStateOut,
                                  const TensorInfo& output,
                                  const QuantizedLstmInputParamsInfo& paramsInfo,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsRankSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsReduceSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ReduceDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsReshapeSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const ReshapeDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsResizeSupported instead")
    bool IsResizeBilinearSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsResizeSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ResizeDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsElementwiseUnarySupported instead")
    bool IsRsqrtSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSliceSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const SliceDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSpaceToBatchNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const SpaceToBatchNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSpaceToDepthSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const SpaceToDepthDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    ARMNN_DEPRECATED_MSG("Use IsSplitterSupported with outputs instead")
    bool IsSplitterSupported(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSplitterSupported(const TensorInfo& input,
                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                          const TensorInfo& output,
                          const StackDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                            const std::vector<const TensorInfo*>& outputs,
                            const StandInDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());


    bool IsStridedSliceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const StridedSliceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSubtractionSupported(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsSwitchSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output0,
                           const TensorInfo& output1,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsTransposeConvolution2dSupported(
            const TensorInfo& input,
            const TensorInfo& output,
            const TransposeConvolution2dDescriptor& descriptor,
            const TensorInfo& weights,
            const Optional<TensorInfo>& biases,
            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

    bool IsTransposeSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const TransposeDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional());

private:
    std::shared_ptr<ILayerSupport> m_LayerSupport;
    const BackendId m_BackendId;
};

/// Convenience function to retrieve the ILayerSupportHandle for a backend
LayerSupportHandle GetILayerSupportByBackendId(const armnn::BackendId& backend);

/// Convenience function to check a capability on a backend
bool IsCapabilitySupported(const armnn::BackendId& backend, armnn::BackendCapability capability);

}
