//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/backends/ILayerSupport.hpp>

namespace armnn
{

class LayerSupportBase : public ILayerSupport
{
public:

    bool IsLayerSupported(const LayerType& type,
                          const std::vector<TensorInfo>& infos,
                          const BaseDescriptor& descriptor,
                          const Optional<LstmInputParamsInfo>& lstmParamsInfo = EmptyOptional(),
                          const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmParamsInfo = EmptyOptional(),
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsActivationSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsArgMinMaxSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const ArgMinMaxDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsBatchNormalizationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsBatchToSpaceNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const BatchToSpaceNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsCastSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsChannelShuffleSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const ChannelShuffleDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsComparisonSupported(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               const ComparisonDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                           const TensorInfo& output,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConstantSupported(const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConvertFp32ToFp16Supported(
            const TensorInfo& input,
            const TensorInfo& output,
            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsConvolution3dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution3dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDebugSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDepthToSpaceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const DepthToSpaceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDequantizeSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDetectionPostProcessSupported(const TensorInfo& boxEncodings,
                                         const TensorInfo& scores,
                                         const TensorInfo& anchors,
                                         const TensorInfo& detectionBoxes,
                                         const TensorInfo& detectionClasses,
                                         const TensorInfo& detectionScores,
                                         const TensorInfo& numDetections,
                                         const DetectionPostProcessDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDilatedDepthwiseConvolutionSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const DepthwiseConvolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported =
                                                    EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsDivisionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsElementwiseUnarySupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const ElementwiseUnaryDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsFakeQuantizationSupported(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    virtual bool IsFillSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const FillDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsFloorSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsGatherSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           const GatherDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsInstanceNormalizationSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const InstanceNormalizationDescriptor& descriptor,
        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsL2NormalizationSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const L2NormalizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsLogicalBinarySupported(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  const LogicalBinaryDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsLogicalUnarySupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const ElementwiseUnaryDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsLogSoftmaxSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const LogSoftmaxDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsLstmSupported(const TensorInfo& input,
                         const TensorInfo& outputStateIn,
                         const TensorInfo& cellStateIn,
                         const TensorInfo& scratchBuffer,
                         const TensorInfo& outputStateOut,
                         const TensorInfo& cellStateOut,
                         const TensorInfo& output,
                         const LstmDescriptor& descriptor,
                         const LstmInputParamsInfo& paramsInfo,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsMaximumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsMeanSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const MeanDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMemCopySupported(const TensorInfo& input,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMemImportSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMergeSupported(const TensorInfo& input0,
                          const TensorInfo& input1,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsMinimumSupported(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsMultiplicationSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsNormalizationSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPadSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const PadDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPermuteSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPooling3dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling3dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPreCompiledSupported(const TensorInfo& input,
                                const PreCompiledDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsPreluSupported(const TensorInfo& input,
                          const TensorInfo& alpha,
                          const TensorInfo& output,
                          Optional<std::string &> reasonIfUnsupported) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsQuantizeSupported(const TensorInfo& input,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsQLstmSupported(const TensorInfo& input,
                          const TensorInfo& previousOutputIn,
                          const TensorInfo& previousCellStateIn,
                          const TensorInfo& outputStateOut,
                          const TensorInfo& cellStateOut,
                          const TensorInfo& output,
                          const QLstmDescriptor& descriptor,
                          const LstmInputParamsInfo& paramsInfo,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsQuantizedLstmSupported(const TensorInfo& input,
                                  const TensorInfo& previousCellStateIn,
                                  const TensorInfo& previousOutputIn,
                                  const TensorInfo& cellStateOut,
                                  const TensorInfo& output,
                                  const QuantizedLstmInputParamsInfo& paramsInfo,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsRankSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsReduceSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ReduceDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsReshapeSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const ReshapeDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsResizeSupported(const TensorInfo& input,
                           const TensorInfo& output,
                           const ResizeDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsShapeSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSliceSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          const SliceDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSpaceToBatchNdSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const SpaceToBatchNdDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSpaceToDepthSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const SpaceToDepthDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSplitterSupported(const TensorInfo& input,
                             const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                          const TensorInfo& output,
                          const StackDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsStandInSupported(const std::vector<const TensorInfo*>& inputs,
                            const std::vector<const TensorInfo*>& outputs,
                            const StandInDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsStridedSliceSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const StridedSliceDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSubtractionSupported(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsSwitchSupported(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output0,
                           const TensorInfo& output1,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsTransposeConvolution2dSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const TransposeConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsTransposeSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const TransposeDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("This method is deprecated. Use IsLayerSupported instead.", "23.08")
    bool IsUnidirectionalSequenceLstmSupported(
        const TensorInfo& input,
        const TensorInfo& outputStateIn,
        const TensorInfo& cellStateIn,
        const TensorInfo& outputStateOut,
        const TensorInfo& cellStateOut,
        const TensorInfo& output,
        const LstmDescriptor& descriptor,
        const LstmInputParamsInfo& paramsInfo,
        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

};

} // namespace armnn
