//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Deprecated.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/LstmParams.hpp>
#include <armnn/Optional.hpp>
#include <armnn/QuantizedLstmParams.hpp>

#include <cctype>
#include <functional>
#include <memory>
#include <vector>

namespace armnn
{

class TensorInfo;

class ILayerSupport
{
protected:
    ILayerSupport() {}
    virtual ~ILayerSupport() {}

public:
    virtual bool IsActivationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const ActivationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsAdditionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsBatchNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TensorInfo& mean,
                                               const TensorInfo& var,
                                               const TensorInfo& beta,
                                               const TensorInfo& gamma,
                                               const BatchNormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsBatchToSpaceNdSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const BatchToSpaceNdDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsConcatSupported(const std::vector<const TensorInfo*> inputs,
                                   const TensorInfo& output,
                                   const OriginsDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsConstantSupported(const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsConvolution2dSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const Convolution2dDescriptor& descriptor,
                                          const TensorInfo& weights,
                                          const Optional<TensorInfo>& biases,
                                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDebugSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDepthwiseConvolutionSupported(
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const DepthwiseConvolution2dDescriptor& descriptor,
                     const TensorInfo& weights,
                     const Optional<TensorInfo>& biases,
                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDequantizeSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDetectionPostProcessSupported(
                     const TensorInfo& input0,
                     const TensorInfo& input1,
                     const DetectionPostProcessDescriptor& descriptor,
                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDilatedDepthwiseConvolutionSupported(
                    const TensorInfo& input,
                    const TensorInfo& output,
                    const DepthwiseConvolution2dDescriptor& descriptor,
                    const TensorInfo& weights,
                    const Optional<TensorInfo>& biases,
                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsDivisionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsEqualSupported(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsFakeQuantizationSupported(const TensorInfo& input,
                                             const FakeQuantizationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsFloorSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsFullyConnectedSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TensorInfo& weights,
                                           const TensorInfo& biases,
                                           const FullyConnectedDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsGatherSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsGreaterSupported(const TensorInfo& input0,
                                    const TensorInfo& input1,
                                    const TensorInfo& ouput,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsInputSupported(const TensorInfo& input,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsL2NormalizationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const L2NormalizationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsLstmSupported(const TensorInfo& input,
                                 const TensorInfo& outputStateIn,
                                 const TensorInfo& cellStateIn,
                                 const TensorInfo& scratchBuffer,
                                 const TensorInfo& outputStateOut,
                                 const TensorInfo& cellStateOut,
                                 const TensorInfo& output,
                                 const LstmDescriptor& descriptor,
                                 const LstmInputParamsInfo& paramsInfo,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMaximumSupported(const TensorInfo& input0,
                                    const TensorInfo& input1,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMeanSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const MeanDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMemCopySupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMemImportSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMergeSupported(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    ARMNN_DEPRECATED_MSG("Use IsConcatSupported instead")
    virtual bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                   const TensorInfo& output,
                                   const OriginsDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMinimumSupported(const TensorInfo& input0,
                                    const TensorInfo& input1,
                                    const TensorInfo& ouput,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsMultiplicationSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsNormalizationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const NormalizationDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsOutputSupported(const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsPadSupported(const TensorInfo& input,
                                const TensorInfo& output,
                                const PadDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsPermuteSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const PermuteDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsPooling2dSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const Pooling2dDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsPreCompiledSupported(const TensorInfo& input,
                                        const PreCompiledDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsPreluSupported(const TensorInfo& input,
                                  const TensorInfo& alpha,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsQuantizeSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsQuantizedLstmSupported(const TensorInfo& input,
                                          const TensorInfo& previousCellStateIn,
                                          const TensorInfo& previousOutputIn,
                                          const TensorInfo& cellStateOut,
                                          const TensorInfo& output,
                                          const QuantizedLstmInputParamsInfo& paramsInfo,
                                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsReshapeSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const ReshapeDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    ARMNN_DEPRECATED_MSG("Use IsResizeSupported instead")
    virtual bool IsResizeBilinearSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsResizeSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const ResizeDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsRsqrtSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSoftmaxSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const SoftmaxDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSpaceToBatchNdSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const SpaceToBatchNdDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSpaceToDepthSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const SpaceToDepthDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    ARMNN_DEPRECATED_MSG("Use IsSplitterSupported with outputs instead")
    virtual bool IsSplitterSupported(const TensorInfo& input,
                                     const ViewsDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSplitterSupported(const TensorInfo& input,
                                     const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                                     const ViewsDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsStackSupported(const std::vector<const TensorInfo*>& inputs,
                                  const TensorInfo& output,
                                  const StackDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsStridedSliceSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const StridedSliceDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSubtractionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsSwitchSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output0,
                                   const TensorInfo& output1,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

    virtual bool IsTransposeConvolution2dSupported(
        const TensorInfo& input,
        const TensorInfo& output,
        const TransposeConvolution2dDescriptor& descriptor,
        const TensorInfo& weights,
        const Optional<TensorInfo>& biases,
        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const = 0;

}; // class ILayerSupport

using ILayerSupportSharedPtr = std::shared_ptr<ILayerSupport>;

} // namespace armnn
