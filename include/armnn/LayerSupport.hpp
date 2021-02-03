//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Deprecated.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Optional.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>
#include "LstmParams.hpp"
#include "QuantizedLstmParams.hpp"

namespace armnn
{

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsActivationSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported = nullptr,
                           size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsAdditionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsBatchNormalizationSupported(const BackendId& backend,
                                   const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& mean,
                                   const TensorInfo& var,
                                   const TensorInfo& beta,
                                   const TensorInfo& gamma,
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported = nullptr,
                                   size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsBatchToSpaceNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const BatchToSpaceNdDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsConcatSupported(const BackendId& backend,
                       const std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsConstantSupported(const BackendId& backend,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsConvertFp16ToFp32Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsConvertFp32ToFp16Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsConvolution2dSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsDebugSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsDepthwiseConvolutionSupported(const BackendId& backend,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsDequantizeSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           char* reasonIfUnsupported = nullptr,
                           size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsDivisionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsEqualSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsFakeQuantizationSupported(const BackendId& backend,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported = nullptr,
                                 size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsFloorSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsFullyConnectedSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsGreaterSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsInputSupported(const BackendId& backend,
                      const TensorInfo& input,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsL2NormalizationSupported(const BackendId& backend,
                                const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                char* reasonIfUnsupported = nullptr,
                                size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsLstmSupported(const BackendId& backend, const TensorInfo& input, const TensorInfo& outputStateIn,
                     const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                     const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                     const TensorInfo& output, const LstmDescriptor& descriptor,
                     const LstmInputParamsInfo& paramsInfo, char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMaximumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnSupported = nullptr,
                        size_t reasonIfUnSupportedMaxLength = 0);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMeanSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMemCopySupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMergeSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
ARMNN_DEPRECATED_MSG("Use IsConcatSupported instead")
bool IsMergerSupported(const BackendId& backend,
                       const std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMinimumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsMultiplicationSupported(const BackendId& backend,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsNormalizationSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsOutputSupported(const BackendId& backend,
                       const TensorInfo& output,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsPadSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const PadDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsPermuteSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsPreCompiledSupported(const BackendId& backend,
                            const TensorInfo& input,
                            char* reasonIfUnsupported = nullptr,
                            size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsPreluSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& alpha,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsPooling2dSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported = nullptr,
                          size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsQuantizedLstmSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& previousCellStateIn,
                              const TensorInfo& previousOutputIn,
                              const TensorInfo& cellStateOut,
                              const TensorInfo& output,
                              const QuantizedLstmInputParamsInfo& paramsInfo,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsReduceSupported(const BackendId& backend,
                       const TensorInfo& input,
                       const TensorInfo& output,
                       const ReduceDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsReshapeSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const ReshapeDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
ARMNN_DEPRECATED_MSG("Use IsResizeSupported instead")
bool IsResizeBilinearSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsResizeSupported(const BackendId& backend,
                       const TensorInfo& input,
                       const TensorInfo& output,
                       const ResizeDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsRsqrtSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSoftmaxSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSpaceToBatchNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const SpaceToBatchNdDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSpaceToDepthSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const SpaceToDepthDescriptor& descriptor,
                             char* reasonIfUnsupported = nullptr,
                             size_t reasonIfUnsupportedMaxLength = 1024);

ARMNN_DEPRECATED_MSG("Use IsSplitterSupported with outputs instead")
bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsStackSupported(const BackendId& backend,
                      const std::vector<const TensorInfo*> inputs,
                      const TensorInfo& output,
                      const StackDescriptor& descriptor,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsStridedSliceSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const StridedSliceDescriptor& descriptor,
                             char* reasonIfUnsupported = nullptr,
                             size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSubtractionSupported(const BackendId& backend,
                            const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            char* reasonIfUnsupported = nullptr,
                            size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsSwitchSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output0,
                       const TensorInfo& output1,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

/// Deprecated in favor of IBackend and ILayerSupport interfaces
bool IsTransposeConvolution2dSupported(const BackendId& backend,
                                       const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TransposeConvolution2dDescriptor& descriptor,
                                       const TensorInfo& weights,
                                       const Optional<TensorInfo>& biases,
                                       char* reasonIfUnsupported = nullptr,
                                       size_t reasonIfUnsupportedMaxLength = 1024);

}
