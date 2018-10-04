//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Optional.hpp>

namespace armnn
{

bool IsActivationSupported(Compute compute,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported = nullptr,
                           size_t reasonIfUnsupportedMaxLength = 1024);

bool IsAdditionSupported(Compute compute,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsBatchNormalizationSupported(Compute compute,
                                   const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& mean,
                                   const TensorInfo& var,
                                   const TensorInfo& beta,
                                   const TensorInfo& gamma,
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported = nullptr,
                                   size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConstantSupported(Compute compute,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvertFp16ToFp32Supported(Compute compute,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvertFp32ToFp16Supported(Compute compute,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvolution2dSupported(Compute compute,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

bool IsDepthwiseConvolutionSupported(Compute compute,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsDivisionSupported(Compute compute,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSubtractionSupported(Compute compute,
                            const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            char* reasonIfUnsupported = nullptr,
                            size_t reasonIfUnsupportedMaxLength = 1024);

bool IsInputSupported(Compute compute,
                      const TensorInfo& input,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFullyConnectedSupported(Compute compute,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsL2NormalizationSupported(Compute compute,
                                const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                char* reasonIfUnsupported = nullptr,
                                size_t reasonIfUnsupportedMaxLength = 1024);

bool IsLstmSupported(Compute compute, const TensorInfo& input, const TensorInfo& outputStateIn,
                     const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                     const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                     const TensorInfo& output, const LstmDescriptor& descriptor,
                     const TensorInfo& inputToForgetWeights, const TensorInfo& inputToCellWeights,
                     const TensorInfo& inputToOutputWeights, const TensorInfo& recurrentToForgetWeights,
                     const TensorInfo& recurrentToCellWeights, const TensorInfo& recurrentToOutputWeights,
                     const TensorInfo& forgetGateBias, const TensorInfo& cellBias,
                     const TensorInfo& outputGateBias, const TensorInfo* inputToInputWeights,
                     const TensorInfo* recurrentToInputWeights, const TensorInfo* cellToInputWeights,
                     const TensorInfo* inputGateBias, const TensorInfo* projectionWeights,
                     const TensorInfo* projectionBias, const TensorInfo* cellToForgetWeights,
                     const TensorInfo* cellToOutputWeights, char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMergerSupported(Compute compute,
                       const std::vector<const TensorInfo*> inputs,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMultiplicationSupported(Compute compute,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsNormalizationSupported(Compute compute,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

bool IsOutputSupported(Compute compute,
                       const TensorInfo& output,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPermuteSupported(Compute compute,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPooling2dSupported(Compute compute,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported = nullptr,
                          size_t reasonIfUnsupportedMaxLength = 1024);

bool IsResizeBilinearSupported(Compute compute,
                               const TensorInfo& input,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSoftmaxSupported(Compute compute,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSplitterSupported(Compute compute,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFakeQuantizationSupported(Compute compute,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported = nullptr,
                                 size_t reasonIfUnsupportedMaxLength = 1024);

bool IsReshapeSupported(Compute compute,
                        const TensorInfo& input,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFloorSupported(Compute compute,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMeanSupported(Compute compute,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPadSupported(Compute compute,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const PadDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

}
