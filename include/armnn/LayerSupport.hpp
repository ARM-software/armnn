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

bool IsActivationSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported = nullptr,
                           size_t reasonIfUnsupportedMaxLength = 1024);

bool IsAdditionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

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

bool IsConstantSupported(const BackendId& backend,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvertFp16ToFp32Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvertFp32ToFp16Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvolution2dSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

bool IsDepthwiseConvolutionSupported(const BackendId& backend,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsDivisionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSubtractionSupported(const BackendId& backend,
                            const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            char* reasonIfUnsupported = nullptr,
                            size_t reasonIfUnsupportedMaxLength = 1024);

bool IsInputSupported(const BackendId& backend,
                      const TensorInfo& input,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFullyConnectedSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsL2NormalizationSupported(const BackendId& backend,
                                const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                char* reasonIfUnsupported = nullptr,
                                size_t reasonIfUnsupportedMaxLength = 1024);

bool IsLstmSupported(const BackendId& backend, const TensorInfo& input, const TensorInfo& outputStateIn,
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

bool IsMergerSupported(const BackendId& backend,
                       const std::vector<const TensorInfo*> inputs,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMultiplicationSupported(const BackendId& backend,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsNormalizationSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

bool IsOutputSupported(const BackendId& backend,
                       const TensorInfo& output,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPermuteSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPooling2dSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported = nullptr,
                          size_t reasonIfUnsupportedMaxLength = 1024);

bool IsResizeBilinearSupported(const BackendId& backend,
                               const TensorInfo& input,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSoftmaxSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFakeQuantizationSupported(const BackendId& backend,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported = nullptr,
                                 size_t reasonIfUnsupportedMaxLength = 1024);

bool IsReshapeSupported(const BackendId& backend,
                        const TensorInfo& input,
                        char* reasonIfUnsupported = nullptr,
                        size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFloorSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMeanSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsPadSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const PadDescriptor& descriptor,
                     char* reasonIfUnsupported = nullptr,
                     size_t reasonIfUnsupportedMaxLength = 1024);

}
