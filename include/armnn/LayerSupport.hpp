//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

bool IsActivationSupported(Compute compute,
                           const TensorInfo& input,
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
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported = nullptr,
                                   size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConstantSupported(Compute compute,
                         const TensorInfo& output,
                         char* reasonIfUnsupported = nullptr,
                         size_t reasonIfUnsupportedMaxLength = 1024);

bool IsConvolution2dSupported(Compute compute,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const TensorInfo& biases,
                              char* reasonIfUnsupported = nullptr,
                              size_t reasonIfUnsupportedMaxLength = 1024);

bool IsDepthwiseConvolutionSupported(Compute compute,
                                     const TensorInfo& input,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024);

bool IsInputSupported(Compute compute,
                      const TensorInfo& input,
                      char* reasonIfUnsupported = nullptr,
                      size_t reasonIfUnsupportedMaxLength = 1024);

bool IsFullyConnectedSupported(Compute compute,
                               const TensorInfo& input,const
                               FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported = nullptr,
                               size_t reasonIfUnsupportedMaxLength = 1024);

bool IsL2NormalizationSupported(Compute compute,
                                const TensorInfo& input,
                                char* reasonIfUnsupported = nullptr,
                                size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMergerSupported(Compute compute,
                       const std::vector<const TensorInfo*> inputs,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported = nullptr,
                       size_t reasonIfUnsupportedMaxLength = 1024);

bool IsMultiplicationSupported(Compute compute,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
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

}
