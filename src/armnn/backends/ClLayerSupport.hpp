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
bool IsClDirectConvolution2dSupported(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc);
bool IsClActivationUint8Supported(std::string* reasonIfUnsupported, const ActivationDescriptor& parameters);
bool IsClDepthwiseConvolution2dDescParamsSupported(std::string* reasonIfUnsupported,
                                                   const DepthwiseConvolution2dDescriptor& parameters,
                                                   const TensorInfo& weights);

bool IsActivationSupportedCl(const TensorInfo& input,
                             const ActivationDescriptor& descriptor,
                             std::string* reasonIfUnsupported = nullptr);

bool IsAdditionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsBatchNormalizationSupportedCl(const TensorInfo& input,
                                     const BatchNormalizationDescriptor& descriptor,
                                     std::string* reasonIfUnsupported = nullptr);

bool IsConstantSupportedCl(const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsConvolution2dSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const Convolution2dDescriptor& descriptor,
                                const TensorInfo& weights,
                                const TensorInfo& biases,
                                std::string* reasonIfUnsupported = nullptr);

bool IsDepthwiseConvolutionSupportedCl(const TensorInfo& input,
                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                       const TensorInfo& weights,
                                       std::string* reasonIfUnsupported = nullptr);

bool IsFullyConnectedSupportedCl(const TensorInfo& input,
                                 const FullyConnectedDescriptor& descriptor,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsInputSupportedCl(const TensorInfo& input,
                        std::string* reasonIfUnsupported = nullptr);

bool IsL2NormalizationSupportedCl(const TensorInfo& input,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsMergerSupportedCl(const std::vector<const TensorInfo*> inputs,
                         const OriginsDescriptor& descriptor,
                         std::string* reasonIfUnsupported = nullptr);

bool IsMultiplicationSupportedCl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsNormalizationSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const NormalizationDescriptor& descriptor,
                                std::string* reasonIfUnsupported = nullptr);

bool IsOutputSupportedCl(const TensorInfo& output,
                         std::string* reasonIfUnsupported = nullptr);

bool IsPermuteSupportedCl(const TensorInfo& input,
                          const TensorInfo& output,
                          const PermuteDescriptor& descriptor,
                          std::string* reasonIfUnsupported = nullptr);

bool IsPooling2dSupportedCl(const TensorInfo& input,
                            const TensorInfo& output,
                            const Pooling2dDescriptor& descriptor,
                            std::string* reasonIfUnsupported = nullptr);

bool IsResizeBilinearSupportedCl(const TensorInfo& input,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsSoftmaxSupportedCl(const TensorInfo& input,
                          const SoftmaxDescriptor& descriptor,
                          std::string* reasonIfUnsupported = nullptr);

bool IsSplitterSupportedCl(const TensorInfo& input,
                           const ViewsDescriptor& descriptor,
                           std::string* reasonIfUnsupported = nullptr);

bool IsFakeQuantizationSupportedCl(const TensorInfo& input,
                                   const FakeQuantizationDescriptor& descriptor,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsReshapeSupportedCl(const TensorInfo& input,
                          std::string* reasonIfUnsupported = nullptr);

bool IsFloorSupportedCl(const TensorInfo& input,
                        const TensorInfo& output,
                        std::string* reasonIfUnsupported = nullptr);
}
