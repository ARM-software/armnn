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

bool IsActivationSupportedRef(const TensorInfo& input,
                              const ActivationDescriptor& descriptor,
                              std::string* reasonIfUnsupported = nullptr);

bool IsAdditionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported = nullptr);

bool IsBatchNormalizationSupportedRef(const TensorInfo& input,
                                      const BatchNormalizationDescriptor& descriptor,
                                      std::string* reasonIfUnsupported = nullptr);

bool IsConstantSupportedRef(const TensorInfo& output,
                            std::string* reasonIfUnsupported = nullptr);

bool IsConvolution2dSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const TensorInfo& biases,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        std::string* reasonIfUnsupported = nullptr);

bool IsFullyConnectedSupportedRef(const TensorInfo& input,
                                  const FullyConnectedDescriptor& descriptor,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsInputSupportedRef(const TensorInfo& input,
                         std::string* reasonIfUnsupported = nullptr);

bool IsL2NormalizationSupportedRef(const TensorInfo& input,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsMergerSupportedRef(const std::vector<const TensorInfo*> inputs,
                          const OriginsDescriptor& descriptor,
                          std::string* reasonIfUnsupported = nullptr);

bool IsMultiplicationSupportedRef(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsNormalizationSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const NormalizationDescriptor& descriptor,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsOutputSupportedRef(const TensorInfo& output,
                          std::string* reasonIfUnsupported = nullptr);

bool IsPermuteSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const PermuteDescriptor& descriptor,
                           std::string* reasonIfUnsupported = nullptr);

bool IsPooling2dSupportedRef(const TensorInfo& input,
                             const TensorInfo& output,
                             const Pooling2dDescriptor& descriptor,
                             std::string* reasonIfUnsupported = nullptr);

bool IsResizeBilinearSupportedRef(const TensorInfo& input,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsSoftmaxSupportedRef(const TensorInfo& input,
                           const SoftmaxDescriptor& descriptor,
                           std::string* reasonIfUnsupported = nullptr);

bool IsSplitterSupportedRef(const TensorInfo& input,
                            const ViewsDescriptor& descriptor,
                            std::string* reasonIfUnsupported = nullptr);

bool IsFakeQuantizationSupportedRef(const TensorInfo& input,
                                    const FakeQuantizationDescriptor& descriptor,
                                    std::string* reasonIfUnsupported = nullptr);

bool IsReshapeSupportedRef(const TensorInfo& input,
                           std::string* reasonIfUnsupported = nullptr);

bool IsFloorSupportedRef(const TensorInfo& input,
                         const TensorInfo& output,
                         std::string* reasonIfUnsupported = nullptr);

}
