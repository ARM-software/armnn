//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/optional.hpp>

namespace armnn
{

bool IsNeonDirectConvolutionPreferred(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc);

bool IsNeonNormalizationDescParamsSupported(std::string* reasonIfUnsupported,
                                            const NormalizationDescriptor& parameters);

bool IsActivationSupportedNeon(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               std::string* reasonIfUnsupported);

bool IsNeonDepthwiseConvolution2dDescParamsSupported(std::string* reasonIfUnsupported,
                                                     const DepthwiseConvolution2dDescriptor& parameters,
                                                     const TensorInfo& weights);

bool IsAdditionSupportedNeon(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             std::string* reasonIfUnsupported);

bool IsBatchNormalizationSupportedNeon(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       std::string* reasonIfUnsupported = nullptr);

bool IsConstantSupportedNeon(const TensorInfo& output,
                             std::string* reasonIfUnsupported = nullptr);

bool IsConvolution2dSupportedNeon(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const boost::optional<TensorInfo>& biases,
                                  std::string* reasonIfUnsupported = nullptr);


bool IsDepthwiseConvolutionSupportedNeon(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const boost::optional<TensorInfo>& biases,
                                         std::string* reasonIfUnsupported = nullptr);

bool IsFullyConnectedSupportedNeon(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsInputSupportedNeon(const TensorInfo& input,
                          std::string* reasonIfUnsupported = nullptr);

bool IsL2NormalizationSupportedNeon(const TensorInfo& input,
                                    const TensorInfo& output,
                                    std::string* reasonIfUnsupported = nullptr);

bool IsMergerSupportedNeon(const std::vector<const TensorInfo*> inputs,
                           const OriginsDescriptor& descriptor,
                           std::string* reasonIfUnsupported = nullptr);

bool IsMultiplicationSupportedNeon(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsNormalizationSupportedNeon(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsOutputSupportedNeon(const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsPermuteSupportedNeon(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            std::string* reasonIfUnsupported = nullptr);

bool IsPooling2dSupportedNeon(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              std::string* reasonIfUnsupported = nullptr);

bool IsResizeBilinearSupportedNeon(const TensorInfo& input,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsSoftmaxSupportedNeon(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            std::string* reasonIfUnsupported = nullptr);

bool IsSplitterSupportedNeon(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             std::string* reasonIfUnsupported = nullptr);

bool IsFakeQuantizationSupportedNeon(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     std::string* reasonIfUnsupported = nullptr);

bool IsReshapeSupportedNeon(const TensorInfo& input,
                            std::string* reasonIfUnsupported = nullptr);

bool IsFloorSupportedNeon(const TensorInfo& input,
                          const TensorInfo& output,
                          std::string* reasonIfUnsupported = nullptr);

bool IsLstmSupportedNeon(const TensorInfo& input, const TensorInfo& outputStateIn,
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
                         const TensorInfo* cellToOutputWeights, std::string* reasonIfUnsupported = nullptr);

bool IsConvertFp16ToFp32SupportedNeon(const TensorInfo& input,
                                      const TensorInfo& output,
                                      std::string* reasonIfUnsupported = nullptr);

bool IsConvertFp32ToFp16SupportedNeon(const TensorInfo& input,
                                      const TensorInfo& output,
                                      std::string* reasonIfUnsupported = nullptr);

}
