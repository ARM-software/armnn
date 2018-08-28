//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <layers/LstmLayer.hpp>
#include <boost/optional.hpp>

#include <boost/optional.hpp>

namespace armnn
{

bool IsActivationSupportedRef(const TensorInfo& input,
                              const TensorInfo& output,
                              const ActivationDescriptor& descriptor,
                              std::string* reasonIfUnsupported = nullptr);

bool IsAdditionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported = nullptr);

bool IsBatchNormalizationSupportedRef(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const TensorInfo& mean,
                                      const TensorInfo& var,
                                      const TensorInfo& beta,
                                      const TensorInfo& gamma,
                                      const BatchNormalizationDescriptor& descriptor,
                                      std::string* reasonIfUnsupported = nullptr);

bool IsConstantSupportedRef(const TensorInfo& output,
                            std::string* reasonIfUnsupported = nullptr);

bool IsConvolution2dSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const boost::optional<TensorInfo>& biases,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        const boost::optional<TensorInfo>& biases,
                                        std::string* reasonIfUnsupported = nullptr);

bool IsFullyConnectedSupportedRef(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const TensorInfo& weights,
                                  const TensorInfo& biases,
                                  const FullyConnectedDescriptor& descriptor,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsInputSupportedRef(const TensorInfo& input,
                         std::string* reasonIfUnsupported = nullptr);

bool IsL2NormalizationSupportedRef(const TensorInfo& input,
                                   const TensorInfo& output,
                                   std::string* reasonIfUnsupported = nullptr);

bool IsLstmSupportedRef(const TensorInfo& input, const TensorInfo& outputStateIn,
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

bool IsMergerSupportedRef(const std::vector<const TensorInfo*> inputs,
                          const OriginsDescriptor& descriptor,
                          std::string* reasonIfUnsupported = nullptr);

bool IsMultiplicationSupportedRef(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
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
                           const TensorInfo& output,
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

bool IsConvertFp16ToFp32SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     std::string* reasonIfUnsupported = nullptr);

bool IsConvertFp32ToFp16SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     std::string* reasonIfUnsupported = nullptr);

}
