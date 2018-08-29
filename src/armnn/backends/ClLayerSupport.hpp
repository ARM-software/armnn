//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/ArmNN.hpp>

#include <boost/optional.hpp>

namespace armnn
{
bool IsClDirectConvolution2dSupported(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc);
bool IsClDepthwiseConvolution2dDescParamsSupported(std::string* reasonIfUnsupported,
                                                   const DepthwiseConvolution2dDescriptor& parameters,
                                                   const TensorInfo& weights);

bool IsActivationSupportedCl(const TensorInfo& input,
                             const TensorInfo& output,
                             const ActivationDescriptor& descriptor,
                             std::string* reasonIfUnsupported = nullptr);

bool IsAdditionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsBatchNormalizationSupportedCl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const TensorInfo& mean,
                                     const TensorInfo& var,
                                     const TensorInfo& beta,
                                     const TensorInfo& gamma,
                                     const BatchNormalizationDescriptor& descriptor,
                                     std::string* reasonIfUnsupported = nullptr);

bool IsConstantSupportedCl(const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsConvolution2dSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const Convolution2dDescriptor& descriptor,
                                const TensorInfo& weights,
                                const boost::optional<TensorInfo>& biases,
                                std::string* reasonIfUnsupported = nullptr);

bool IsDepthwiseConvolutionSupportedCl(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                       const TensorInfo& weights,
                                       const boost::optional<TensorInfo>& biases,
                                       std::string* reasonIfUnsupported = nullptr);

bool IsDivisionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           std::string* reasonIfUnsupported = nullptr);

bool IsFullyConnectedSupportedCl(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const TensorInfo& weights,
                                 const TensorInfo& biases,
                                 const FullyConnectedDescriptor& descriptor,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsInputSupportedCl(const TensorInfo& input,
                        std::string* reasonIfUnsupported = nullptr);

bool IsL2NormalizationSupportedCl(const TensorInfo& input,
                                  const TensorInfo& output,
                                  std::string* reasonIfUnsupported = nullptr);

bool IsLstmSupportedCl(const TensorInfo& input, const TensorInfo& outputStateIn,
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

bool IsMergerSupportedCl(const std::vector<const TensorInfo*> inputs,
                         const OriginsDescriptor& descriptor,
                         std::string* reasonIfUnsupported = nullptr);

bool IsMultiplicationSupportedCl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 const TensorInfo& output,
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
                          const TensorInfo& output,
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

bool IsConvertFp16ToFp32SupportedCl(const TensorInfo& input,
                                    const TensorInfo& output,
                                    std::string* reasonIfUnsupported = nullptr);

bool IsConvertFp32ToFp16SupportedCl(const TensorInfo& input,
                                    const TensorInfo& output,
                                    std::string* reasonIfUnsupported = nullptr);

}
