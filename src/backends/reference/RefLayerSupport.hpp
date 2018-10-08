//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>
#include <layers/LstmLayer.hpp>

namespace armnn
{

class RefLayerSupport : public ILayerSupport
{
public:
    bool IsActivationSupported(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsAdditionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsBatchNormalizationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConstantSupported(const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsConvolution2dSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsDivisionSupported(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFakeQuantizationSupported(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFloorSupported(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsFullyConnectedSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsInputSupported(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsL2NormalizationSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const L2NormalizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsLstmSupported(const TensorInfo& input,
                         const TensorInfo& outputStateIn,
                         const TensorInfo& cellStateIn,
                         const TensorInfo& scratchBuffer,
                         const TensorInfo& outputStateOut,
                         const TensorInfo& cellStateOut,
                         const TensorInfo& output,
                         const LstmDescriptor& descriptor,
                         const TensorInfo& inputToForgetWeights,
                         const TensorInfo& inputToCellWeights,
                         const TensorInfo& inputToOutputWeights,
                         const TensorInfo& recurrentToForgetWeights,
                         const TensorInfo& recurrentToCellWeights,
                         const TensorInfo& recurrentToOutputWeights,
                         const TensorInfo& forgetGateBias,
                         const TensorInfo& cellBias,
                         const TensorInfo& outputGateBias,
                         const TensorInfo* inputToInputWeights,
                         const TensorInfo* recurrentToInputWeights,
                         const TensorInfo* cellToInputWeights,
                         const TensorInfo* inputGateBias,
                         const TensorInfo* projectionWeights,
                         const TensorInfo* projectionBias,
                         const TensorInfo* cellToForgetWeights,
                         const TensorInfo* cellToOutputWeights,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMeanSupported(const TensorInfo& input,
                         const TensorInfo& output,
                         const MeanDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsMultiplicationSupported(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsNormalizationSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsOutputSupported(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPadSupported(const TensorInfo& input,
                        const TensorInfo& output,
                        const PadDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPermuteSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsPooling2dSupported(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsReshapeSupported(const TensorInfo& input,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsResizeBilinearSupported(const TensorInfo& input,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSoftmaxSupported(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSplitterSupported(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;

    bool IsSubtractionSupported(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const override;
};

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
                                 const Optional<TensorInfo>& biases,
                                 std::string* reasonIfUnsupported = nullptr);

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        const Optional<TensorInfo>& biases,
                                        std::string* reasonIfUnsupported = nullptr);

bool IsDivisionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported = nullptr);

bool IsSubtractionSupportedRef(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
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
                                   const L2NormalizationDescriptor& descriptor,
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

bool IsMeanSupportedRef(const TensorInfo& input,
                        const TensorInfo& output,
                        const MeanDescriptor& descriptor,
                        std::string* reasonIfUnsupported = nullptr);

bool IsPadSupportedRef(const TensorInfo& input,
                       const TensorInfo& output,
                       const PadDescriptor& descriptor,
                       std::string* reasonIfUnsupported = nullptr);

}
