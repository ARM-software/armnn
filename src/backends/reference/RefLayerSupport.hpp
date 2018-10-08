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
                              Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsAdditionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsBatchNormalizationSupportedRef(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const TensorInfo& mean,
                                      const TensorInfo& var,
                                      const TensorInfo& beta,
                                      const TensorInfo& gamma,
                                      const BatchNormalizationDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsConstantSupportedRef(const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsConvolution2dSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const Optional<TensorInfo>& biases,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        const Optional<TensorInfo>& biases,
                                        Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsDivisionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsSubtractionSupportedRef(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsFullyConnectedSupportedRef(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const TensorInfo& weights,
                                  const TensorInfo& biases,
                                  const FullyConnectedDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsInputSupportedRef(const TensorInfo& input,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsL2NormalizationSupportedRef(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const L2NormalizationDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsLstmSupportedRef(const TensorInfo& input,
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
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsMergerSupportedRef(const std::vector<const TensorInfo*> inputs,
                          const OriginsDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsMultiplicationSupportedRef(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsNormalizationSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const NormalizationDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsOutputSupportedRef(const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsPermuteSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const PermuteDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsPooling2dSupportedRef(const TensorInfo& input,
                             const TensorInfo& output,
                             const Pooling2dDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsResizeBilinearSupportedRef(const TensorInfo& input,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsSoftmaxSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const SoftmaxDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsSplitterSupportedRef(const TensorInfo& input,
                            const ViewsDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsFakeQuantizationSupportedRef(const TensorInfo& input,
                                    const FakeQuantizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsReshapeSupportedRef(const TensorInfo& input,
                           Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsFloorSupportedRef(const TensorInfo& input,
                         const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsConvertFp16ToFp32SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsConvertFp32ToFp16SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsMeanSupportedRef(const TensorInfo& input,
                        const TensorInfo& output,
                        const MeanDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported = EmptyOptional());

bool IsPadSupportedRef(const TensorInfo& input,
                       const TensorInfo& output,
                       const PadDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported = EmptyOptional());

}
