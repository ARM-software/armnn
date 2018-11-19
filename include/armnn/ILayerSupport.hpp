//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Optional.hpp>
#include <vector>
#include <cctype>
#include <memory>

namespace armnn
{

class TensorInfo;

class ILayerSupport
{
protected:
    ILayerSupport() {}
    virtual ~ILayerSupport() {}

public:
    virtual bool IsActivationSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const ActivationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsAdditionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsBatchNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TensorInfo& mean,
                                               const TensorInfo& var,
                                               const TensorInfo& beta,
                                               const TensorInfo& gamma,
                                               const BatchNormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsBatchToSpaceNdSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const BatchToSpaceNdDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsConstantSupported(const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsConvolution2dSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const Convolution2dDescriptor& descriptor,
                                          const TensorInfo& weights,
                                          const Optional<TensorInfo>& biases,
                                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const DepthwiseConvolution2dDescriptor& descriptor,
                                                 const TensorInfo& weights,
                                                 const Optional<TensorInfo>& biases,
                                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsDivisionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsFakeQuantizationSupported(const TensorInfo& input,
                                             const FakeQuantizationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsFloorSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsFullyConnectedSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TensorInfo& weights,
                                           const TensorInfo& biases,
                                           const FullyConnectedDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsInputSupported(const TensorInfo& input,
                                  Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsL2NormalizationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const L2NormalizationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsLstmSupported(const TensorInfo& input,
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
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsMeanSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const MeanDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                   const TensorInfo& output,
                                   const OriginsDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsMultiplicationSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsNormalizationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const NormalizationDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsOutputSupported(const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsPadSupported(const TensorInfo& input,
                                const TensorInfo& output,
                                const PadDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsPermuteSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const PermuteDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsPooling2dSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const Pooling2dDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsReshapeSupported(const TensorInfo& input,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsSpaceToBatchNdSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const SpaceToBatchNdDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsResizeBilinearSupported(const TensorInfo& input,
                                           Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsSoftmaxSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const SoftmaxDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsSplitterSupported(const TensorInfo& input,
                                     const ViewsDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsStridedSliceSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const StridedSliceDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;

    virtual bool IsSubtractionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported = EmptyOptional()) const;
}; // class ILayerSupport

using ILayerSupportSharedPtr = std::shared_ptr<ILayerSupport>;

} // namespace armnn
