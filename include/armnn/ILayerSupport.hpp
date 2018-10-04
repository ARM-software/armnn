//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/DescriptorsFwd.hpp>
#include <armnn/Optional.hpp>
#include <vector>
#include <cctype>

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
                                       char* reasonIfUnsupported = nullptr,
                                       size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsAdditionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsBatchNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TensorInfo& mean,
                                               const TensorInfo& var,
                                               const TensorInfo& beta,
                                               const TensorInfo& gamma,
                                               const BatchNormalizationDescriptor& descriptor,
                                               char* reasonIfUnsupported = nullptr,
                                               size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsConstantSupported(const TensorInfo& output,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              char* reasonIfUnsupported = nullptr,
                                              size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              char* reasonIfUnsupported = nullptr,
                                              size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsConvolution2dSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const Convolution2dDescriptor& descriptor,
                                          const TensorInfo& weights,
                                          const Optional<TensorInfo>& biases,
                                          char* reasonIfUnsupported = nullptr,
                                          size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const DepthwiseConvolution2dDescriptor& descriptor,
                                                 const TensorInfo& weights,
                                                 const Optional<TensorInfo>& biases,
                                                 char* reasonIfUnsupported = nullptr,
                                                 size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsDivisionSupported(const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const TensorInfo& output,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsSubtractionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        char* reasonIfUnsupported = nullptr,
                                        size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsInputSupported(const TensorInfo& input,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsFullyConnectedSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const TensorInfo& weights,
                                           const TensorInfo& biases,
                                           const FullyConnectedDescriptor& descriptor,
                                           char* reasonIfUnsupported = nullptr,
                                           size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsL2NormalizationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const L2NormalizationDescriptor& descriptor,
                                            char* reasonIfUnsupported = nullptr,
                                            size_t reasonIfUnsupportedMaxLength = 1024) const;

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
                                 char* reasonIfUnsupported = nullptr,
                                 size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                   const OriginsDescriptor& descriptor,
                                   char* reasonIfUnsupported = nullptr,
                                   size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsMultiplicationSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           char* reasonIfUnsupported = nullptr,
                                           size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsNormalizationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const NormalizationDescriptor& descriptor,
                                          char* reasonIfUnsupported = nullptr,
                                          size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsOutputSupported(const TensorInfo& output,
                                   char* reasonIfUnsupported = nullptr,
                                   size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsPermuteSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const PermuteDescriptor& descriptor,
                                    char* reasonIfUnsupported = nullptr,
                                    size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsPooling2dSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const Pooling2dDescriptor& descriptor,
                                      char* reasonIfUnsupported = nullptr,
                                      size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsResizeBilinearSupported(const TensorInfo& input,
                                           char* reasonIfUnsupported = nullptr,
                                           size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsSoftmaxSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const SoftmaxDescriptor& descriptor,
                                    char* reasonIfUnsupported = nullptr,
                                    size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsSplitterSupported(const TensorInfo& input,
                                     const ViewsDescriptor& descriptor,
                                     char* reasonIfUnsupported = nullptr,
                                     size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsFakeQuantizationSupported(const TensorInfo& input,
                                             const FakeQuantizationDescriptor& descriptor,
                                             char* reasonIfUnsupported = nullptr,
                                             size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsReshapeSupported(const TensorInfo& input,
                                    char* reasonIfUnsupported = nullptr,
                                    size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsFloorSupported(const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported = nullptr,
                                  size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsMeanSupported(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const MeanDescriptor& descriptor,
                                 char* reasonIfUnsupported = nullptr,
                                 size_t reasonIfUnsupportedMaxLength = 1024) const;

    virtual bool IsPadSupported(const TensorInfo& input,
                                const TensorInfo& output,
                                const PadDescriptor& descriptor,
                                char* reasonIfUnsupported = nullptr,
                                size_t reasonIfUnsupportedMaxLength = 1024) const;

}; // class ILayerSupport

} // namespace armnn
