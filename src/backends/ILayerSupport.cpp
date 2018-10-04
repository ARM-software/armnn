//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ILayerSupport.hpp>

namespace armnn
{

namespace
{

bool DefaultLayerSupport(const char* func,
                         const char* file,
                         unsigned int line,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    if (reasonIfUnsupported != nullptr && reasonIfUnsupportedMaxLength > 0)
    {
        snprintf(reasonIfUnsupported,
                 reasonIfUnsupportedMaxLength,
                 "%s is not supported [%s:%d]",
                 func,
                 file,
                 line);
    }
    return false;
}

}

bool ILayerSupport::IsActivationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const ActivationDescriptor& descriptor,
                                          char* reasonIfUnsupported,
                                          size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        char* reasonIfUnsupported,
                                        size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const TensorInfo& mean,
                                                  const TensorInfo& var,
                                                  const TensorInfo& beta,
                                                  const TensorInfo& gamma,
                                                  const BatchNormalizationDescriptor& descriptor,
                                                  char* reasonIfUnsupported,
                                                  size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsConstantSupported(const TensorInfo& output,
                                        char* reasonIfUnsupported,
                                        size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 char* reasonIfUnsupported,
                                                 size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 char* reasonIfUnsupported,
                                                 size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const Convolution2dDescriptor& descriptor,
                                             const TensorInfo& weights,
                                             const Optional<TensorInfo>& biases,
                                             char* reasonIfUnsupported,
                                             size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const DepthwiseConvolution2dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    char* reasonIfUnsupported,
                                                    size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        char* reasonIfUnsupported,
                                        size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           char* reasonIfUnsupported,
                                           size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsInputSupported(const TensorInfo& input,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TensorInfo& weights,
                                              const TensorInfo& biases,
                                              const FullyConnectedDescriptor& descriptor,
                                              char* reasonIfUnsupported,
                                              size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const L2NormalizationDescriptor& descriptor,
                                               char* reasonIfUnsupported,
                                               size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsLstmSupported(const TensorInfo& input,
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
                                    char* reasonIfUnsupported,
                                    size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                      const OriginsDescriptor& descriptor,
                                      char* reasonIfUnsupported,
                                      size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              char* reasonIfUnsupported,
                                              size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const NormalizationDescriptor& descriptor,
                                             char* reasonIfUnsupported,
                                             size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsOutputSupported(const TensorInfo& output,
                                      char* reasonIfUnsupported,
                                      size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsPermuteSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const PermuteDescriptor& descriptor,
                                       char* reasonIfUnsupported,
                                       size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const Pooling2dDescriptor& descriptor,
                                         char* reasonIfUnsupported,
                                         size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                              char* reasonIfUnsupported,
                                              size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SoftmaxDescriptor& descriptor,
                                       char* reasonIfUnsupported,
                                       size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsSplitterSupported(const TensorInfo& input,
                                        const ViewsDescriptor& descriptor,
                                        char* reasonIfUnsupported,
                                        size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                const FakeQuantizationDescriptor& descriptor,
                                                char* reasonIfUnsupported,
                                                size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsReshapeSupported(const TensorInfo& input,
                                       char* reasonIfUnsupported,
                                       size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsFloorSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsMeanSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const MeanDescriptor& descriptor,
                                    char* reasonIfUnsupported,
                                    size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

bool ILayerSupport::IsPadSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const PadDescriptor& descriptor,
                                   char* reasonIfUnsupported,
                                   size_t reasonIfUnsupportedMaxLength) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported, reasonIfUnsupportedMaxLength);
}

}
