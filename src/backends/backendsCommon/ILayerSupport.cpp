//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/ILayerSupport.hpp>
#include <armnn/Exceptions.hpp>

namespace armnn
{

namespace
{

bool DefaultLayerSupport(const char* func,
                         const char* file,
                         unsigned int line,
                         Optional<std::string&> reasonIfUnsupported)
{
    // NOTE: We only need to return the reason if the optional parameter is not empty
    if (reasonIfUnsupported)
    {
        std::stringstream message;
        message << func << " is not implemented [" << file << ":" << line << "]";

        reasonIfUnsupported.value() = message.str();
    }

    return false;
}

} // anonymous namespace

bool ILayerSupport::IsActivationSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const ActivationDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const TensorInfo& mean,
                                                  const TensorInfo& var,
                                                  const TensorInfo& beta,
                                                  const TensorInfo& gamma,
                                                  const BatchNormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsBatchToSpaceNdSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const BatchToSpaceNdDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsConstantSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const Convolution2dDescriptor& descriptor,
                                             const TensorInfo& weights,
                                             const Optional<TensorInfo>& biases,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsDebugSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DebugDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const DepthwiseConvolution2dDescriptor& descriptor,
                                                    const TensorInfo& weights,
                                                    const Optional<TensorInfo>& biases,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                        const TensorInfo& input1,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsEqualSupported(const armnn::TensorInfo& input0,
                                     const armnn::TensorInfo& input1,
                                     const armnn::TensorInfo& output,
                                     armnn::Optional<std::string &> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                const FakeQuantizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsFloorSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const TensorInfo& weights,
                                              const TensorInfo& biases,
                                              const FullyConnectedDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsInputSupported(const TensorInfo& input,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const L2NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
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
                                    Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsMaximumSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsMeanSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const MeanDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                      const TensorInfo& output,
                                      const OriginsDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsMinimumSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const NormalizationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsOutputSupported(const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsPadSupported(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const PadDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsPermuteSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const PermuteDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const Pooling2dDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsReshapeSupported(const TensorInfo& input,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const SoftmaxDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsSpaceToBatchNdSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SpaceToBatchNdDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsSplitterSupported(const TensorInfo& input,
                                        const ViewsDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsStridedSliceSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const StridedSliceDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

bool ILayerSupport::IsGreaterSupported(const TensorInfo& input0,
                                       const TensorInfo& input1,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return DefaultLayerSupport(__func__, __FILE__, __LINE__, reasonIfUnsupported);
}

} // namespace armnn
