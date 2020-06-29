//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/LayerSupport.hpp>
#include <armnn/Optional.hpp>
#include <armnn/ILayerSupport.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <armnn/utility/Assert.hpp>

#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace
{

/// Helper function to copy a full string to a truncated version.
void CopyErrorMessage(char* truncatedString, const char* fullString, size_t maxLength)
{
    if(truncatedString != nullptr)
    {
        std::snprintf(truncatedString, maxLength, "%s", fullString);
    }
}

} // anonymous namespace

namespace armnn
{

// Helper macro to avoid code duplication.
// Forwards function func to funcRef, funcNeon or funcCl, depending on the value of backendId.
#define FORWARD_LAYER_SUPPORT_FUNC(backendId, func, ...) \
    std::string reasonIfUnsupportedFull; \
    bool isSupported; \
    try { \
        auto const& backendRegistry = BackendRegistryInstance(); \
        if (!backendRegistry.IsBackendRegistered(backendId)) \
        { \
            std::stringstream ss; \
            ss << __func__ << " is not supported on " << backendId << " because this backend is not registered."; \
            reasonIfUnsupportedFull = ss.str(); \
            isSupported = false; \
        } \
        else \
        { \
            auto factoryFunc = backendRegistry.GetFactory(backendId); \
            auto backendObject = factoryFunc(); \
            auto layerSupportObject = backendObject->GetLayerSupport(); \
            isSupported = layerSupportObject->func(__VA_ARGS__, Optional<std::string&>(reasonIfUnsupportedFull)); \
            CopyErrorMessage(reasonIfUnsupported, reasonIfUnsupportedFull.c_str(), reasonIfUnsupportedMaxLength); \
        } \
    } catch (const InvalidArgumentException &e) { \
        /* re-throwing with more context information */ \
        throw InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
    } \
    return isSupported;

bool CheckTensorDataTypesEqual(const TensorInfo& input0, const TensorInfo& input1)
{
    return input0.GetDataType() == input1.GetDataType();
}

bool IsActivationSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported,
                           size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsActivationSupported, input, output, descriptor);
}

bool IsAdditionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    if(!CheckTensorDataTypesEqual(input0, input1))
    {
        return false;
    }

    FORWARD_LAYER_SUPPORT_FUNC(backend, IsAdditionSupported, input0, input1, output);
}

bool IsArgMinMaxSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const ArgMinMaxDescriptor& descriptor,
                          char* reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsArgMinMaxSupported, input, output, descriptor);
}

bool IsBatchNormalizationSupported(const BackendId& backend,
                                   const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& mean,
                                   const TensorInfo& var,
                                   const TensorInfo& beta,
                                   const TensorInfo& gamma,
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported,
                                   size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsBatchNormalizationSupported,
                               input,
                               output,
                               mean,
                               var,
                               beta,
                               gamma,
                               descriptor);
}

bool IsBatchToSpaceNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const BatchToSpaceNdDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsBatchToSpaceNdSupported,
                               input,
                               output,
                               descriptor);
}

bool IsConcatSupported(const BackendId& backend,
                       std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    ARMNN_ASSERT(inputs.size() > 0);

    FORWARD_LAYER_SUPPORT_FUNC(backend, IsConcatSupported, inputs, output, descriptor);
}

bool IsConstantSupported(const BackendId& backend,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsConstantSupported, output);
}

bool IsConvertFp16ToFp32Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported,
                                  size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsConvertFp16ToFp32Supported, input, output);
}

bool IsConvertFp32ToFp16Supported(const BackendId& backend,
                                  const TensorInfo& input,
                                  const TensorInfo& output,
                                  char* reasonIfUnsupported,
                                  size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsConvertFp32ToFp16Supported, input, output);
}

bool IsConvolution2dSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              const Optional<TensorInfo>& biases,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsConvolution2dSupported, input, output, descriptor, weights, biases);
}

bool IsDebugSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDebugSupported, input, output);
}

bool IsDepthwiseConvolutionSupported(const BackendId& backend,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength)
{
    if (descriptor.m_DilationX == 1 && descriptor.m_DilationY == 1)
    {
        // Pre 19.05 ArmNN did not have the dilation parameters.
        // This version of IsDepthwiseConvolutionSupported is called for backwards-compatibility
        FORWARD_LAYER_SUPPORT_FUNC(backend,
                                   IsDepthwiseConvolutionSupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
    }
    else
    {
        FORWARD_LAYER_SUPPORT_FUNC(backend,
                                   IsDilatedDepthwiseConvolutionSupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
    }
}

bool IsDequantizeSupported(const BackendId& backend,
                           const TensorInfo& input,
                           const TensorInfo& output,
                           char* reasonIfUnsupported,
                           size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDequantizeSupported, input, output);
}

bool IsDetectionPostProcessSupported(const BackendId& backend,
                                     const TensorInfo& input0,
                                     const TensorInfo& input1,
                                     const DetectionPostProcessDescriptor& descriptor,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength);

bool IsDivisionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDivisionSupported, input0, input1, output);
}

bool IsEqualSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsComparisonSupported,
                               input0,
                               input1,
                               output,
                               ComparisonDescriptor(ComparisonOperation::Equal));
}

bool IsFakeQuantizationSupported(const BackendId& backend,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported,
                                 size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsFakeQuantizationSupported, input, descriptor);
}

bool IsFloorSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    // By definition (that is, regardless of compute device), shapes and data type must match.
    if (input.GetShape() != output.GetShape() || input.GetDataType() != output.GetDataType())
    {
        return false;
    }

    FORWARD_LAYER_SUPPORT_FUNC(backend, IsFloorSupported, input, output);
}
bool IsFullyConnectedSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const TensorInfo& weights,
                               const TensorInfo& biases,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsFullyConnectedSupported, input, output, weights, biases, descriptor);
}

ARMNN_DEPRECATED_MSG("Use IsGatherSupported with descriptor instead")
bool IsGatherSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    const GatherDescriptor descriptor{};
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsGatherSupported, input0, input1, output, descriptor);
}

bool IsGatherSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output,
                       const GatherDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsGatherSupported, input0, input1, output, descriptor);
}

bool IsGreaterSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsComparisonSupported,
                               input0,
                               input1,
                               output,
                               ComparisonDescriptor(ComparisonOperation::Greater));
}

bool IsInputSupported(const BackendId& backend,
                      const TensorInfo& input,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsInputSupported, input);
}


bool IsL2NormalizationSupported(const BackendId& backend,
                                const TensorInfo& input,
                                const TensorInfo& output,
                                const L2NormalizationDescriptor& descriptor,
                                char* reasonIfUnsupported,
                                size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsL2NormalizationSupported, input, output, descriptor);
}

bool IsLstmSupported(const BackendId& backend, const TensorInfo& input, const TensorInfo& outputStateIn,
                     const TensorInfo& cellStateIn, const TensorInfo& scratchBuffer,
                     const TensorInfo& outputStateOut, const TensorInfo& cellStateOut,
                     const TensorInfo& output, const LstmDescriptor& descriptor,
                     const LstmInputParamsInfo& paramsInfo, char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)

{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsLstmSupported, input, outputStateIn, cellStateIn,
                               scratchBuffer, outputStateOut, cellStateOut,
                               output, descriptor, paramsInfo);
}

bool IsMaximumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMaximumSupported, input0, input1, output);
}

bool IsMeanSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMeanSupported, input, output, descriptor);
}

bool IsMemCopySupported(const BackendId &backend,
                        const TensorInfo &input,
                        const TensorInfo &output,
                        char *reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMemCopySupported, input, output);
}

bool IsMemImportSupported(const BackendId &backend,
                          const TensorInfo &input,
                          const TensorInfo &output,
                          char *reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMemImportSupported, input, output);
}

bool IsMergeSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMergeSupported, input0, input1, output);
}

ARMNN_DEPRECATED_MSG("Use IsConcatSupported instead")
bool IsMergerSupported(const BackendId& backend,
                       std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    ARMNN_ASSERT(inputs.size() > 0);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMergerSupported, inputs, output, descriptor);
    ARMNN_NO_DEPRECATE_WARN_END
}

bool IsMinimumSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMinimumSupported, input0, input1, output);
}

bool IsMultiplicationSupported(const BackendId& backend,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMultiplicationSupported, input0, input1, output);
}

bool IsNormalizationSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsNormalizationSupported, input, output, descriptor);
}

bool IsOutputSupported(const BackendId& backend,
                       const TensorInfo& output,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsOutputSupported, output);
}

bool IsPadSupported(const BackendId& backend,
                    const TensorInfo& input,
                    const TensorInfo& output,
                    const PadDescriptor& descriptor,
                    char* reasonIfUnsupported,
                    size_t reasonIfUnsupportedMaxLength)
{

    FORWARD_LAYER_SUPPORT_FUNC(backend, IsPadSupported, input, output, descriptor);
}

bool IsQuantizeSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsQuantizeSupported, input, output);
}

bool IsQLstmSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& previousOutputIn,
                      const TensorInfo& previousCellStateIn,
                      const TensorInfo& outputStateOut,
                      const TensorInfo& cellStateOut,
                      const TensorInfo& output,
                      const QLstmDescriptor& descriptor,
                      const LstmInputParamsInfo& paramsInfo,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)

{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsQLstmSupported, input, previousOutputIn, previousCellStateIn,
                               outputStateOut, cellStateOut, output, descriptor, paramsInfo);
}

bool IsQuantizedLstmSupported(const BackendId& backend,
                              const TensorInfo& input,
                              const TensorInfo& previousCellStateIn,
                              const TensorInfo& previousOutputIn,
                              const TensorInfo& cellStateOut,
                              const TensorInfo& output,
                              const QuantizedLstmInputParamsInfo& paramsInfo,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)

{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsQuantizedLstmSupported, input, previousCellStateIn, previousOutputIn,
                               cellStateOut, output, paramsInfo);
}

bool IsPermuteSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsPermuteSupported, input, output, descriptor);
}

bool IsPooling2dSupported(const BackendId& backend,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsPooling2dSupported, input, output, descriptor);
}

bool IsPreluSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& alpha,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsPreluSupported, input, alpha, output);
}

bool IsReshapeSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const ReshapeDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsReshapeSupported, input, output, descriptor);
}

bool IsResizeSupported(const BackendId& backend,
                       const TensorInfo& input,
                       const TensorInfo& output,
                       const ResizeDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsResizeSupported, input, output, descriptor);
}

ARMNN_DEPRECATED_MSG("Use IsResizeSupported instead")
bool IsResizeBilinearSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    ResizeDescriptor descriptor;
    descriptor.m_Method = ResizeMethod::Bilinear;

    const TensorShape& outputShape = output.GetShape();
    descriptor.m_TargetWidth  = outputShape[3];
    descriptor.m_TargetHeight = outputShape[2];

    FORWARD_LAYER_SUPPORT_FUNC(backend, IsResizeSupported, input, output, descriptor);
}

bool IsRsqrtSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend,
                               IsElementwiseUnarySupported,
                               input,
                               output,
                               ElementwiseUnaryDescriptor(UnaryOperation::Rsqrt));
}

bool IsSoftmaxSupported(const BackendId& backend,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSoftmaxSupported, input, output, descriptor);
}

bool IsSpaceToBatchNdSupported(const BackendId& backend,
                               const TensorInfo& input,
                               const TensorInfo& output,
                               const SpaceToBatchNdDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSpaceToBatchNdSupported, input, output, descriptor);
}

bool IsSpaceToDepthSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const SpaceToDepthDescriptor& descriptor,
                             char* reasonIfUnsupported,
                             size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSpaceToDepthSupported, input, output, descriptor);
}

ARMNN_DEPRECATED_MSG("Use IsSplitterSupported with outputs instead")
bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSplitterSupported, input, descriptor);
    ARMNN_NO_DEPRECATE_WARN_END
}

bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const std::vector<std::reference_wrapper<TensorInfo>>& outputs,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSplitterSupported, input, outputs, descriptor);
}

bool IsStridedSliceSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const StridedSliceDescriptor& descriptor,
                             char* reasonIfUnsupported,
                             size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsStridedSliceSupported, input, output, descriptor);
}

bool IsSubtractionSupported(const BackendId& backend,
                            const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            char* reasonIfUnsupported,
                            size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSubtractionSupported, input0, input1, output);
}

bool IsSwitchSupported(const BackendId& backend,
                       const TensorInfo& input0,
                       const TensorInfo& input1,
                       const TensorInfo& output0,
                       const TensorInfo& output1,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSwitchSupported, input0, input1, output0, output1);
}

} // namespace armnn
