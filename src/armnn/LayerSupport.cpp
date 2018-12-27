//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/LayerSupport.hpp>
#include <armnn/Optional.hpp>
#include <armnn/ILayerSupport.hpp>

#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/IBackendInternal.hpp>

#include <boost/assert.hpp>

#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <armnn/ArmNN.hpp>

namespace armnn
{

namespace
{
/// Helper function to copy a full string to a truncated version.
void CopyErrorMessage(char* truncatedString, const char* fullString, size_t maxLength)
{
    if(truncatedString != nullptr)
    {
        size_t copyLength = std::min(maxLength, strlen(fullString));
        std::strncpy(truncatedString, fullString, copyLength);
        // Ensure null-terminated string.
        truncatedString[copyLength] = '\0';
    }
}

}

// Helper macro to avoid code duplication.
// Forwards function func to funcRef, funcNeon or funcCl, depending on the value of compute.
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
    } catch (InvalidArgumentException e) { \
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
                      const DebugDescriptor& descriptor,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDebugSupported, input, output, descriptor);
}

bool IsDivisionSupported(const BackendId& backend,
                         const TensorInfo& input0,
                         const TensorInfo& input1,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDivisionSupported, input0, input1, output);
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

bool IsDepthwiseConvolutionSupported(const BackendId& backend,
                                     const TensorInfo& input,
                                     const TensorInfo& output,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     const Optional<TensorInfo>& biases,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsDepthwiseConvolutionSupported, input, output, descriptor, weights, biases);
}

bool IsInputSupported(const BackendId& backend,
                      const TensorInfo& input,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsInputSupported, input);
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
                     const TensorInfo& inputToForgetWeights, const TensorInfo& inputToCellWeights,
                     const TensorInfo& inputToOutputWeights, const TensorInfo& recurrentToForgetWeights,
                     const TensorInfo& recurrentToCellWeights, const TensorInfo& recurrentToOutputWeights,
                     const TensorInfo& forgetGateBias, const TensorInfo& cellBias,
                     const TensorInfo& outputGateBias, const TensorInfo* inputToInputWeights,
                     const TensorInfo* recurrentToInputWeights, const TensorInfo* cellToInputWeights,
                     const TensorInfo* inputGateBias, const TensorInfo* projectionWeights,
                     const TensorInfo* projectionBias, const TensorInfo* cellToForgetWeights,
                     const TensorInfo* cellToOutputWeights, char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)

{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsLstmSupported, input, outputStateIn, cellStateIn,
                               scratchBuffer, outputStateOut, cellStateOut,
                               output, descriptor, inputToForgetWeights, inputToCellWeights,
                               inputToOutputWeights, recurrentToForgetWeights,
                               recurrentToCellWeights, recurrentToOutputWeights,
                               forgetGateBias, cellBias, outputGateBias,
                               inputToInputWeights, recurrentToInputWeights,
                               cellToInputWeights, inputGateBias, projectionWeights,
                               projectionBias, cellToForgetWeights, cellToOutputWeights);
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

bool IsMergerSupported(const BackendId& backend,
                       std::vector<const TensorInfo*> inputs,
                       const TensorInfo& output,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    BOOST_ASSERT(inputs.size() > 0);
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMergerSupported, inputs, output, descriptor);
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

bool IsResizeBilinearSupported(const BackendId& backend,
                               const TensorInfo& input,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsResizeBilinearSupported, input);
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

bool IsSplitterSupported(const BackendId& backend,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsSplitterSupported, input, descriptor);
}

bool IsFakeQuantizationSupported(const BackendId& backend,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported,
                                 size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsFakeQuantizationSupported, input, descriptor);
}

bool IsReshapeSupported(const BackendId& backend,
                        const TensorInfo& input,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsReshapeSupported, input);
}

bool IsRsqrtSupported(const BackendId& backend,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsRsqrtSupported, input, output);
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

bool IsMeanSupported(const BackendId& backend,
                     const TensorInfo& input,
                     const TensorInfo& output,
                     const MeanDescriptor& descriptor,
                     char* reasonIfUnsupported,
                     size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMeanSupported, input, output, descriptor);
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

bool IsStridedSliceSupported(const BackendId& backend,
                             const TensorInfo& input,
                             const TensorInfo& output,
                             const StridedSliceDescriptor& descriptor,
                             char* reasonIfUnsupported,
                             size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsStridedSliceSupported, input, output, descriptor);
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

bool IsGreaterSupported(const BackendId& backend,
                        const TensorInfo& input0,
                        const TensorInfo& input1,
                        const TensorInfo& output,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsGreaterSupported, input0, input1, output);
}

bool IsEqualSupported(const BackendId& backend,
                      const TensorInfo& input0,
                      const TensorInfo& input1,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsEqualSupported, input0, input1, output);
}

}
