//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <armnn/LayerSupport.hpp>
#include <armnn/Optional.hpp>

#include <backends/BackendRegistry.hpp>

#include <boost/assert.hpp>

#include <cstring>
#include <algorithm>
#include <unordered_map>

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

IBackend& GetBackend(const BackendId& id)
{
    static std::unordered_map<BackendId, IBackendUniquePtr> cachedBackends;
    auto it = cachedBackends.find(id);
    if (it == cachedBackends.end())
    {
        auto factoryFunc = BackendRegistry::Instance().GetFactory(id);
        auto emplaceResult =
            cachedBackends.emplace(
                std::make_pair(id, factoryFunc())
            );
        BOOST_ASSERT(emplaceResult.second);
        it = emplaceResult.first;
    }

    return *(it->second.get());
}

}

// Helper macro to avoid code duplication.
// Forwards function func to funcRef, funcNeon or funcCl, depending on the value of compute.
#define FORWARD_LAYER_SUPPORT_FUNC(backend, func, ...) \
    std::string reasonIfUnsupportedFull; \
    bool isSupported; \
    try { \
        auto const& layerSupportObject = GetBackend(backend).GetLayerSupport(); \
        isSupported = layerSupportObject.func(__VA_ARGS__, Optional<std::string&>(reasonIfUnsupportedFull)); \
        CopyErrorMessage(reasonIfUnsupported, reasonIfUnsupportedFull.c_str(), reasonIfUnsupportedMaxLength); \
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
bool IsMergerSupported(const BackendId& backend,
                       std::vector<const TensorInfo*> inputs,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    BOOST_ASSERT(inputs.size() > 0);
    FORWARD_LAYER_SUPPORT_FUNC(backend, IsMergerSupported, inputs, descriptor);
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

}
