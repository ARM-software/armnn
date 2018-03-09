//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "armnn/LayerSupport.hpp"

#include "backends/RefLayerSupport.hpp"
#include "backends/NeonLayerSupport.hpp"
#include "backends/ClLayerSupport.hpp"

#include <boost/assert.hpp>

#include <cstring>
#include <algorithm>

namespace armnn
{

// Helper function to copy a full string to a truncated version
void CopyErrorMessage(char* truncatedString, const char* fullString, size_t maxLength)
{
    if(truncatedString != nullptr)
    {
        size_t copyLength = std::min(maxLength, strlen(fullString));
        std::strncpy(truncatedString, fullString, copyLength);
        // Ensure null-terminated string
        truncatedString[copyLength] = '\0';
    }
}

// Helper macro to avoid code duplication.
// Forwards function func to funcRef, funcNeon or funcCl, depending on the value of compute
#define FORWARD_LAYER_SUPPORT_FUNC(compute, func, ...) \
    std::string reasonIfUnsupportedFull; \
    bool isSupported; \
    switch(compute) \
    { \
        case Compute::CpuRef: \
            isSupported = func##Ref(__VA_ARGS__, &reasonIfUnsupportedFull); \
            break; \
        case Compute::CpuAcc: \
            isSupported = func##Neon(__VA_ARGS__, &reasonIfUnsupportedFull); \
            break; \
        case Compute::GpuAcc: \
            isSupported = func##Cl(__VA_ARGS__, &reasonIfUnsupportedFull); \
            break; \
        default: \
            isSupported = func##Ref(__VA_ARGS__, &reasonIfUnsupportedFull); \
            break; \
    } \
    CopyErrorMessage(reasonIfUnsupported, reasonIfUnsupportedFull.c_str(), reasonIfUnsupportedMaxLength); \
    return isSupported;

bool CheckTensorDataTypesEqual(const TensorInfo& input0, const TensorInfo& input1)
{
    return input0.GetDataType() == input1.GetDataType();
}

bool IsActivationSupported(Compute compute,
                           const TensorInfo& input,
                           const ActivationDescriptor& descriptor,
                           char* reasonIfUnsupported,
                           size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsActivationSupported, input, descriptor);
}

bool IsAdditionSupported(Compute compute,
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

    FORWARD_LAYER_SUPPORT_FUNC(compute, IsAdditionSupported, input0, input1, output);
}

bool IsBatchNormalizationSupported(Compute compute,
                                   const TensorInfo& input,
                                   const BatchNormalizationDescriptor& descriptor,
                                   char* reasonIfUnsupported,
                                   size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsBatchNormalizationSupported, input, descriptor);
}

bool IsConstantSupported(Compute compute,
                         const TensorInfo& output,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsConstantSupported, output);
}

bool IsConvolution2dSupported(Compute compute,
                              const TensorInfo& input,
                              const Convolution2dDescriptor& descriptor,
                              const TensorInfo& weights,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsConvolution2dSupported, input, descriptor, weights);
}

bool IsDepthwiseConvolutionSupported(Compute compute,
                                     const TensorInfo& input,
                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                     const TensorInfo& weights,
                                     char* reasonIfUnsupported,
                                     size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsDepthwiseConvolutionSupported, input, descriptor, weights);
}

bool IsInputSupported(Compute compute,
                      const TensorInfo& input,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsInputSupported, input);
}

bool IsFullyConnectedSupported(Compute compute,
                               const TensorInfo& input,
                               const FullyConnectedDescriptor& descriptor,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsFullyConnectedSupported, input, descriptor);
}

bool IsL2NormalizationSupported(Compute compute,
                                const TensorInfo& input,
                                char* reasonIfUnsupported,
                                size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsL2NormalizationSupported, input);
}

bool IsMergerSupported(Compute compute,
                       std::vector<const TensorInfo*> inputs,
                       const OriginsDescriptor& descriptor,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    BOOST_ASSERT(inputs.size() > 0);
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsMergerSupported, inputs, descriptor);
}

bool IsMultiplicationSupported(Compute compute,
                               const TensorInfo& input0,
                               const TensorInfo& input1,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsMultiplicationSupported, input0, input1);
}

bool IsNormalizationSupported(Compute compute,
                              const TensorInfo& input,
                              const TensorInfo& output,
                              const NormalizationDescriptor& descriptor,
                              char* reasonIfUnsupported,
                              size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsNormalizationSupported, input, output, descriptor);
}

bool IsOutputSupported(Compute compute,
                       const TensorInfo& output,
                       char* reasonIfUnsupported,
                       size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsOutputSupported, output);
}

bool IsPermuteSupported(Compute compute,
                        const TensorInfo& input,
                        const TensorInfo& output,
                        const PermuteDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsPermuteSupported, input, output, descriptor);
}

bool IsPooling2dSupported(Compute compute,
                          const TensorInfo& input,
                          const TensorInfo& output,
                          const Pooling2dDescriptor& descriptor,
                          char* reasonIfUnsupported,
                          size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsPooling2dSupported, input, output, descriptor);
}

bool IsResizeBilinearSupported(Compute compute,
                               const TensorInfo& input,
                               char* reasonIfUnsupported,
                               size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsResizeBilinearSupported, input);
}

bool IsSoftmaxSupported(Compute compute,
                        const TensorInfo& input,
                        const SoftmaxDescriptor& descriptor,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsSoftmaxSupported, input, descriptor);
}

bool IsSplitterSupported(Compute compute,
                         const TensorInfo& input,
                         const ViewsDescriptor& descriptor,
                         char* reasonIfUnsupported,
                         size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsSplitterSupported, input, descriptor);
}

bool IsFakeQuantizationSupported(Compute compute,
                                 const TensorInfo& input,
                                 const FakeQuantizationDescriptor& descriptor,
                                 char* reasonIfUnsupported,
                                 size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsFakeQuantizationSupported, input, descriptor);
}

bool IsReshapeSupported(Compute compute,
                        const TensorInfo& input,
                        char* reasonIfUnsupported,
                        size_t reasonIfUnsupportedMaxLength)
{
    FORWARD_LAYER_SUPPORT_FUNC(compute, IsReshapeSupported, input);
}

bool IsFloorSupported(Compute compute,
                      const TensorInfo& input,
                      const TensorInfo& output,
                      char* reasonIfUnsupported,
                      size_t reasonIfUnsupportedMaxLength)
{
    // By definition (that is, regardless of compute device), shapes and data type must match
    if (input.GetShape() != output.GetShape() || input.GetDataType() != output.GetDataType())
    {
        return false;
    }

    FORWARD_LAYER_SUPPORT_FUNC(compute, IsFloorSupported, input, output);
}

}
