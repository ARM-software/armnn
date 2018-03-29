//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonLayerSupport.hpp"

#include "LayerSupportCommon.hpp"
#include "InternalTypes.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/core/ignore_unused.hpp>

#ifdef ARMCOMPUTENEON_ENABLED
#include "NeonWorkloads/NeonPooling2dBaseWorkload.hpp"
#include "NeonWorkloads/NeonPermuteWorkload.hpp"
#endif

using namespace boost;

namespace armnn
{
bool IsNeonActivationUint8Supported(std::string* reasonIfUnsupported, const ActivationDescriptor& parameters)
{
    if (parameters.m_Function != ActivationFunction::BoundedReLu)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Unsupported activation function, only BoundedReLu is supported)";
        }

        return false;
    }

    return true;
}

bool IsNeonDirectConvolutionPreferred(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc)
{
    // See arm_compute::NEDirectConvolutionLayer documentation for the supported cases,
    // and complement with NEDirectConvolutionLayerKernel::configure() implementation

    // Only 1x1 is using direct convolution. Performance results and details are in:
    //    https://jira.arm.com/browse/IVGCVSW-1003
    // Measurements were taken as of clframework: f105ab972135bcd21304883eff040d7e587099bc

    const bool dataTypeSupported = (weightInfo.GetDataType() == armnn::DataType::Float32);

    // Strides: 1|2|3
    const bool strideSupported = (desc.m_StrideX == 1 || desc.m_StrideX == 2 || desc.m_StrideX == 3) &&
                                 (desc.m_StrideY == 1 || desc.m_StrideY == 2 || desc.m_StrideY == 3);

    auto paddingLargerThan = [](const Convolution2dDescriptor& desc, unsigned int value)
    {
        return desc.m_PadLeft > value || desc.m_PadRight > value || desc.m_PadTop > value || desc.m_PadBottom > value;
    };

    // Supported sizes and padding
    const bool sizeAndPaddingSupported =
        // Pad > 0 not supported for 1x1 weights
        (weightInfo.GetShape()[2] == 1 && weightInfo.GetShape()[3] == 1 && !paddingLargerThan(desc, 0u));

    const bool preferDirectConvolution = dataTypeSupported &&
                                         strideSupported &&
                                         sizeAndPaddingSupported &&
                                         // NEDirectConvolutionLayerKernel doesn't support NULL bias
                                         desc.m_BiasEnabled;
    return preferDirectConvolution;
}

bool IsNeonMultiplicationParamsSupported(std::string* reasonIfUnsupported,
                                         const TensorInfo& info0,
                                         const TensorInfo& info1)
{
    if (info0.GetShape() == info1.GetShape())
    {
        return true;
    }

    if (reasonIfUnsupported)
    {
        *reasonIfUnsupported = "Multiplication on Neon does not support implicit broadcast.";
    }
    return false;
}

bool IsNeonNormalizationDescParamsSupported(std::string* reasonIfUnsupported, const NormalizationDescriptor& parameters)
{
    if (parameters.m_NormMethodType != NormalizationAlgorithmMethod::LocalBrightness)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Unsupported normalisation method type, only LocalBrightness is supported";
        }
        return false;
    }
    if (parameters.m_NormSize % 2 == 0)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Normalization size must be an odd number.";
        }
        return false;
    }

    return true;
}

bool IsNeonBackendSupported(std::string* reasonIfUnsupported)
{
#if ARMCOMPUTENEON_ENABLED
    return true;
#else
    if (reasonIfUnsupported != nullptr)
    {
        *reasonIfUnsupported = "The armnn library has been built without NEON support";
    }
    return false;
#endif
}

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeNeon(std::string* reasonIfUnsupported,
                                DataType dataType,
                                Float32Func floatFuncPtr,
                                Uint8Func uint8FuncPtr,
                                Params&&... params)
{
    return IsNeonBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         std::forward<Params>(params)...);
}

#if ARMCOMPUTENEON_ENABLED
template<class FuncType, class... Args>
inline bool IsWorkloadSupported(FuncType& func, std::string* reasonIfUnsupported, Args&&... args)
{
    arm_compute::Status aclStatus = func(std::forward<Args>(args)...);
    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        *reasonIfUnsupported = aclStatus.error_description();
    }
    return supported;
}

#define FORWARD_WORKLOAD_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return IsWorkloadSupported(func, reasonIfUnsupported, __VA_ARGS__);
#else
#define FORWARD_WORKLOAD_VALIDATE_FUNC(func, reasonIfUnsupported, ...) \
    return IsNeonBackendSupported(reasonIfUnsupported);
#endif

bool IsActivationSupportedNeon(const TensorInfo& input,
                               const ActivationDescriptor& descriptor,
                               std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<const ActivationDescriptor&>,
                                      &IsNeonActivationUint8Supported,
                                      descriptor);
}

bool IsNeonDepthwiseConvolution2dDescParamsSupported(std::string* reasonIfUnsupported,
                                                     const DepthwiseConvolution2dDescriptor& parameters,
                                                     const TensorInfo& weights)
{
    ignore_unused(weights);

    if (parameters.m_StrideX < 1 || parameters.m_StrideX > 3)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "m_StrideX can only be 1, 2 or 3";
        }
        return false;
    }

    // weights.GetShape()[0] = channel multiplier
    if (weights.GetShape()[0] != 1)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Channel multiplier only supports the value 1 in the NEON backend";
        }
        return false;
    }

    if (parameters.m_PadLeft != parameters.m_PadRight || parameters.m_PadTop != parameters.m_PadBottom)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Asymmetric padding for depthwise convolution currently not supported "
                "in Neon backend";
        }
        return false;
    }

    return true;
}

bool IsAdditionSupportedNeon(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input0.GetDataType(),
                                      &TrueFunc<>,
                                      &FalseFuncU8<>);
}

bool IsBatchNormalizationSupportedNeon(const TensorInfo& input,
                                       const BatchNormalizationDescriptor& descriptor,
                                       std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &FalseFuncU8<>);
}

bool IsConstantSupportedNeon(const TensorInfo& output,
                             std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsConvolution2dSupportedNeon(const TensorInfo& input,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsDepthwiseConvolutionSupportedNeon(const TensorInfo& input,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &IsNeonDepthwiseConvolution2dDescParamsSupported,
                                      &IsNeonDepthwiseConvolution2dDescParamsSupported,
                                      descriptor,
                                      weights);
}

bool IsFullyConnectedSupportedNeon(const TensorInfo& input,
                                   const FullyConnectedDescriptor& descriptor,
                                   std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &FalseFuncU8<>);
}

bool IsInputSupportedNeon(const TensorInfo& input,
                          std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsL2NormalizationSupportedNeon(const TensorInfo& input,
                                    std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &FalseFunc<>);
}

bool IsMergerSupportedNeon(const std::vector<const TensorInfo*> inputs,
                           const OriginsDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      inputs[0]->GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsMultiplicationSupportedNeon(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input0.GetDataType(),
                                      &IsNeonMultiplicationParamsSupported,
                                      &FalseFuncU8<const TensorInfo&, const TensorInfo&>,
                                      input0,
                                      input1
                            );
}

bool IsNormalizationSupportedNeon(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &IsNeonNormalizationDescParamsSupported,
                                      &FalseFuncU8<const NormalizationDescriptor&>,
                                      descriptor);
}

bool IsOutputSupportedNeon(const TensorInfo& output,
                           std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsPermuteSupportedNeon(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            std::string* reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPermuteWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsPooling2dSupportedNeon(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              std::string* reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsResizeBilinearSupportedNeon(const TensorInfo& input,
                                   std::string* reasonIfUnsupported)
{
    ignore_unused(input);
    return false;
}

bool IsSoftmaxSupportedNeon(const TensorInfo& input,
                            const SoftmaxDescriptor& descriptor,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsSplitterSupportedNeon(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsFakeQuantizationSupportedNeon(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     std::string* reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(descriptor);
    return false;
}

bool IsReshapeSupportedNeon(const TensorInfo& input,
                            std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsFloorSupportedNeon(const TensorInfo& input,
                          const TensorInfo& output,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &FalseFuncU8<>);
}

}
