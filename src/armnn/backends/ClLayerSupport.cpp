//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "LayerSupportCommon.hpp"

#include "ClLayerSupport.hpp"
#include "InternalTypes.hpp"

#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/core/ignore_unused.hpp>

#ifdef ARMCOMPUTECL_ENABLED
#include "ClWorkloads/ClAdditionFloat32Workload.hpp"
#include "ClWorkloads/ClConvolution2dBaseWorkload.hpp"
#include "ClWorkloads/ClPooling2dBaseWorkload.hpp"
#include "ClWorkloads/ClPermuteWorkload.hpp"
#include "ClWorkloads/ClNormalizationFloat32Workload.hpp"
#endif

using namespace boost;

namespace armnn
{
namespace
{
template<unsigned int FilterSize>
bool IsMatchingSize2d(const TensorInfo& weightInfo)
{
    // Width & Height must match
    return (weightInfo.GetShape()[3] == FilterSize) && (weightInfo.GetShape()[2] == FilterSize);
}

template<uint32_t ValidStride>
bool IsMatchingStride(uint32_t actualStride)
{
    return ValidStride == actualStride;
}

template<uint32_t FirstStride, uint32_t SecondStride, uint32_t... ValidStrides>
bool IsMatchingStride(uint32_t actualStride)
{
    return IsMatchingStride<FirstStride>(actualStride) || IsMatchingStride<SecondStride, ValidStrides...>(actualStride);
};

bool IsClBackendSupported(std::string* reasonIfUnsupported)
{
#if ARMCOMPUTECL_ENABLED
    return true;
#else
    if (reasonIfUnsupported != nullptr)
    {
        *reasonIfUnsupported = "The armnn library has been built without CL support";
    }
    return false;
#endif
}

#if ARMCOMPUTECL_ENABLED
#define FORWARD_CL_LAYER_SUPPORT_FUNC(expr) (expr)
#else
#define FORWARD_CL_LAYER_SUPPORT_FUNC(expr) IsClBackendSupported(reasonIfUnsupported)
#endif

#if ARMCOMPUTECL_ENABLED
template<class FuncType, class... Args>
inline bool IsWorkloadSupported(FuncType&& func, std::string* reasonIfUnsupported, Args&&... args)
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
    return IsClBackendSupported(reasonIfUnsupported);
#endif

} //namespace

bool IsClActivationUint8Supported(std::string* reasonIfUnsupported, const ActivationDescriptor& parameters)
{
    if (parameters.m_Function != ActivationFunction::BoundedReLu)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Unsupported activation function, only BoundedReLu is supported";
        }

        return false;
    }

    return true;
}

bool IsClDepthwiseConvolution2dDescParamsSupported(std::string* reasonIfUnsupported,
                                                   const DepthwiseConvolution2dDescriptor& parameters,
                                                   const TensorInfo& weights)
{
    if (weights.GetNumDimensions() != 4)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Depthwise convolution Weight tensor needs to be 4d";
        }
        return false;
    }
    // weights.GetShape()[0] = channel multiplier
    if (weights.GetShape()[0] != 1)
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "Channel multiplier only supports the value 1 in the CL backend";
        }
        return false;
    }
    else if ((weights.GetDataType() == armnn::DataType::QuantisedAsymm8) && !IsMatchingSize2d<3>(weights))
    {
        if (reasonIfUnsupported)
        {
            *reasonIfUnsupported = "CL backend only supports 3x3 filtering for Depthwise Convolution on 8-bit";
        }
        return false;
    }

    return true;
}

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeCl(std::string* reasonIfUnsupported,
                              DataType dataType,
                              Float32Func floatFuncPtr,
                              Uint8Func uint8FuncPtr,
                              Params&&... params)
{
    return IsClBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                      dataType,
                                      floatFuncPtr,
                                      uint8FuncPtr,
                                      std::forward<Params>(params)...);
}

bool IsActivationSupportedCl(const TensorInfo& input,
                             const ActivationDescriptor& descriptor,
                             std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<const ActivationDescriptor&>,
                                    &IsClActivationUint8Supported,
                                    descriptor);
}

bool IsAdditionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           std::string* reasonIfUnsupported)
{
    return FORWARD_CL_LAYER_SUPPORT_FUNC(ClAdditionFloat32Workload::IsSupported(input0,
        input1,
        output,
        reasonIfUnsupported));
}

bool IsBatchNormalizationSupportedCl(const TensorInfo& input,
                                     const BatchNormalizationDescriptor& descriptor,
                                     std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<const BatchNormalizationDescriptor&>,
                                    &FalseFuncU8<const BatchNormalizationDescriptor&>,
                                    descriptor);
}

bool IsConstantSupportedCl(const TensorInfo& output,
                           std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    output.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsClDirectConvolution2dSupported(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc)
{
    bool isSupported = false;

    bool strideXIsOneOrTwo = IsMatchingStride<1, 2>(desc.m_StrideX);
    bool strideXIsThree    = IsMatchingStride<3>(desc.m_StrideX);

    bool strideYIsOneOrTwo = IsMatchingStride<1, 2>(desc.m_StrideY);
    bool strideYIsThree    = IsMatchingStride<3>(desc.m_StrideY);

    bool strideIsOneOrTwo        = strideXIsOneOrTwo && strideYIsOneOrTwo;
    bool strideIsOneOrTwoOrThree = ( strideXIsOneOrTwo || strideXIsThree ) && ( strideYIsOneOrTwo || strideYIsThree );

    // 1x1 convolution with strides of 1,2,3
    isSupported |= IsMatchingSize2d<1>(weightInfo) && ( strideIsOneOrTwoOrThree );

    // 3x3 convolution with strides of 1,2
    isSupported |= IsMatchingSize2d<3>(weightInfo) && ( strideIsOneOrTwo );

    // 5x5 convolution with strides of 1,2
    isSupported |= IsMatchingSize2d<5>(weightInfo) && ( strideIsOneOrTwo );

    //fall back to normal convolution for the asymmetric padding case.
    if (desc.m_PadLeft != desc.m_PadRight ||
        desc.m_PadTop != desc.m_PadBottom)
    {
        //direct convolution does not support asymmetric padding yet.
        isSupported = false;
    }

    return isSupported;
}

bool IsDirectConvolution2dParamsSupportedCl(std::string* reasonIfUnsupported,
                                            const Convolution2dDescriptor& parameters,
                                            const TensorInfo& weightInfo)
{
    return IsClDirectConvolution2dSupported(weightInfo, parameters);
}

bool IsConvolution2dSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const Convolution2dDescriptor& descriptor,
                                const TensorInfo& weights,
                                const TensorInfo& biases,
                                std::string* reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool IsDepthwiseConvolutionSupportedCl(const TensorInfo& input,
                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                       const TensorInfo& weights,
                                       std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &IsClDepthwiseConvolution2dDescParamsSupported,
                                    &IsClDepthwiseConvolution2dDescParamsSupported,
                                    descriptor,
                                    weights);
}

bool IsFullyConnectedSupportedCl(const TensorInfo& input,
                                 const FullyConnectedDescriptor& descriptor,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsInputSupportedCl(const TensorInfo& input,
    std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsL2NormalizationSupportedCl(const TensorInfo& input,
                                  std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsMergerSupportedCl(const std::vector<const TensorInfo*> inputs,
                         const OriginsDescriptor& descriptor,
                         std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    inputs[0]->GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsMultiplicationSupportedCl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input0.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsNormalizationSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const NormalizationDescriptor& descriptor,
                                std::string* reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClNormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsOutputSupportedCl(const TensorInfo& output,
                         std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    output.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsPermuteSupportedCl(const TensorInfo& input,
                          const TensorInfo& output,
                          const PermuteDescriptor& descriptor,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPermuteWorkloadValidate, reasonIfUnsupported, descriptor);
}

bool IsPooling2dSupportedCl(const TensorInfo& input,
                            const TensorInfo& output,
                            const Pooling2dDescriptor& descriptor,
                            std::string* reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsResizeBilinearSupportedCl(const TensorInfo& input,
                                 std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsSoftmaxSupportedCl(const TensorInfo& input,
                          const SoftmaxDescriptor& descriptor,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsSplitterSupportedCl(const TensorInfo& input,
                           const ViewsDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsFakeQuantizationSupportedCl(const TensorInfo& input,
                                   const FakeQuantizationDescriptor& descriptor,
                                   std::string* reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(descriptor);
    return false;
}

bool IsReshapeSupportedCl(const TensorInfo& input,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(input);
    return true;
}

bool IsFloorSupportedCl(const TensorInfo& input,
                        const TensorInfo& output,
                        std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

}
