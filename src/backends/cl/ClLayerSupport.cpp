//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLayerSupport.hpp"

#include "InternalTypes.hpp"
#include "LayerSupportCommon.hpp"

#include <boost/core/ignore_unused.hpp>

#ifdef ARMCOMPUTECL_ENABLED
#include "workloads/ClAdditionWorkload.hpp"
#include "workloads/ClActivationWorkload.hpp"
#include "workloads/ClBatchNormalizationFloatWorkload.hpp"
#include "workloads/ClConvertFp16ToFp32Workload.hpp"
#include "workloads/ClConvertFp32ToFp16Workload.hpp"
#include "workloads/ClConvolution2dWorkload.hpp"
#include "workloads/ClDepthwiseConvolutionWorkload.hpp"
#include "workloads/ClDivisionFloatWorkload.hpp"
#include "workloads/ClFullyConnectedWorkload.hpp"
#include "workloads/ClL2NormalizationFloatWorkload.hpp"
#include "workloads/ClLstmFloatWorkload.hpp"
#include "workloads/ClMultiplicationWorkload.hpp"
#include "workloads/ClNormalizationFloatWorkload.hpp"
#include "workloads/ClPadWorkload.hpp"
#include "workloads/ClPermuteWorkload.hpp"
#include "workloads/ClPooling2dWorkload.hpp"
#include "workloads/ClSoftmaxBaseWorkload.hpp"
#include "workloads/ClSubtractionWorkload.hpp"
#endif

using namespace boost;

namespace armnn
{

bool ClLayerSupport::IsActivationSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const ActivationDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsActivationSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsAdditionSupportedCl(input0, input1, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   const TensorInfo& mean,
                                                   const TensorInfo& var,
                                                   const TensorInfo& beta,
                                                   const TensorInfo& gamma,
                                                   const BatchNormalizationDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsBatchNormalizationSupportedCl(input,
                                                  output,
                                                  mean,
                                                  var,
                                                  beta,
                                                  gamma,
                                                  descriptor,
                                                  reasonIfUnsupported);
}

bool ClLayerSupport::IsConstantSupported(const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConstantSupportedCl(output, reasonIfUnsupported);
}

bool ClLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp16ToFp32SupportedCl(input, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp32ToFp16SupportedCl(input, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const Convolution2dDescriptor& descriptor,
                                              const TensorInfo& weights,
                                              const Optional<TensorInfo>& biases,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvolution2dSupportedCl(input,
                                             output,
                                             descriptor,
                                             weights,
                                             biases,
                                             reasonIfUnsupported);
}

bool ClLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const DepthwiseConvolution2dDescriptor& descriptor,
                                                     const TensorInfo& weights,
                                                     const Optional<TensorInfo>& biases,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDepthwiseConvolutionSupportedCl(input,
                                                    output,
                                                    descriptor,
                                                    weights,
                                                    biases,
                                                    reasonIfUnsupported);
}

bool ClLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                         const TensorInfo& input1,
                                         const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDivisionSupportedCl(input0, input1, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                 const FakeQuantizationDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFakeQuantizationSupportedCl(input, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsFloorSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFloorSupportedCl(input, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const TensorInfo& weights,
                                               const TensorInfo& biases,
                                               const FullyConnectedDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFullyConnectedSupportedCl(input,
                                              output,
                                              weights,
                                              biases,
                                              descriptor,
                                              reasonIfUnsupported);
}

bool ClLayerSupport::IsInputSupported(const TensorInfo& input,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsInputSupportedCl(input, reasonIfUnsupported);
}

bool ClLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const L2NormalizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsL2NormalizationSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsLstmSupported(const TensorInfo& input,
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
    return armnn::IsLstmSupportedCl(input,
                                    outputStateIn,
                                    cellStateIn,
                                    scratchBuffer,
                                    outputStateOut,
                                    cellStateOut,
                                    output,
                                    descriptor,
                                    inputToForgetWeights,
                                    inputToCellWeights,
                                    inputToOutputWeights,
                                    recurrentToForgetWeights,
                                    recurrentToCellWeights,
                                    recurrentToOutputWeights,
                                    forgetGateBias,
                                    cellBias,
                                    outputGateBias,
                                    inputToInputWeights,
                                    recurrentToInputWeights,
                                    cellToInputWeights,
                                    inputGateBias,
                                    projectionWeights,
                                    projectionBias,
                                    cellToForgetWeights,
                                    cellToOutputWeights,
                                    reasonIfUnsupported);
}

bool ClLayerSupport::IsMeanSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const MeanDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMeanSupportedCl(input, output, descriptor,reasonIfUnsupported);
}

bool ClLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                       const OriginsDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMergerSupportedCl(inputs, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                               const TensorInfo& input1,
                                               const TensorInfo& output,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMultiplicationSupportedCl(input0, input1, output, reasonIfUnsupported);
}

bool ClLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const NormalizationDescriptor& descriptor,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsNormalizationSupportedCl(input,
                                             output,
                                             descriptor,
                                             reasonIfUnsupported);
}

bool ClLayerSupport::IsOutputSupported(const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsOutputSupportedCl(output, reasonIfUnsupported);
}

bool ClLayerSupport::IsPadSupported(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const PadDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPadSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const PermuteDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPermuteSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const Pooling2dDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPooling2dSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsReshapeSupportedCl(input, reasonIfUnsupported);
}

bool ClLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsResizeBilinearSupportedCl(input, reasonIfUnsupported);
}

bool ClLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const SoftmaxDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSoftmaxSupportedCl(input, output, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                         const ViewsDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSplitterSupportedCl(input, descriptor, reasonIfUnsupported);
}

bool ClLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                            const TensorInfo& input1,
                                            const TensorInfo& output,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSubtractionSupportedCl(input0, input1, output, reasonIfUnsupported);
}

//
// Implementation functions
//
// TODO: Functions kept for backward compatibility. Remove redundant functions
// once transition to plugable backends is complete.

namespace
{
template<unsigned int FilterSize>
bool IsMatchingSize2d(const TensorInfo& weightInfo)
{
    // Width & Height must match.
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

bool IsClBackendSupported(Optional<std::string&> reasonIfUnsupported)
{
#if ARMCOMPUTECL_ENABLED
    return true;
#else
    if (reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = "The armnn library has been built without CL support";
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
inline bool IsWorkloadSupported(FuncType&& func, Optional<std::string&> reasonIfUnsupported, Args&&... args)
{
    arm_compute::Status aclStatus = func(std::forward<Args>(args)...);
    const bool supported = (aclStatus.error_code() == arm_compute::ErrorCode::OK);
    if (!supported && reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = aclStatus.error_description();
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

template<typename FloatFunc, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeCl(Optional<std::string&> reasonIfUnsupported,
                              DataType dataType,
                              FloatFunc floatFuncPtr,
                              Uint8Func uint8FuncPtr,
                              Params&&... params)
{
    return IsClBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                      dataType,
                                      floatFuncPtr,
                                      floatFuncPtr,
                                      uint8FuncPtr,
                                      std::forward<Params>(params)...);
}

bool IsActivationSupportedCl(const TensorInfo& input,
                             const TensorInfo& output,
                             const ActivationDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClActivationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool IsAdditionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClAdditionValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsBatchNormalizationSupportedCl(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const TensorInfo& mean,
                                     const TensorInfo& var,
                                     const TensorInfo& beta,
                                     const TensorInfo& gamma,
                                     const BatchNormalizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClBatchNormalizationValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   mean,
                                   var,
                                   beta,
                                   gamma,
                                   descriptor);
}

bool IsConstantSupportedCl(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported)
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

    // 1x1 convolution with strides of 1,2,3.
    isSupported |= IsMatchingSize2d<1>(weightInfo) && ( strideIsOneOrTwoOrThree );

    // 3x3 convolution with strides of 1,2.
    isSupported |= IsMatchingSize2d<3>(weightInfo) && ( strideIsOneOrTwo );

    // 5x5 convolution with strides of 1,2
    isSupported |= IsMatchingSize2d<5>(weightInfo) && ( strideIsOneOrTwo );

    //Fall back to normal convolution for the asymmetric padding case.
    if (desc.m_PadLeft != desc.m_PadRight ||
        desc.m_PadTop != desc.m_PadBottom)
    {
        //Direct convolution does not support asymmetric padding yet.
        isSupported = false;
    }

    return isSupported;
}

bool IsDirectConvolution2dParamsSupportedCl(Optional<std::string&> reasonIfUnsupported,
                                            const Convolution2dDescriptor& parameters,
                                            const TensorInfo& weightInfo)
{
    ignore_unused(reasonIfUnsupported);
    return IsClDirectConvolution2dSupported(weightInfo, parameters);
}

bool IsConvolution2dSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const Convolution2dDescriptor& descriptor,
                                const TensorInfo& weights,
                                const Optional<TensorInfo>& biases,
                                Optional<std::string&> reasonIfUnsupported)
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
                                       const TensorInfo& output,
                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                       const TensorInfo& weights,
                                       const Optional<TensorInfo>& biases,
                                       Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool IsDivisionSupportedCl(const TensorInfo& input0,
                           const TensorInfo& input1,
                           const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClDivisionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsSubtractionSupportedCl(const TensorInfo& input0,
                              const TensorInfo& input1,
                              const TensorInfo& output,
                              Optional<std::string&> reasonIfUnsupported)
{

    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSubtractionValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsFullyConnectedSupportedCl(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const TensorInfo& weights,
                                 const TensorInfo& biases,
                                 const FullyConnectedDescriptor& descriptor,
                                 Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClFullyConnectedWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   weights,
                                   biases,
                                   descriptor);
}

bool IsInputSupportedCl(const TensorInfo& input,
                        Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsL2NormalizationSupportedCl(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const L2NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClL2NormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsMergerSupportedCl(const std::vector<const TensorInfo*> inputs,
                         const OriginsDescriptor& descriptor,
                         Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    inputs[0]->GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsMultiplicationSupportedCl(const TensorInfo& input0,
                                 const TensorInfo& input1,
                                 const TensorInfo& output,
                                 Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClMultiplicationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsNormalizationSupportedCl(const TensorInfo& input,
                                const TensorInfo& output,
                                const NormalizationDescriptor& descriptor,
                                Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClNormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsOutputSupportedCl(const TensorInfo& output,
                         Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    output.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsPermuteSupportedCl(const TensorInfo& input,
                          const TensorInfo& output,
                          const PermuteDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPermuteWorkloadValidate, reasonIfUnsupported, descriptor);
}

bool IsPooling2dSupportedCl(const TensorInfo& input,
                            const TensorInfo& output,
                            const Pooling2dDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsResizeBilinearSupportedCl(const TensorInfo& input,
                                 Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &FalseFuncU8<>);
}

bool IsSoftmaxSupportedCl(const TensorInfo& input,
                          const TensorInfo& output,
                          const SoftmaxDescriptor& descriptor,
                          Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClSoftmaxWorkloadValidate, reasonIfUnsupported, input, output);
}

bool IsSplitterSupportedCl(const TensorInfo& input,
                           const ViewsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeCl(reasonIfUnsupported,
                                    input.GetDataType(),
                                    &TrueFunc<>,
                                    &TrueFunc<>);
}

bool IsFakeQuantizationSupportedCl(const TensorInfo& input,
                                   const FakeQuantizationDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsReshapeSupportedCl(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool IsFloorSupportedCl(const TensorInfo& input,
                        const TensorInfo& output,
                        Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(output);
    return IsClBackendSupported(reasonIfUnsupported) &&
           IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         input.GetDataType(),
                                         &FalseFuncF16<>,
                                         &TrueFunc<>,
                                         &FalseFuncU8<>);
}

bool IsLstmSupportedCl(const TensorInfo& input,
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
                       Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClLstmFloatWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   outputStateIn,
                                   cellStateIn,
                                   scratchBuffer,
                                   outputStateOut,
                                   cellStateOut,
                                   output,
                                   descriptor,
                                   inputToForgetWeights,
                                   inputToCellWeights,
                                   inputToOutputWeights,
                                   recurrentToForgetWeights,
                                   recurrentToCellWeights,
                                   recurrentToOutputWeights,
                                   forgetGateBias,
                                   cellBias,
                                   outputGateBias,
                                   inputToInputWeights,
                                   recurrentToInputWeights,
                                   cellToInputWeights,
                                   inputGateBias,
                                   projectionWeights,
                                   projectionBias,
                                   cellToForgetWeights,
                                   cellToOutputWeights);
}

bool IsConvertFp16ToFp32SupportedCl(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvertFp16ToFp32WorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool IsConvertFp32ToFp16SupportedCl(const TensorInfo& input,
                                    const TensorInfo& output,
                                    Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClConvertFp32ToFp16WorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output);
}

bool IsMeanSupportedCl(const TensorInfo& input,
                       const TensorInfo& output,
                       const MeanDescriptor& descriptor,
                       Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsPadSupportedCl(const TensorInfo& input,
                      const TensorInfo& output,
                      const PadDescriptor& descriptor,
                      Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(ClPadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

}
