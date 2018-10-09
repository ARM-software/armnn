//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLayerSupport.hpp"

#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>

#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/core/ignore_unused.hpp>

#ifdef ARMCOMPUTENEON_ENABLED
#include "workloads/NeonAdditionFloatWorkload.hpp"
#include "workloads/NeonActivationFloatWorkload.hpp"
#include "workloads/NeonBatchNormalizationFloatWorkload.hpp"
#include "workloads/NeonConvolution2dBaseWorkload.hpp"
#include "workloads/NeonDepthwiseConvolutionBaseWorkload.hpp"
#include "workloads/NeonL2NormalizationFloatWorkload.hpp"
#include "workloads/NeonMultiplicationFloatWorkload.hpp"
#include "workloads/NeonNormalizationFloatWorkload.hpp"
#include "workloads/NeonFullyConnectedWorkload.hpp"
#include "workloads/NeonPermuteWorkload.hpp"
#include "workloads/NeonPooling2dBaseWorkload.hpp"
#include "workloads/NeonSoftmaxBaseWorkload.hpp"
#include "workloads/NeonSubtractionFloatWorkload.hpp"
#endif

using namespace boost;

namespace armnn
{

bool NeonLayerSupport::IsActivationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ActivationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsActivationSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsAdditionSupportedNeon(input0, input1, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                     const TensorInfo& output,
                                                     const TensorInfo& mean,
                                                     const TensorInfo& var,
                                                     const TensorInfo& beta,
                                                     const TensorInfo& gamma,
                                                     const BatchNormalizationDescriptor& descriptor,
                                                     Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsBatchNormalizationSupportedNeon(input,
                                                    output,
                                                    mean,
                                                    var,
                                                    beta,
                                                    gamma,
                                                    descriptor,
                                                    reasonIfUnsupported);
}

bool NeonLayerSupport::IsConstantSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConstantSupportedNeon(output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp16ToFp32SupportedNeon(input, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp32ToFp16SupportedNeon(input, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvolution2dSupportedNeon(input,
                                               output,
                                               descriptor,
                                               weights,
                                               biases,
                                               reasonIfUnsupported);
}

bool NeonLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDepthwiseConvolutionSupportedNeon(input,
                                                      output,
                                                      descriptor,
                                                      weights,
                                                      biases,
                                                      reasonIfUnsupported);
}

bool NeonLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDivisionSupportedNeon(input0, input1, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                   const FakeQuantizationDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFakeQuantizationSupportedNeon(input, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsFloorSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFloorSupportedNeon(input, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const TensorInfo& weights,
                                                 const TensorInfo& biases,
                                                 const FullyConnectedDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFullyConnectedSupportedNeon(input,
                                                output,
                                                weights,
                                                biases,
                                                descriptor,
                                                reasonIfUnsupported);
}

bool NeonLayerSupport::IsInputSupported(const TensorInfo& input,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsInputSupportedNeon(input, reasonIfUnsupported);
}

bool NeonLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const L2NormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsL2NormalizationSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsLstmSupported(const TensorInfo& input,
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
    return armnn::IsLstmSupportedNeon(input,
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

bool NeonLayerSupport::IsMeanSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const MeanDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMeanSupportedNeon(input, output, descriptor,reasonIfUnsupported);
}

bool NeonLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                         const OriginsDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMergerSupportedNeon(inputs, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMultiplicationSupportedNeon(input0, input1, output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const NormalizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsNormalizationSupportedNeon(input,
                                               output,
                                               descriptor,
                                               reasonIfUnsupported);
}

bool NeonLayerSupport::IsOutputSupported(const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsOutputSupportedNeon(output, reasonIfUnsupported);
}

bool NeonLayerSupport::IsPadSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const PadDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPadSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const PermuteDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPermuteSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const Pooling2dDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPooling2dSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsReshapeSupportedNeon(input, reasonIfUnsupported);
}

bool NeonLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsResizeBilinearSupportedNeon(input, reasonIfUnsupported);
}

bool NeonLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SoftmaxDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSoftmaxSupportedNeon(input, output, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                           const ViewsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSplitterSupportedNeon(input, descriptor, reasonIfUnsupported);
}

bool NeonLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSubtractionSupportedNeon(input0, input1, output, reasonIfUnsupported);
}

//
// Implementation functions
//
// TODO: Functions kept for backward compatibility. Remove once transition to plugable backends is complete!

bool IsNeonDirectConvolutionPreferred(const TensorInfo& weightInfo, const Convolution2dDescriptor& desc)
{
    // See arm_compute::NEDirectConvolutionLayer documentation for the supported cases,
    // and complement with NEDirectConvolutionLayerKernel::configure() implementation.

    // Only 1x1 is using direct convolution. Performance results and details are in:
    //    https://jira.arm.com/browse/IVGCVSW-1003
    // Measurements were taken as of clframework: f105ab972135bcd21304883eff040d7e587099bc

    const bool dataTypeSupported = (weightInfo.GetDataType() == armnn::DataType::Float32);

    // Strides: 1|2|3
    const bool strideSupported = (desc.m_StrideX == 1 || desc.m_StrideX == 2 || desc.m_StrideX == 3) &&
                                 (desc.m_StrideY == 1 || desc.m_StrideY == 2 || desc.m_StrideY == 3);

    auto paddingLargerThan = [](const Convolution2dDescriptor& conv2ddesc, unsigned int value)
    {
        return conv2ddesc.m_PadLeft > value || conv2ddesc.m_PadRight > value ||
               conv2ddesc.m_PadTop > value || conv2ddesc.m_PadBottom > value;
    };

    // Supported sizes and padding.
    const bool sizeAndPaddingSupported =
        // Pad > 0 not supported for 1x1 weights.
        (weightInfo.GetShape()[2] == 1 && weightInfo.GetShape()[3] == 1 && !paddingLargerThan(desc, 0u));

    const bool preferDirectConvolution = dataTypeSupported &&
                                         strideSupported &&
                                         sizeAndPaddingSupported &&
                                         // NEDirectConvolutionLayerKernel doesn't support NULL bias.
                                         desc.m_BiasEnabled;
    return preferDirectConvolution;
}

bool IsNeonNormalizationDescParamsSupported(Optional<std::string&> reasonIfUnsupported,
                                            const NormalizationDescriptor& parameters)
{
    if (parameters.m_NormMethodType != NormalizationAlgorithmMethod::LocalBrightness)
    {
        if (reasonIfUnsupported)
        {
            reasonIfUnsupported.value() = "Unsupported normalisation method type, only LocalBrightness is supported";
        }
        return false;
    }
    if (parameters.m_NormSize % 2 == 0)
    {
        if (reasonIfUnsupported)
        {
            reasonIfUnsupported.value() = "Normalization size must be an odd number.";
        }
        return false;
    }

    return true;
}

bool IsNeonBackendSupported(Optional<std::string&> reasonIfUnsupported)
{
#if ARMCOMPUTENEON_ENABLED
    return true;
#else
    if (reasonIfUnsupported)
    {
        reasonIfUnsupported.value() = "The armnn library has been built without NEON support";
    }
    return false;
#endif
}

template<typename FloatFunc, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeNeon(Optional<std::string&> reasonIfUnsupported,
                                DataType dataType,
                                FloatFunc floatFuncPtr,
                                Uint8Func uint8FuncPtr,
                                Params&&... params)
{
    return IsNeonBackendSupported(reasonIfUnsupported) &&
        IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         floatFuncPtr,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         std::forward<Params>(params)...);
}

#if ARMCOMPUTENEON_ENABLED
template<class FuncType, class... Args>
inline bool IsWorkloadSupported(FuncType& func, Optional<std::string&> reasonIfUnsupported, Args&&... args)
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
    return IsNeonBackendSupported(reasonIfUnsupported);
#endif

bool IsActivationSupportedNeon(const TensorInfo& input,
                               const TensorInfo& output,
                               const ActivationDescriptor& descriptor,
                               Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonActivationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool IsAdditionSupportedNeon(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonAdditionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsBatchNormalizationSupportedNeon(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const TensorInfo& mean,
                                       const TensorInfo& var,
                                       const TensorInfo& beta,
                                       const TensorInfo& gamma,
                                       const BatchNormalizationDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonBatchNormalizationValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   mean,
                                   var,
                                   beta,
                                   gamma,
                                   descriptor);
}

bool IsConstantSupportedNeon(const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsConvolution2dSupportedNeon(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const Convolution2dDescriptor& descriptor,
                                  const TensorInfo& weights,
                                  const Optional<TensorInfo>& biases,
                                  Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool IsDepthwiseConvolutionSupportedNeon(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         const TensorInfo& weights,
                                         const Optional<TensorInfo>& biases,
                                         Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool IsDivisionSupportedNeon(const TensorInfo& input0,
                             const TensorInfo& input1,
                             const TensorInfo& output,
                             Optional<std::string&> reasonIfUnsupported)
{
    // At the moment division is not supported
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsSubtractionSupportedNeon(const TensorInfo& input0,
                                const TensorInfo& input1,
                                const TensorInfo& output,
                                Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSubtractionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsFullyConnectedSupportedNeon(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const TensorInfo& weights,
                                   const TensorInfo& biases,
                                   const FullyConnectedDescriptor& descriptor,
                                   Optional<std::string&> reasonIfUnsupported)
{
    // At the moment U8 is unsupported
    if (input.GetDataType() == DataType::QuantisedAsymm8)
    {
        return false;
    }
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonFullyConnectedWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   weights,
                                   biases,
                                   descriptor);
}

bool IsInputSupportedNeon(const TensorInfo& input,
                          Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsL2NormalizationSupportedNeon(const TensorInfo& input,
                                    const TensorInfo& output,
                                    const L2NormalizationDescriptor& descriptor,
                                    Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonL2NormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsMergerSupportedNeon(const std::vector<const TensorInfo*> inputs,
                           const OriginsDescriptor& descriptor,
                           Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      inputs[0]->GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsMultiplicationSupportedNeon(const TensorInfo& input0,
                                   const TensorInfo& input1,
                                   const TensorInfo& output,
                                   Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMultiplicationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool IsNormalizationSupportedNeon(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const NormalizationDescriptor& descriptor,
                                  Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonNormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsOutputSupportedNeon(const TensorInfo& output,
                           Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsPermuteSupportedNeon(const TensorInfo& input,
                            const TensorInfo& output,
                            const PermuteDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPermuteWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsPooling2dSupportedNeon(const TensorInfo& input,
                              const TensorInfo& output,
                              const Pooling2dDescriptor& descriptor,
                              Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsResizeBilinearSupportedNeon(const TensorInfo& input,
                                   Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsSoftmaxSupportedNeon(const TensorInfo& input,
                            const TensorInfo& output,
                            const SoftmaxDescriptor& descriptor,
                            Optional<std::string&> reasonIfUnsupported)
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSoftmaxWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool IsSplitterSupportedNeon(const TensorInfo& input,
                             const ViewsDescriptor& descriptor,
                             Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsFakeQuantizationSupportedNeon(const TensorInfo& input,
                                     const FakeQuantizationDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsReshapeSupportedNeon(const TensorInfo& input,
                            Optional<std::string&> reasonIfUnsupported)
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool IsFloorSupportedNeon(const TensorInfo& input,
                          const TensorInfo& output,
                          Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(output);
    return IsNeonBackendSupported(reasonIfUnsupported) &&
           IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         input.GetDataType(),
                                         &FalseFuncF16<>,
                                         &TrueFunc<>,
                                         &FalseFuncU8<>);
}

bool IsLstmSupportedNeon(const TensorInfo& input,
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
    ignore_unused(input);
    ignore_unused(outputStateIn);
    ignore_unused(cellStateIn);
    ignore_unused(scratchBuffer);
    ignore_unused(outputStateOut);
    ignore_unused(cellStateOut);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(inputToForgetWeights);
    ignore_unused(inputToCellWeights);
    ignore_unused(inputToOutputWeights);
    ignore_unused(recurrentToForgetWeights);
    ignore_unused(recurrentToCellWeights);
    ignore_unused(recurrentToOutputWeights);
    ignore_unused(forgetGateBias);
    ignore_unused(cellBias);
    ignore_unused(outputGateBias);
    ignore_unused(inputToInputWeights);
    ignore_unused(recurrentToInputWeights);
    ignore_unused(cellToInputWeights);
    ignore_unused(inputGateBias);
    ignore_unused(projectionWeights);
    ignore_unused(projectionBias);
    ignore_unused(cellToForgetWeights);
    ignore_unused(cellToOutputWeights);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool IsConvertFp16ToFp32SupportedNeon(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool IsConvertFp32ToFp16SupportedNeon(const TensorInfo& input,
                                      const TensorInfo& output,
                                      Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool IsMeanSupportedNeon(const TensorInfo& input,
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

bool IsPadSupportedNeon(const TensorInfo& input,
                        const TensorInfo& output,
                        const PadDescriptor& descriptor,
                        Optional<std::string&> reasonIfUnsupported)
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

}
