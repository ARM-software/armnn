//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonLayerSupport.hpp"
#include "NeonBackendId.hpp"

#include <armnn/Descriptors.hpp>
#include <InternalTypes.hpp>
#include <LayerSupportCommon.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/Types.hpp>

#include <backendsCommon/BackendRegistry.hpp>

#include <boost/core/ignore_unused.hpp>

#ifdef ARMCOMPUTENEON_ENABLED
#include "workloads/NeonAdditionWorkload.hpp"
#include "workloads/NeonActivationWorkload.hpp"
#include "workloads/NeonBatchNormalizationFloatWorkload.hpp"
#include "workloads/NeonConvolution2dWorkload.hpp"
#include "workloads/NeonDepthwiseConvolutionWorkload.hpp"
#include "workloads/NeonL2NormalizationFloatWorkload.hpp"
#include "workloads/NeonMergerWorkload.hpp"
#include "workloads/NeonMultiplicationFloatWorkload.hpp"
#include "workloads/NeonNormalizationFloatWorkload.hpp"
#include "workloads/NeonFullyConnectedWorkload.hpp"
#include "workloads/NeonPermuteWorkload.hpp"
#include "workloads/NeonPooling2dWorkload.hpp"
#include "workloads/NeonSoftmaxBaseWorkload.hpp"
#include "workloads/NeonSubtractionFloatWorkload.hpp"
#endif

using namespace boost;

namespace armnn
{

namespace
{

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

} // anonymous namespace

bool NeonLayerSupport::IsActivationSupported(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ActivationDescriptor& descriptor,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonActivationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonAdditionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
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

bool NeonLayerSupport::IsConstantSupported(const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool NeonLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool NeonLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return true;
}

bool NeonLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const Convolution2dDescriptor& descriptor,
                                                const TensorInfo& weights,
                                                const Optional<TensorInfo>& biases,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonConvolution2dWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool NeonLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                       const TensorInfo& output,
                                                       const DepthwiseConvolution2dDescriptor& descriptor,
                                                       const TensorInfo& weights,
                                                       const Optional<TensorInfo>& biases,
                                                       Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonDepthwiseConvolutionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor,
                                   weights,
                                   biases);
}

bool NeonLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                           const TensorInfo& input1,
                                           const TensorInfo& output,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input0);
    ignore_unused(input1);
    ignore_unused(output);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NeonLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                   const FakeQuantizationDescriptor& descriptor,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NeonLayerSupport::IsFloorSupported(const TensorInfo& input,
                                        const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(output);
    return IsNeonBackendSupported(reasonIfUnsupported) &&
           IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         input.GetDataType(),
                                         &FalseFuncF16<>,
                                         &TrueFunc<>,
                                         &FalseFuncU8<>);
}

bool NeonLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const TensorInfo& weights,
                                                 const TensorInfo& biases,
                                                 const FullyConnectedDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonFullyConnectedWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   weights,
                                   biases,
                                   descriptor);
}

bool NeonLayerSupport::IsInputSupported(const TensorInfo& input,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool NeonLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const L2NormalizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonL2NormalizationWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
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

bool NeonLayerSupport::IsMeanSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       const MeanDescriptor& descriptor,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NeonLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                         const TensorInfo& output,
                                         const OriginsDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    if(descriptor.GetNumDimensions() - descriptor.GetConcatAxis() == 1)
    {
        FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMergerWorkloadValidate,
                                       reasonIfUnsupported,
                                       inputs,
                                       output,
                                       descriptor);
    }
    else
     {
         return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                           inputs[0]->GetDataType(),
                                           &TrueFunc<>,
                                           &TrueFunc<>);
      }
}

bool NeonLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                 const TensorInfo& input1,
                                                 const TensorInfo& output,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonMultiplicationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

bool NeonLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const NormalizationDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonNormalizationWorkloadValidate,
                                   reasonIfUnsupported,
                                   input,
                                   output,
                                   descriptor);
}

bool NeonLayerSupport::IsOutputSupported(const TensorInfo& output,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      output.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool NeonLayerSupport::IsPadSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const PadDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NeonLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const PermuteDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPermuteWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const Pooling2dDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonPooling2dWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool NeonLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(input);
    ignore_unused(reasonIfUnsupported);
    return false;
}

bool NeonLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                          const TensorInfo& output,
                                          const SoftmaxDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSoftmaxWorkloadValidate, reasonIfUnsupported, input, output, descriptor);
}

bool NeonLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                           const ViewsDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeNeon(reasonIfUnsupported,
                                      input.GetDataType(),
                                      &TrueFunc<>,
                                      &TrueFunc<>);
}

bool NeonLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                              const TensorInfo& input1,
                                              const TensorInfo& output,
                                              Optional<std::string&> reasonIfUnsupported) const
{
    FORWARD_WORKLOAD_VALIDATE_FUNC(NeonSubtractionWorkloadValidate,
                                   reasonIfUnsupported,
                                   input0,
                                   input1,
                                   output);
}

} // namespace armnn
