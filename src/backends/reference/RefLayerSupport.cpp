//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayerSupportCommon.hpp"
#include "RefLayerSupport.hpp"
#include <armnn/Descriptors.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

#include <boost/core/ignore_unused.hpp>
#include "InternalTypes.hpp"

using namespace boost;

namespace armnn
{

namespace
{

std::string* GetReasonIfUnsupportedPtr(const Optional<std::string&>& reasonIfUnsupported)
{
    return reasonIfUnsupported ? &reasonIfUnsupported.value() : nullptr;
}

} // anonymous namespace

bool RefLayerSupport::IsActivationSupported(const TensorInfo& input,
                                            const TensorInfo& output,
                                            const ActivationDescriptor& descriptor,
                                            Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsActivationSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsAdditionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsAdditionSupportedRef(input0,
                                         input1,
                                         output,
                                         GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsBatchNormalizationSupported(const TensorInfo& input,
                                                    const TensorInfo& output,
                                                    const TensorInfo& mean,
                                                    const TensorInfo& var,
                                                    const TensorInfo& beta,
                                                    const TensorInfo& gamma,
                                                    const BatchNormalizationDescriptor& descriptor,
                                                    Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsBatchNormalizationSupportedRef(input,
                                                   output,
                                                   mean,
                                                   var,
                                                   beta,
                                                   gamma,
                                                   descriptor,
                                                   GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsConstantSupported(const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConstantSupportedRef(output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsConvertFp16ToFp32Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp16ToFp32SupportedRef(input, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsConvertFp32ToFp16Supported(const TensorInfo& input,
                                                   const TensorInfo& output,
                                                   Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvertFp32ToFp16SupportedRef(input, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsConvolution2dSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const Convolution2dDescriptor& descriptor,
                                               const TensorInfo& weights,
                                               const Optional<TensorInfo>& biases,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsConvolution2dSupportedRef(input,
                                              output,
                                              descriptor,
                                              weights,
                                              biases,
                                              GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsDepthwiseConvolutionSupported(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const DepthwiseConvolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDepthwiseConvolutionSupportedRef(input,
                                                     output,
                                                     descriptor,
                                                     weights,
                                                     biases,
                                                     GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsDivisionSupported(const TensorInfo& input0,
                                          const TensorInfo& input1,
                                          const TensorInfo& output,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsDivisionSupportedRef(input0, input1, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsFakeQuantizationSupported(const TensorInfo& input,
                                                  const FakeQuantizationDescriptor& descriptor,
                                                  Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFakeQuantizationSupportedRef(input, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsFloorSupported(const TensorInfo& input,
                                       const TensorInfo& output,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFloorSupportedRef(input, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsFullyConnectedSupported(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TensorInfo& weights,
                                                const TensorInfo& biases,
                                                const FullyConnectedDescriptor& descriptor,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsFullyConnectedSupportedRef(input,
                                               output,
                                               weights,
                                               biases,
                                               descriptor,
                                               GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsInputSupported(const TensorInfo& input,
                                       Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsInputSupportedRef(input, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsL2NormalizationSupported(const TensorInfo& input,
                                                 const TensorInfo& output,
                                                 const L2NormalizationDescriptor& descriptor,
                                                 Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsL2NormalizationSupportedRef(input,
                                                output,
                                                descriptor,
                                                GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsLstmSupported(const TensorInfo& input,
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
    return armnn::IsLstmSupportedRef(input,
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
                                     GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsMeanSupported(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const MeanDescriptor& descriptor,
                                      Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMeanSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsMergerSupported(const std::vector<const TensorInfo*> inputs,
                                        const OriginsDescriptor& descriptor,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMergerSupportedRef(inputs, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsMultiplicationSupported(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsMultiplicationSupportedRef(input0, input1, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsNormalizationSupported(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const NormalizationDescriptor& descriptor,
                                               Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsNormalizationSupportedRef(input,
                                              output,
                                              descriptor,
                                              GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsOutputSupported(const TensorInfo& output,
                                        Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsOutputSupportedRef(output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsPadSupported(const TensorInfo& input,
                                     const TensorInfo& output,
                                     const PadDescriptor& descriptor,
                                     Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPadSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsPermuteSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const PermuteDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPermuteSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsPooling2dSupported(const TensorInfo& input,
                                           const TensorInfo& output,
                                           const Pooling2dDescriptor& descriptor,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsPooling2dSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsReshapeSupported(const TensorInfo& input,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsReshapeSupportedRef(input, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsResizeBilinearSupported(const TensorInfo& input,
                                                Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsResizeBilinearSupportedRef(input, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsSoftmaxSupported(const TensorInfo& input,
                                         const TensorInfo& output,
                                         const SoftmaxDescriptor& descriptor,
                                         Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSoftmaxSupportedRef(input, output, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsSplitterSupported(const TensorInfo& input,
                                          const ViewsDescriptor& descriptor,
                                          Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSplitterSupportedRef(input, descriptor, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

bool RefLayerSupport::IsSubtractionSupported(const TensorInfo& input0,
                                             const TensorInfo& input1,
                                             const TensorInfo& output,
                                             Optional<std::string&> reasonIfUnsupported) const
{
    return armnn::IsSubtractionSupportedRef(input0, input1, output, GetReasonIfUnsupportedPtr(reasonIfUnsupported));
}

//
// Implementation functions
//
// TODO: Functions kept for backward compatibility. Remove once transition to plugable backends is complete!

template<typename Float32Func, typename Uint8Func, typename ... Params>
bool IsSupportedForDataTypeRef(std::string* reasonIfUnsupported,
                               DataType dataType,
                               Float32Func floatFuncPtr,
                               Uint8Func uint8FuncPtr,
                               Params&&... params)
{
    return IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                         dataType,
                                         &FalseFunc<Params...>,
                                         floatFuncPtr,
                                         uint8FuncPtr,
                                         std::forward<Params>(params)...);
}

bool IsActivationSupportedRef(const TensorInfo& input,
                              const TensorInfo& output,
                              const ActivationDescriptor& descriptor,
                              std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsAdditionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsBatchNormalizationSupportedRef(const TensorInfo& input,
                                      const TensorInfo& output,
                                      const TensorInfo& mean,
                                      const TensorInfo& var,
                                      const TensorInfo& beta,
                                      const TensorInfo& gamma,
                                      const BatchNormalizationDescriptor& descriptor,
                                      std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsConstantSupportedRef(const TensorInfo& output,
                            std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsConvolution2dSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const Convolution2dDescriptor& descriptor,
                                 const TensorInfo& weights,
                                 const Optional<TensorInfo>& biases,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    ignore_unused(output);
    ignore_unused(weights);
    ignore_unused(biases);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsDepthwiseConvolutionSupportedRef(const TensorInfo& input,
                                        const TensorInfo& output,
                                        const DepthwiseConvolution2dDescriptor& descriptor,
                                        const TensorInfo& weights,
                                        const Optional<TensorInfo>& biases,
                                        std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsDivisionSupportedRef(const TensorInfo& input0,
                            const TensorInfo& input1,
                            const TensorInfo& output,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsSubtractionSupportedRef(const TensorInfo& input0,
                               const TensorInfo& input1,
                               const TensorInfo& output,
                               std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFullyConnectedSupportedRef(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const TensorInfo& weights,
                                  const TensorInfo& biases,
                                  const FullyConnectedDescriptor& descriptor,
                                  std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    ignore_unused(weights);
    ignore_unused(biases);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsInputSupportedRef(const TensorInfo& input,
                         std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsL2NormalizationSupportedRef(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const L2NormalizationDescriptor& descriptor,
                                   std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsMergerSupportedRef(const std::vector<const TensorInfo*> inputs,
                          const OriginsDescriptor& descriptor,
                          std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     inputs[0]->GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsMultiplicationSupportedRef(const TensorInfo& input0,
                                  const TensorInfo& input1,
                                  const TensorInfo& output,
                                  std::string* reasonIfUnsupported)
{
    ignore_unused(input1);
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input0.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsNormalizationSupportedRef(const TensorInfo& input,
                                 const TensorInfo& output,
                                 const NormalizationDescriptor& descriptor,
                                 std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsOutputSupportedRef(const TensorInfo& output,
                          std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     output.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsPermuteSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const PermuteDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsPooling2dSupportedRef(const TensorInfo& input,
                             const TensorInfo& output,
                             const Pooling2dDescriptor& descriptor,
                             std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsResizeBilinearSupportedRef(const TensorInfo& input,
                                  std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsSoftmaxSupportedRef(const TensorInfo& input,
                           const TensorInfo& output,
                           const SoftmaxDescriptor& descriptor,
                           std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsSplitterSupportedRef(const TensorInfo& input,
                            const ViewsDescriptor& descriptor,
                            std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFakeQuantizationSupportedRef(const TensorInfo& input,
                                    const FakeQuantizationDescriptor& descriptor,
                                    std::string* reasonIfUnsupported)
{
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsReshapeSupportedRef(const TensorInfo& input,
                           std::string* reasonIfUnsupported)
{
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsFloorSupportedRef(const TensorInfo& input,
                         const TensorInfo& output,
                         std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &FalseFuncU8<>);
}

bool IsLstmSupportedRef(const TensorInfo& input, const TensorInfo& outputStateIn,
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
                        const TensorInfo* cellToOutputWeights, std::string* reasonIfUnsupported)
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
    return false;
}

bool IsConvertFp16ToFp32SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     std::string* reasonIfUnsupported)
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseInputFuncF32<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &FalseOutputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>));
}

bool IsConvertFp32ToFp16SupportedRef(const TensorInfo& input,
                                     const TensorInfo& output,
                                     std::string* reasonIfUnsupported)
{
    return (IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          input.GetDataType(),
                                          &FalseInputFuncF16<>,
                                          &TrueFunc<>,
                                          &FalseFuncU8<>) &&
            IsSupportedForDataTypeGeneric(reasonIfUnsupported,
                                          output.GetDataType(),
                                          &TrueFunc<>,
                                          &FalseOutputFuncF32<>,
                                          &FalseFuncU8<>));
}

bool IsMeanSupportedRef(const TensorInfo& input,
                        const TensorInfo& output,
                        const MeanDescriptor& descriptor,
                        std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return IsSupportedForDataTypeRef(reasonIfUnsupported,
                                     input.GetDataType(),
                                     &TrueFunc<>,
                                     &TrueFunc<>);
}

bool IsPadSupportedRef(const TensorInfo& input,
                       const TensorInfo& output,
                       const PadDescriptor& descriptor,
                       std::string* reasonIfUnsupported)
{
    ignore_unused(output);
    ignore_unused(descriptor);
    return false;
}

} // namespace armnn
