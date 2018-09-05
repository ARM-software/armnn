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
                                 const boost::optional<TensorInfo>& biases,
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
                                        const boost::optional<TensorInfo>& biases,
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
    // At the moment subtraction is not supported
    return false;
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
                                   std::string* reasonIfUnsupported)
{
    ignore_unused(output);
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

}
